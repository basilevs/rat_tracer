"""A researcher's pass through a video, and everything it takes to make one.

The model behind the application: where the video is stopped, what the
detector says about the frame stopped on, and which frames have been marked as
failures. It is the whole review minus the parts that genuinely need Qt --
those stay in :mod:`rat_tracer.ui`, which drives this and gives it somewhere to
put the work that cannot be done inline.

The behaviour used to live in the widget, which put it in the one place
hardest to test; the split exists so a review can be driven without an event
loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from threading import Event
from time import time

from ultralytics import YOLO

from rat_tracer.background import BackgroundExecutor, InlineExecutor, Job
from rat_tracer.bad_frames import BadFrameStore, Detection, MarkRequest
from rat_tracer.frame_detector import FrameDetector
from rat_tracer.lib import model_path
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore, RenderOutcome
from rat_tracer.paint import presence_frames
from rat_tracer.progress_cache import load_progress, save_progress, video_key
from rat_tracer.review_modes import CoverageMode, ProblemReportMode, Retraction, ReviewMode

logger = getLogger(__name__)


@dataclass(frozen=True)
class ReviewListener:
    """What the review tells the UI, as plain callbacks.

    Deliberately not Qt signals: the review must stay drivable without an
    event loop. ``VideoMasker`` supplies callbacks that emit the real signals.

    Three, not one. ``changed`` is everything the UI can work out for itself by
    looking -- where the video is, what the controls should show, whether a
    repaint is due. A finished write cannot be worked out by looking: by the
    time anything repaints, a frame is simply stored or not, with no record of
    which write just landed or whether it succeeded. So the mark outcomes stay
    separate and carry the frame index the researcher is told about.
    """

    #: Something the UI displays has moved, and a render may be due with it.
    #: May arrive on the cumulative pass's thread.
    changed: Callable[[], None] = lambda: None
    #: A frame is safely on disk -- its index is shown in the confirmation.
    mark_stored: Callable[[int], None] = lambda _index: None
    #: A frame could *not* be stored; the researcher must be told.
    mark_failed: Callable[[int], None] = lambda _index: None


class VideoReview:
    """A video being reviewed frame by frame, with the detector's help.

    Hours of recording hold a handful of frames worth acting on, and reaching
    them is the whole task. This exposes the vocabulary that task is made of --
    :meth:`open_video`, :meth:`set_playing`, :meth:`seek`, :meth:`step` a
    single frame, look at what the detector found there, :meth:`toggle_mark`
    the ones it got wrong, :meth:`undo` -- so that every control in the UI is
    one call against it and no decision is left for the widget to re-derive.

    What is drawn over the video is the one thing that varies: the cumulative
    track the background pass has built up, or one frame's detection alone.
    Those are the two modes (:mod:`rat_tracer.review_modes`); keeping them
    consistent with playback and with each other is in service of the
    navigation above, not a purpose of its own.

    Nothing here imports Qt. The work that cannot be done inline is handed to a
    :class:`~rat_tracer.background.BackgroundExecutor` and answered back through
    it, so a test drives an entire review synchronously with
    :class:`~rat_tracer.background.InlineExecutor` -- no threads, no event loop,
    no model weights, no temp directories.

    **Threading.** Everything here belongs to one thread except
    :meth:`process_video`, which is meant to be run on another and touches only
    the lock-guarded coverage history and the listener.
    """

    def __init__(
        self,
        listener: ReviewListener | None = None,
        executor: BackgroundExecutor | None = None,
        detector: FrameDetector | None = None,
        store: BadFrameStore | None = None,
    ):
        self.listener = listener if listener is not None else ReviewListener()
        self._executor = executor if executor is not None else InlineExecutor()
        self._detector = detector
        self._store = store
        self._render = MaskRenderCore()
        self._coverage = CoverageMode(self._render.history)
        self._problem = ProblemReportMode(view=self._render, detector=detector, store=store)
        self._render.adopt_mode(self._coverage)
        #: The one outstanding detection, so a newer frame can abandon it.
        self._detection_job: tuple[int, Job] | None = None
        self._prewarmed = False
        #: Set by the pass's thread, consumed by the next render. Following the
        #: processed frontier has to be something the pass *did*, not something
        #: a render polls for -- polling would drag playback back to the
        #: frontier every time, and a seek made while playing would never stick.
        self._pass_advanced_since_render = Event()

    # --- the video ----------------------------------------------------------

    def open_video(self, cap: FrameCapture, video_path: Path) -> None:
        """Show *video_path*. Nothing may be marked until the pass names it."""
        self._render.open(cap, video_path)
        self.listener.changed()

    def close_video(self) -> None:
        self._abandon_detection()
        self._pass_advanced_since_render.clear()
        self._render.reset()
        self._problem.forget_video()
        self._select(self._coverage)
        self.listener.changed()

    @property
    def video_open(self) -> bool:
        return self._render.video_open

    # --- the cumulative pass ------------------------------------------------

    def process_video(self, is_interrupted: Callable[[], bool]) -> None:
        """Run the cumulative pass over the open video, until it ends or stops.

        Blocking, and the only method here meant to be called from another
        thread. It touches the coverage history, which is lock-guarded, and
        reports progress through the listener -- never the renderer, whose state
        belongs to the thread that navigates.

        *is_interrupted* is asked between frames. Saying yes saves what has been
        computed so far and returns, so reopening the video resumes rather than
        restarts -- which is also what the cache on disk is for.
        """
        path = self._render.video_path
        if path is None:
            return
        started = time()
        logger.info("Processing video: %s", path)
        # Fingerprinting reads the whole file, so it is paid here rather than by
        # whoever opened the video. Marks are stored under this key, so nothing
        # can be marked until it lands -- which is why the UI is told at once.
        key = video_key(path)
        self._render.identify(key)
        self.listener.changed()

        history = self._render.history
        loaded = load_progress(key)
        if loaded is not None:
            history.replace_with(loaded)
            self._pass_advanced()
        start_frame = len(history)
        logger.info("Starting from frame %d", start_frame)
        model = YOLO(model_path())
        for _frame, mask in presence_frames(path, model=model, start_frame=start_frame):
            history.append(mask)
            if is_interrupted():
                save_progress(history, key)
                return
            self._pass_advanced()
        self._pass_advanced()
        save_progress(history, key)
        logger.info("Finished processing video: %s in %.2f seconds", path, time() - started)

    def _pass_advanced(self) -> None:
        """The pass appended a frame -- on the pass's own thread.

        Nothing the pass produces is on screen in problem reporting mode, so its
        progress is not a reason to wake the UI there. Otherwise the listener
        hears about it only once the frontier has actually overtaken what is
        displayed; what to do about that is decided in :meth:`render_frame`,
        back on the thread that owns the renderer.
        """
        if self.problem_mode:
            return
        self._pass_advanced_since_render.set()
        if not self._render.repaint_due:
            return
        self.listener.changed()

    # --- mode selection -----------------------------------------------------

    @property
    def problem_mode(self) -> bool:
        return self._render.mode is self._problem

    def set_problem_mode(self, value: bool) -> None:
        """Enter or leave problem reporting mode.

        Entering pauses: a frame can only be judged if the researcher has
        stopped on it, and playback draws no detections in any case.
        """
        if self.problem_mode == value:
            return
        if value:
            self.set_playing(False)
            self._prewarm()
        self._select(self._problem if value else self._coverage)
        self.listener.changed()

    def _prewarm(self) -> None:
        """Load the detection model before a frame is waiting on it, once.

        The first inference in a process costs seconds while later ones cost a
        fraction of one. The queue is serial, so paying it as the mode is
        entered puts it ahead of the request it would otherwise delay.
        """
        detector = self._detector
        if self._prewarmed or detector is None:
            return
        self._prewarmed = True
        self._executor.submit(detector.prewarm)

    def _select(self, mode: ReviewMode) -> None:
        if self._render.set_mode(mode):
            self.listener.changed()

    # --- navigation ---------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._render.playing

    def set_playing(self, value: bool) -> None:
        """Play or pause.

        Resuming leaves problem reporting mode: the two answer unrelated
        questions -- how far the background pass has got, versus whether one
        frame's detection is right -- and playback draws no detections, so
        staying would park the researcher in a mode showing nothing.
        """
        if value and self.problem_mode:
            self._select(self._coverage)
        self._render.set_playing(value)
        self.listener.changed()

    @property
    def position(self) -> float:
        return self._render.position

    def seek(self, position: float) -> None:
        if self._render.set_position(position):
            self.listener.changed()

    def step(self, delta: int) -> None:
        """Move exactly *delta* frames, pausing playback.

        Stepping is how a defect is reached at all: the normalized slider
        cannot reliably land on one frame at ordinary frame rates.
        """
        target = self._render.step_frame(delta)
        if target is None:
            return
        self.set_playing(False)
        self.seek(target)

    @property
    def frame_index(self) -> int:
        return self._render.current_frame_index if self.video_open else 0

    @property
    def time_text(self) -> str:
        """The displayed frame's position in the recording as HH:MM:SS."""
        if not self.video_open:
            return "00:00:00"
        elapsed_seconds = self._render.timestamp_ms(self._render.current_frame_index) // 1000
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # --- rendering ----------------------------------------------------------

    def render_frame(self) -> RenderOutcome:
        """Produce the frame to display now, if anything has changed.

        The one entry point for everything that may have moved since the last
        one: a seek, a mode change, or the cumulative pass advancing on its own
        thread while nobody was looking.
        """
        if self._pass_advanced_since_render.is_set():
            # The pass appended frames on its own thread and could not act on
            # them there. Acting on them is what draws the growing track, and
            # what makes playback follow the frontier.
            self._pass_advanced_since_render.clear()
            self._render.frame_ready()
        outcome = self._render.render_now()
        if self.problem_mode:
            # The newly drawn frame may be one nobody has asked about yet.
            self._request_detection()
        return outcome

    # --- detection ----------------------------------------------------------

    def _request_detection(self) -> None:
        pending = self._problem.detection_request()
        if pending is None:
            return
        frame_index, image = pending
        detector = self._detector
        assert detector is not None  # detection_request answers None without one
        self._abandon_detection()
        self._detection_job = (
            frame_index,
            self._executor.submit(
                lambda: detector.detect(image),
                on_done=lambda detection: self._detection_ready(frame_index, detection),
                on_error=lambda _error: self._detection_failed(frame_index),
            ),
        )

    def _abandon_detection(self) -> None:
        """Drop a request the researcher has already seeked past.

        Seeking produces requests faster than inference answers them, and they
        are looking at the newest frame -- so an outstanding request is
        cancelled rather than left queued in front of the one that matters. A
        cancelled frame goes back to being unasked, so returning to it asks
        again.
        """
        outstanding = self._detection_job
        self._detection_job = None
        if outstanding is None:
            return
        frame_index, job = outstanding
        if job.cancel():
            self._problem.detection_failed(frame_index)

    def _detection_ready(self, frame_index: int, detection: Detection) -> None:
        logger.debug("detection_ready: frame %d, %d box(es)", frame_index, len(detection.boxes))
        if self._problem.detection_ready(frame_index, detection) and self.problem_mode:
            self._render.force_repaint()
        # An answer for a frame already left is kept for the researcher's
        # return; it just does not change anything on screen.
        self.listener.changed()

    def _detection_failed(self, frame_index: int) -> None:
        self._problem.detection_failed(frame_index)
        self.listener.changed()

    # --- marking ------------------------------------------------------------

    @property
    def _showing_judged_frame(self) -> bool:
        """Whether a detection result is actually displayed right now."""
        return self.problem_mode and self._render.showing_judged_frame

    @property
    def can_mark(self) -> bool:
        return self._problem.can_mark(self._showing_judged_frame)

    @property
    def frame_marked(self) -> bool:
        return self.problem_mode and self._problem.frame_marked

    def toggle_mark(self) -> None:
        """Store the frame on screen, or withdraw it if it is already stored.

        One entry point for the whole control, so the choice is made against
        the state that decides whether the control is usable at all -- never
        re-derived in the UI from a tick a click has already flipped.
        """
        if self.frame_marked:
            self._submit_retraction(self._problem.unmark(self._showing_judged_frame))
        else:
            self._submit_mark(self._problem.mark(self._showing_judged_frame))
        # Even a refused click refreshes the control: it flips its own tick, so
        # it has to be sent back to reporting what is on disk.
        self.listener.changed()

    def undo(self) -> None:
        """Delete everything stored for the most recent mark."""
        if self._submit_retraction(self._problem.undo()):
            self.listener.changed()

    def _submit_mark(self, request: MarkRequest | None) -> bool:
        if request is None:
            return False
        store = self._store
        assert store is not None  # mark() answers None without one
        frame_index = request.frame_index
        self._executor.submit(
            lambda: store.mark(request),
            on_done=lambda _name: self._mark_stored(frame_index),
            on_error=lambda _error: self._mark_failed(frame_index),
        )
        return True

    def _submit_retraction(self, retraction: Retraction | None) -> bool:
        if retraction is None:
            return False
        store = self._store
        assert store is not None  # unmark()/undo() answer None without one
        key, frame_index, stem = retraction
        self._executor.submit(
            lambda: store.retract(key, frame_index, stem),
            on_done=lambda _none: self._mark_removed(frame_index),
            on_error=lambda _error: self._retraction_failed(frame_index),
        )
        return True

    def _mark_stored(self, frame_index: int) -> None:
        logger.info("Marked frame %d", frame_index)
        self._problem.storage_finished(frame_index)
        self.listener.changed()
        self.listener.mark_stored(frame_index)

    def _mark_failed(self, frame_index: int) -> None:
        logger.error("Could not store frame %d", frame_index)
        self._problem.storage_finished(frame_index)
        self._problem.forget_last_mark()
        self.listener.changed()
        self.listener.mark_failed(frame_index)

    def _mark_removed(self, frame_index: int) -> None:
        # The files are gone only once storage has run, so the control is
        # refreshed then rather than when the removal was requested.
        logger.info("Retracted frame %d", frame_index)
        self._problem.storage_finished(frame_index)
        self.listener.changed()

    def _retraction_failed(self, frame_index: int) -> None:
        # Nothing was removed, but the control must not stay disabled for a
        # write that will never finish -- the researcher can try again.
        logger.error("Could not retract frame %d", frame_index)
        self._problem.storage_finished(frame_index)
        self.listener.changed()
