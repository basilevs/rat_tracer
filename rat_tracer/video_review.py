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
from logging import getLogger
from pathlib import Path
from threading import Event
from time import time

from ultralytics import YOLO

from rat_tracer.background import BackgroundExecutor
from rat_tracer.bad_frames import BadFrameStore
from rat_tracer.frame_detector import FrameDetector
from rat_tracer.lib import model_path
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore, RenderOutcome
from rat_tracer.paint import presence_frames
from rat_tracer.progress_cache import video_key
from rat_tracer.review_listener import ReviewListener
from rat_tracer.review_modes import CoverageMode, ProblemReportMode, ReviewMode

logger = getLogger(__name__)

__all__ = ["ReviewListener", "VideoReview"]


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
        self._render = MaskRenderCore()
        self._coverage = CoverageMode(frontier_moved=self._frontier_moved)
        self._problem = ProblemReportMode(
            view=self._render,
            executor=executor,
            listener=self.listener,
            detector=detector,
            store=store,
        )
        #: Which way the video is being looked at. The renderer is not told:
        #: it decodes frames and has no idea anything is drawn over them.
        self._mode: ReviewMode = self._coverage
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
        self._pass_advanced_since_render.clear()
        self._render.reset()
        self._problem.forget_video()
        self._coverage.forget()
        self._select(self._coverage)
        self.listener.changed()

    @property
    def video_open(self) -> bool:
        return self._render.video_open

    # --- the cumulative pass ------------------------------------------------

    def process_video(self, is_interrupted: Callable[[], bool]) -> None:
        """Run the cumulative pass over the open video, until it ends or stops.

        Blocking, and the only method here meant to be called from another
        thread. It hands each frame to the coverage track, which is lock-guarded
        -- never to the renderer, whose state belongs to the thread that
        navigates.

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

        coverage = self._coverage
        coverage.resume(key)
        start_frame = coverage.processed_frames
        logger.info("Starting from frame %d", start_frame)
        model = YOLO(model_path())
        for _frame, mask in presence_frames(path, model=model, start_frame=start_frame):
            coverage.record(mask)
            if is_interrupted():
                coverage.save(key)
                return
        coverage.save(key)
        logger.info("Finished processing video: %s in %.2f seconds", path, time() - started)

    def _frontier_moved(self, frame_index: int) -> None:
        """The pass processed another frame -- on the pass's own thread.

        Nothing the pass produces is on screen in problem reporting mode, so its
        progress is not a reason to wake the UI there. Otherwise the listener
        hears about it only when it would change something: playing means
        following the frontier, and paused means only that the track has now
        reached the frame being looked at. Acting on it happens in
        :meth:`render_frame`, back on the thread that owns the renderer.
        """
        if self.problem_mode:
            return
        self._pass_advanced_since_render.set()
        if self.playing or self._coverage.repaint_needed(self._render.displayed_frame_index):
            self.listener.changed()

    # --- mode selection -----------------------------------------------------

    @property
    def problem_mode(self) -> bool:
        return self._mode is self._problem

    def set_problem_mode(self, value: bool) -> None:
        """Enter or leave problem reporting mode.

        Entering pauses: a frame can only be judged if the researcher has
        stopped on it, and playback draws no detections in any case.
        """
        if self.problem_mode == value:
            return
        if value:
            self.set_playing(False)
        self._select(self._problem if value else self._coverage)
        self.listener.changed()

    def _select(self, mode: ReviewMode) -> None:
        """Show the video the way *mode* does from now on.

        Nothing has to force a repaint: telling the outgoing mode it has ``left``
        clears its record of what is on screen, and the incoming one has no
        record of the current frame either, so it asks to draw of its own
        accord. The recorded coverage is untouched, so going back brings the
        track return with nothing lost.
        """
        if mode is self._mode:
            return
        self._mode.left()
        self._mode = mode
        mode.entered()
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
        if value:
            # Resuming catches up with the pass now, rather than sitting still
            # until it happens to produce another frame -- or forever, if it has
            # already finished.
            self._pass_advanced_since_render.set()
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
            # them there. Playing means following the frontier, so acting on
            # them is what advances playback.
            self._pass_advanced_since_render.clear()
            self._follow_frontier()
        outcome = self._render.render_now(
            paint=self._mode.draw,
            repaint_wanted=self._mode.repaint_needed(self._render.displayed_frame_index),
        )
        if self.problem_mode:
            # The newly drawn frame may be one nobody has asked about yet. What
            # is worth asking, and what to do with the answer, is the mode's.
            self._problem.request_detection()
        return outcome

    def _follow_frontier(self) -> None:
        """Playing means watching the pass work, one processed frame behind."""
        if not self.playing:
            return
        frontier = self._coverage.frontier
        if frontier is None:
            return
        self._render.set_position(self._render.frame_index_to_position(frontier))

    # --- marking ------------------------------------------------------------

    @property
    def can_mark(self) -> bool:
        return self._problem.can_mark

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
            self._problem.unmark()
        else:
            self._problem.mark()
        # Even a refused click refreshes the control: it flips its own tick, so
        # it has to be sent back to reporting what is on disk.
        self.listener.changed()

    def undo(self) -> None:
        """Delete everything stored for the most recent mark."""
        self._problem.undo()
