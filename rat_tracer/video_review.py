"""A researcher's pass through a video, and everything it takes to make one.

The model behind the application: where the video is stopped, what the
detector says about the frame stopped on, and which frames have been marked as
failures. It is the whole review minus the parts that genuinely need Qt --
those stay in :mod:`rat_tracer.ui`, which drives this and reports completions
back to it.

The behaviour used to live in the widget, which put it in the one place
hardest to test; the split exists so a review can be driven without an event
loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

from rat_tracer.bad_frames import Detection
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore, RenderOutcome
from rat_tracer.review_modes import (
    CoverageMode,
    DetectionSource,
    MarkStorage,
    ProblemReportMode,
)

logger = getLogger(__name__)


@dataclass(frozen=True)
class ReviewListener:
    """What the review tells the UI, as plain callbacks.

    Deliberately not Qt signals: the review must stay drivable without an
    event loop. ``VideoMasker`` supplies callbacks that emit the real signals.
    """

    #: A render is due; the UI should schedule one and call :meth:`render_frame`.
    schedule_render: Callable[[], None] = lambda: None
    #: Something the controls display has changed (mode, position, mark state).
    state_changed: Callable[[], None] = lambda: None
    #: A frame was stored -- its index is shown in the confirmation.
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

    Nothing here imports Qt. Anything that cannot finish immediately is a
    request handed to a service plus a completion reported back
    (:meth:`detection_ready`, :meth:`mark_stored` ...), so a test drives an
    entire review synchronously with fakes -- no threads, no event loop, no
    model weights, no temp directories.
    """

    def __init__(
        self,
        listener: ReviewListener | None = None,
        storage: MarkStorage | None = None,
        detection: DetectionSource | None = None,
    ):
        self.listener = listener if listener is not None else ReviewListener()
        self.render = MaskRenderCore()
        self.coverage = CoverageMode(self.render.history)
        self.problem = ProblemReportMode(view=self.render, detection=detection, storage=storage)
        self.render.mode = self.coverage

    # --- the video ----------------------------------------------------------

    def open_video(self, cap: FrameCapture, video_path: Path, video_key: str) -> None:
        self.render.open(cap, video_path, video_key)
        self.listener.state_changed()

    def close_video(self) -> None:
        self.render.reset()
        self.problem.forget_video()
        self._select(self.coverage)
        self.listener.state_changed()

    @property
    def video_open(self) -> bool:
        return self.render.cap is not None

    @property
    def history(self):
        """The coverage history the background pass appends to."""
        return self.render.history

    # --- mode selection -----------------------------------------------------

    @property
    def problem_mode(self) -> bool:
        return self.render.mode is self.problem

    def set_problem_mode(self, value: bool) -> None:
        """Enter or leave problem reporting mode.

        Entering pauses: a frame can only be judged if the researcher has
        stopped on it, and playback draws no detections in any case.
        """
        if self.problem_mode == value:
            return
        if value:
            self.set_playing(False)
        self._select(self.problem if value else self.coverage)
        self.listener.state_changed()

    def _select(self, mode) -> None:
        if self.render.set_mode(mode):
            self.listener.schedule_render()

    # --- navigation ---------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self.render.playing

    def set_playing(self, value: bool) -> None:
        """Play or pause.

        Resuming leaves problem reporting mode: the two answer unrelated
        questions -- how far the background pass has got, versus whether one
        frame's detection is right -- and playback draws no detections, so
        staying would park the researcher in a mode showing nothing.
        """
        if value and self.problem_mode:
            self._select(self.coverage)
        if self.render.set_playing(value):
            self.listener.schedule_render()
        self.listener.state_changed()

    @property
    def position(self) -> float:
        return self.render.position

    def seek(self, position: float) -> None:
        if self.render.set_position(position):
            self.listener.schedule_render()

    def step(self, delta: int) -> None:
        """Move exactly *delta* frames, pausing playback.

        Stepping is how a defect is reached at all: the normalized slider
        cannot reliably land on one frame at ordinary frame rates.
        """
        target = self.render.step_frame(delta)
        if target is None:
            return
        self.set_playing(False)
        self.seek(target)

    @property
    def frame_index(self) -> int:
        return self.render.current_frame_index if self.video_open else 0

    @property
    def time_text(self) -> str:
        """The displayed frame's position in the recording as HH:MM:SS."""
        if not self.video_open:
            return "00:00:00"
        elapsed_seconds = self.render.timestamp_ms(self.render.current_frame_index) // 1000
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # --- rendering ----------------------------------------------------------

    def render_frame(self) -> RenderOutcome:
        """Produce the frame to display now, if anything has changed."""
        outcome = self.render.render_now()
        if outcome.should_emit:
            self.listener.state_changed()
        if self.problem_mode:
            # The newly drawn frame may be one nobody has asked about yet.
            self.problem.request_detection()
        return outcome

    def frame_processed(self) -> None:
        """The cumulative pass appended a frame."""
        if self.problem_mode:
            # Nothing the cumulative pass produces is on screen in this mode,
            # so its progress is not a reason to repaint.
            return
        if self.render.frame_ready():
            self.listener.schedule_render()

    # --- detection ----------------------------------------------------------

    def request_detection(self) -> None:
        if self.problem_mode:
            self.problem.request_detection()

    def detection_ready(self, frame_index: int, detection: Detection) -> None:
        logger.debug("detection_ready: frame %d, %d box(es)", frame_index, len(detection.boxes))
        if self.problem.detection_ready(frame_index, detection) and self.problem_mode:
            self.render.force_repaint()
            self.listener.schedule_render()
        else:
            # An answer for a frame already left is kept for the researcher's
            # return, but nothing on screen changes.
            self.listener.state_changed()

    def detection_failed(self, frame_index: int) -> None:
        self.problem.detection_failed(frame_index)
        self.listener.state_changed()

    # --- marking ------------------------------------------------------------

    @property
    def _showing_judged_frame(self) -> bool:
        """Whether a detection result is actually displayed right now.

        True only in problem reporting mode, stopped, on a frame whose answer
        has arrived and been drawn. Every stored mark is therefore something
        the researcher looked at, which is why no metadata field has to assert
        it.
        """
        return (
            self.problem_mode
            and not self.playing
            and self.video_open
            and self.render.rendered_frame_index == self.render.current_frame_index
            and self.render.overlay_complete
        )

    @property
    def can_mark(self) -> bool:
        return self.problem.can_mark(self._showing_judged_frame)

    @property
    def frame_marked(self) -> bool:
        return self.problem_mode and self.problem.frame_marked

    def toggle_mark(self) -> None:
        """Store the frame on screen, or withdraw it if it is already stored.

        One entry point for the whole control, so the choice is made against
        the state that decides whether the control is usable at all -- never
        re-derived in the UI from a tick a click has already flipped.
        """
        if self.frame_marked:
            self.unmark()
        else:
            self.mark()

    def mark(self) -> None:
        self.problem.mark(self._showing_judged_frame)
        # Even a refused mark refreshes the control: a click flips the tick
        # itself, so it has to be sent back to reporting what is on disk.
        self.listener.state_changed()

    def unmark(self) -> None:
        self.problem.unmark(self._showing_judged_frame)
        self.listener.state_changed()

    def undo(self) -> None:
        """Delete everything stored for the most recent mark."""
        if self.problem.undo():
            self.listener.state_changed()

    # --- storage completions ------------------------------------------------

    def mark_stored(self, frame_index: int) -> None:
        logger.info("Marked frame %d", frame_index)
        self.problem.storage_finished(frame_index)
        self.listener.state_changed()
        self.listener.mark_stored(frame_index)

    def mark_failed(self, frame_index: int) -> None:
        logger.error("Could not store frame %d", frame_index)
        self.problem.storage_finished(frame_index)
        self.problem.forget_last_mark()
        self.listener.state_changed()
        self.listener.mark_failed(frame_index)

    def mark_removed(self, frame_index: int) -> None:
        # The files are gone only once storage has run, so the control is
        # refreshed then rather than when the removal was requested.
        logger.info("Retracted frame %d", frame_index)
        self.problem.storage_finished(frame_index)
        self.listener.state_changed()
