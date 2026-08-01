"""Every decision behind the video view, with no Qt in sight.

The application is split so that this module answers *what should happen* and
``VideoMasker`` (rat_tracer.ui) only arranges *when it happens*: signals,
QThread, QTimer, QVideoSink. Nothing here imports Qt, so all of it can be
driven directly in tests -- no event loop, no real threads, no monkeypatching
of Qt's scheduling.

The decisions living here are:

* what to draw on the current frame -- the cumulative coverage mask, or, in
  problem reporting mode, the boxes the detector produced for that frame alone;
* whether a re-render is needed at all, and when the caller should schedule one;
* which frame index and timestamp the researcher is looking at;
* whether a detection should be requested, and for which frame;
* whether the frame on screen may be marked, is already stored, or has a write
  still in flight -- and what a mark record for it would contain.

That last group is why this is not called a renderer: the same state that
decides what is drawn decides what may be marked, and splitting them would
mean keeping two copies of "which frame is the researcher actually judging".
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Protocol

from numpy import ndarray

from rat_tracer.bad_frames import Detection, MarkRequest
from rat_tracer.coverage import CoverageHistory
from rat_tracer.paint import apply_red_mask, draw_detection_boxes

logger = getLogger(__name__)


class FrameCapture(Protocol):
    """Structural type for the subset of cv2.VideoCapture used here."""

    def frame_count(self) -> int: ...

    def fps(self) -> float: ...

    def read(self, frame_idx: int) -> ndarray | None: ...


class MarkedFrames(Protocol):
    """The subset of :class:`~rat_tracer.bad_frames.BadFrameStore` needed here.

    Injected rather than imported so the core keeps no opinion about storage --
    and so its tests need no temp directories.
    """

    def is_marked(self, video_key: str, frame_index: int) -> bool: ...


@dataclass(frozen=True)
class DetectionRequest:
    """A frame whose detection should be computed now."""

    frame_index: int
    image: ndarray


@dataclass(frozen=True)
class RenderOutcome:
    """Result of :meth:`FrameReviewCore.render_now`.

    ``should_emit`` distinguishes "nothing changed, leave the current frame
    on screen" (False) from "show something now" (True). When ``should_emit``
    is True, ``image`` is either the frame to display or ``None`` for an
    empty/placeholder frame -- three states that a plain ``ndarray | None``
    can't express on its own.
    """

    should_emit: bool
    image: ndarray | None = None


_NOTHING_TO_RENDER = RenderOutcome(should_emit=False)


class FrameReviewCore:
    """Decides what is shown, and what may be marked, as the pass progresses."""

    def __init__(self, marks: MarkedFrames | None = None) -> None:
        self.marks = marks
        self.video_path: Path | None = None
        self.video_key: str | None = None
        self.history = CoverageHistory()
        self.position = 0.0
        self.playing = True
        self.cap: FrameCapture | None = None
        self.total_frame_count = 0.0
        self.frame_count = 0
        self.mask_rendered = False
        self._pending_position: float | None = None
        self._render_pending = False
        self.position_seconds = 0.0
        # --- problem reporting mode ---
        self.problem_mode = False
        self.fps = 0.0
        # Detections are cached per frame index so returning to a frame does
        # not pay for inference twice, and so the mark control can tell "the
        # detector found nothing" from "no answer yet".
        self._detections: dict[int, Detection] = {}
        self.detection_rendered = False
        # The frame as decoded, before any overlay. apply_red_mask and
        # draw_detection_boxes both mutate in place, and a marked frame must be
        # stored raw -- annotated pixels are unusable as training data (FR-12).
        self.raw_frame: ndarray | None = None
        self.rendered_frame_index: int | None = None
        self._force_render = False
        # Frames already asked about, so seeking back and forth does not
        # re-request an answer that is already on its way.
        self._requested_detections: set[int] = set()
        # Frames whose write or removal has been handed to storage but has not
        # finished. Storage is asynchronous, so "marked" stays false meanwhile
        # and without this the control would look available for a second click.
        self._frames_in_flight: set[int] = set()

    def reset(self) -> None:
        self.history.clear()
        self.mask_rendered = False
        self.position = 0.0
        self.cap = None
        self.video_path = None
        self.video_key = None
        self.total_frame_count = 0.0
        self.fps = 0.0
        self._detections.clear()
        self._requested_detections.clear()
        self._frames_in_flight.clear()
        self.detection_rendered = False
        self.raw_frame = None
        self.rendered_frame_index = None

    def open(self, cap: FrameCapture, video_path: Path, video_key: str) -> None:
        self.cap = cap
        self.video_path = video_path
        self.video_key = video_key
        self.total_frame_count = cap.frame_count()
        self.fps = cap.fps()
        if self.fps <= 0:
            logger.warning("Video reports no frame rate; timestamps will read 00:00:00")

    def set_playing(self, value: bool) -> bool:
        """Returns True if the caller should schedule a render now."""
        self.playing = value
        if value and self.problem_mode:
            # Playback and problem reporting answer unrelated questions --
            # "how far has the background pass got" versus "is this frame's
            # detection right" -- and playback draws no detections anyway
            # (FR-5). Resuming therefore leaves the mode rather than parking
            # the researcher in one that shows nothing.
            self.set_problem_mode(False)
        return self.frame_ready()

    def set_problem_mode(self, value: bool) -> bool:
        """Enter or leave problem reporting mode.

        Returns True if the caller should schedule a render now. The mode is a
        display state only: it changes what is drawn over the current frame and
        never touches the recorded coverage, so leaving it brings the
        cumulative mask back with nothing lost.
        """
        if self.problem_mode == value:
            return False
        logger.debug("set_problem_mode: %s", value)
        if value:
            # Judging a frame means stopping on it, so entering the mode pauses
            # rather than leaving the researcher chasing the playhead.
            self.playing = False
        self.problem_mode = value
        # The frame on screen was drawn for the other mode, and neither the
        # position nor the coverage has changed -- so ask for a repaint
        # explicitly rather than relying on the usual change detection.
        self._force_render = True
        return self._schedule_render()

    def frame_ready(self) -> bool:
        """Call whenever the background pass appends a frame (or playing/
        video-output changes). Returns True if the caller should schedule a
        render now."""
        total = self.total_frame_count
        if total == 0:
            logger.debug("frame_ready: no frames yet (total=0)")
            return False
        last_frame = len(self.history) - 1
        processed_position = float(last_frame) / total
        logger.debug(
            "frame_ready: %d/%d, playing: %s, mask_rendered: %s",
            last_frame,
            total,
            self.playing,
            self.mask_rendered,
        )
        if self.playing:
            if self.cap:
                return self.set_position(processed_position)
            return False
        if self.problem_mode:
            # Nothing the cumulative pass produces is on screen in this mode,
            # so its progress is not a reason to repaint.
            return False
        if not self.mask_rendered and self.position < processed_position:
            return self._schedule_render()
        return False

    def set_position(self, new_value: float) -> bool:
        """Returns True if the caller should schedule a render now."""
        if self._pending_position == new_value:
            return False
        self._pending_position = new_value
        logger.debug("set_position: requested %.3f", new_value)
        return self._schedule_render()

    def _schedule_render(self) -> bool:
        if self._render_pending:
            return False
        self._render_pending = True
        return True

    def render_now(self) -> RenderOutcome:
        """Mirrors the original ``_rerender_if_needed`` + ``_produce_frame``.

        Always clears the render-pending flag, even if nothing is rendered,
        so a later ``set_position``/``frame_ready`` call can schedule again.
        """
        try:
            if (
                not self._force_render
                and self.position == self._pending_position
                and not self._repaint_needed()
            ):
                logger.debug("render_now: nothing to render")
                return _NOTHING_TO_RENDER
            new_value = self._pending_position
            if new_value is None:
                # A forced repaint can be the very first render of a video, so
                # there is no requested position yet -- repaint where we are.
                new_value = self.position
                self._pending_position = new_value
            self.position = new_value
            return self._produce_frame(new_value)
        finally:
            self._render_pending = False
            self._force_render = False

    def _repaint_needed(self) -> bool:
        """True when the frame on screen is missing an overlay it could have."""
        frame_idx = self.position_to_frame_index(self.position)
        if self.problem_mode and not self.playing:
            # Waiting on the detector: repaint once its answer lands.
            return not self.detection_rendered and frame_idx in self._detections
        # Waiting on the cumulative pass: repaint once it reaches this frame.
        return not self.mask_rendered and self.history.contains(frame_idx)

    def _produce_frame(self, position: float) -> RenderOutcome:
        capture = self.cap
        if not capture:
            logger.warning("_produce_frame: no video capture available for rendering")
            return RenderOutcome(should_emit=True, image=None)
        frame_idx = self.position_to_frame_index(position)
        r = capture.read(frame_idx)
        if r is None:
            logger.warning("_produce_frame: cannot read frame %d", frame_idx)
            return RenderOutcome(should_emit=True, image=None)
        img: ndarray = r
        self.mask_rendered = False
        self.detection_rendered = False
        self.rendered_frame_index = frame_idx
        if self.problem_mode and not self.playing:
            # A single frame cannot be judged against the cumulative mask: it is
            # the union of every detection so far, so a red region says nothing
            # about *this* frame. Hide it and draw only what the detector
            # produced here.
            self.raw_frame = img.copy()
            detection = self._detections.get(frame_idx)
            if detection is not None:
                draw_detection_boxes(img, detection.boxes)
                self.detection_rendered = True
            else:
                logger.debug("_produce_frame: no detection for frame %d yet", frame_idx)
        else:
            self.raw_frame = None
            if 0 <= frame_idx < len(self.history):
                apply_red_mask(img, self.history[frame_idx])
                self.mask_rendered = True
            else:
                logger.debug("_produce_frame: frame index %d is not processed yet", frame_idx)
        self.frame_count += 1
        return RenderOutcome(should_emit=True, image=img)

    def position_to_frame_index(self, position: float) -> int:
        frame_idx = int(position * self.total_frame_count)
        # A slider at its maximum maps one past the last frame; clamping keeps
        # every reported index one a researcher and a technician can both name.
        last = int(self.total_frame_count) - 1
        return max(0, min(frame_idx, last)) if last >= 0 else 0

    def frame_index_to_position(self, frame_index: int) -> float:
        """Inverse of :meth:`position_to_frame_index`, for frame stepping.

        Aims at the middle of the frame's slice of the slider so that rounding
        cannot land the result back on a neighbouring frame.
        """
        if self.total_frame_count <= 0:
            return 0.0
        return (frame_index + 0.5) / self.total_frame_count

    @property
    def current_position(self) -> float:
        """Where the researcher is now, seek included.

        ``position`` only catches up when a render happens, so it answers
        "what is on screen" rather than "where are we". Frame stepping,
        detection requests and the index readout all need the latter: a
        detection must be requested for the frame just seeked to, not for the
        one still being displayed.
        """
        return self._pending_position if self._pending_position is not None else self.position

    @property
    def current_frame_index(self) -> int:
        return self.position_to_frame_index(self.current_position)

    def timestamp_ms(self, frame_index: int) -> int:
        """Position of *frame_index* in the recording, in milliseconds.

        Derived from the frame index rather than read back from the capture:
        the capture's position is a side effect of the last read, which the
        detector and the renderer both perform.
        """
        if self.fps <= 0:
            return 0
        return int(frame_index / self.fps * 1000)

    def step_frame(self, delta: int) -> float | None:
        """Position *delta* frames away, or None when it would leave the video.

        Returns the new position without applying it; the caller pauses
        playback and seeks, so that stepping is meaningful in the first place.
        """
        if self.cap is None or self.total_frame_count <= 0:
            return None
        target = self.current_frame_index + delta
        if target < 0 or target >= int(self.total_frame_count):
            return None
        return self.frame_index_to_position(target)

    def needs_detection(self) -> int | None:
        """The frame index whose detection should be requested now, if any."""
        if not self.problem_mode or self.playing or self.cap is None:
            return None
        frame_idx = self.current_frame_index
        if frame_idx in self._detections:
            return None
        return frame_idx

    def take_detection_request(self) -> DetectionRequest | None:
        """Claim the next detection to compute, or None if there is nothing.

        Claiming is the point: the same frame is never asked about twice, and
        the image handed over is a copy, because the caller runs the detector
        on another thread while rendering keeps mutating its own frame.
        """
        frame_index = self.needs_detection()
        if frame_index is None or frame_index in self._requested_detections:
            return None
        if self.rendered_frame_index != frame_index or self.raw_frame is None:
            # The frame is not on screen yet; the render that puts it there
            # gives the caller another chance to ask.
            return None
        self._requested_detections.add(frame_index)
        logger.debug("take_detection_request: frame %d", frame_index)
        return DetectionRequest(frame_index=frame_index, image=self.raw_frame.copy())

    def detection_failed(self, frame_index: int) -> None:
        """Let a frame be asked about again after its detection failed.

        Otherwise a single failure -- a model that will not load, say -- would
        leave the control disabled for that frame for the rest of the session.
        """
        self._requested_detections.discard(frame_index)

    def set_detection(self, frame_index: int, detection: Detection) -> bool:
        """Record a detection result. Returns True if a render is needed now."""
        self._detections[frame_index] = detection
        if not self.problem_mode or self.playing:
            return False
        if frame_index != self.current_frame_index:
            # The researcher moved on while this was computing; the answer is
            # kept for when they come back, but nothing on screen changes.
            return False
        self._force_render = True
        return self._schedule_render()

    def detection_for(self, frame_index: int) -> Detection | None:
        return self._detections.get(frame_index)

    @property
    def current_detection(self) -> Detection | None:
        return self._detections.get(self.current_frame_index)

    @property
    def showing_judged_frame(self) -> bool:
        """Whether a detection result is actually displayed right now.

        True only in problem reporting mode, stopped, on a frame whose answer
        has arrived and been drawn. Every stored mark is therefore something
        the researcher looked at, which is why no metadata field has to assert
        it.
        """
        return (
            self.problem_mode
            and not self.playing
            and self.cap is not None
            and self.rendered_frame_index is not None
            and self.rendered_frame_index == self.current_frame_index
            and self.detection_rendered
        )

    @property
    def can_mark(self) -> bool:
        """Whether the control acts on the frame on screen at all.

        Covers both directions: a judged frame can be stored, and a stored one
        can be withdrawn. Only a frame whose storage is mid-flight is off
        limits, so a second click cannot queue a second write or removal.
        """
        return (
            self.showing_judged_frame
            and self.video_path is not None
            and self.video_key is not None
            and self.current_frame_index not in self._frames_in_flight
        )

    @property
    def frame_marked(self) -> bool:
        """Whether the displayed frame is already stored.

        The control is stateful so the researcher sees the answer before
        acting, rather than discovering it by pressing.
        """
        if self.marks is None or self.video_key is None:
            return False
        return self.marks.is_marked(self.video_key, self.current_frame_index)

    def build_mark_request(self, model_id: str) -> MarkRequest | None:
        """Describe the frame on screen as a record to store, if it may be.

        Returns None -- rather than raising -- for every reason a mark cannot
        be made, because the caller's job is only to relay the answer to the
        control, not to distinguish them.
        """
        if not self.can_mark:
            logger.debug("build_mark_request: nothing judged on screen")
            return None
        assert self.video_path is not None and self.video_key is not None
        frame_index = self.current_frame_index
        image = self.raw_frame
        detection = self._detections.get(frame_index)
        if image is None or detection is None:
            logger.warning("build_mark_request: no raw frame or detection for %d", frame_index)
            return None
        if self.frame_marked:
            logger.debug("build_mark_request: frame %d is already marked", frame_index)
            return None
        return MarkRequest(
            image=image.copy(),
            video_path=self.video_path,
            video_key=self.video_key,
            frame_index=frame_index,
            timestamp_ms=self.timestamp_ms(frame_index),
            detection=detection,
            model_id=model_id,
        )

    def unmark_target(self) -> tuple[str, int, str] | None:
        """The ``(video_key, frame_index, stem)`` to withdraw, if any.

        The five-second Undo is the correction for a misclick, but it cannot
        help a researcher looking straight at a frame they marked earlier.
        Nothing has to be navigated to for that -- the frame is on screen and
        the control already says it is stored.
        """
        if not self.can_mark or not self.frame_marked:
            return None
        assert self.video_path is not None and self.video_key is not None
        return (self.video_key, self.current_frame_index, self.video_path.stem)

    def begin_storage(self, frame_index: int) -> None:
        """Note that storage is now working on *frame_index*."""
        self._frames_in_flight.add(frame_index)

    def end_storage(self, frame_index: int) -> None:
        """Note that storage has finished with *frame_index*, either way."""
        self._frames_in_flight.discard(frame_index)

    @property
    def time_text(self) -> str:
        """The displayed frame's position in the recording as HH:MM:SS."""
        if self.cap is None:
            return "00:00:00"
        elapsed_seconds = self.timestamp_ms(self.current_frame_index) // 1000
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
