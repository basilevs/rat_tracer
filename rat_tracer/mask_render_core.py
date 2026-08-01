"""Decoding, position, and when a frame needs drawing again.

This holds no Qt dependency (no QObject, QThread, QTimer, QVideoFrame) so it
can be driven and unit-tested directly, without an event loop, real threads,
or monkeypatching Qt's scheduling primitives.

Its remit stops at the frame: which one is displayed, when it must be redrawn,
and the index and timestamp that name it. *What* is drawn over it belongs to
the active mode (:mod:`rat_tracer.review_modes`), which this asks rather than
decides -- so adding a way of looking at the video does not touch this file,
and this file has no opinion about which way is in use.
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Protocol

from numpy import ndarray

from rat_tracer.coverage import CoverageHistory
from rat_tracer.review_modes import ReviewMode

logger = getLogger(__name__)


class FrameCapture(Protocol):
    """Structural type for the subset of cv2.VideoCapture used here."""

    def frame_count(self) -> int: ...

    def fps(self) -> float: ...

    def read(self, frame_idx: int) -> ndarray | None: ...


@dataclass(frozen=True)
class RenderOutcome:
    """Result of :meth:`MaskRenderCore.render_now`.

    ``should_emit`` distinguishes "nothing changed, leave the current frame
    on screen" (False) from "show something now" (True). When ``should_emit``
    is True, ``image`` is either the frame to display or ``None`` for an
    empty/placeholder frame -- three states that a plain ``ndarray | None``
    can't express on its own.
    """

    should_emit: bool
    image: ndarray | None = None


_NOTHING_TO_RENDER = RenderOutcome(should_emit=False)


class MaskRenderCore:
    """Decides what is shown, and what may be marked, as the pass progresses."""

    def __init__(self, mode: ReviewMode | None = None) -> None:
        #: The active way of looking at the video. Set by the session; this
        #: class never chooses one.
        self.mode = mode
        self.video_path: Path | None = None
        self.video_key: str | None = None
        self.history = CoverageHistory()
        self.position = 0.0
        self.playing = True
        self.cap: FrameCapture | None = None
        self.total_frame_count = 0.0
        self.frame_count = 0
        #: Whether the active mode drew everything it currently has. False
        #: means something is still outstanding -- an unprocessed frame, a
        #: detection not back yet -- and a repaint is due when it arrives.
        self.overlay_complete = False
        self._pending_position: float | None = None
        self._render_pending = False
        self.position_seconds = 0.0
        self.fps = 0.0
        # The frame as decoded, before the mode drew on it. Every overlay
        # mutates in place, and a marked frame must be stored raw -- annotated
        # pixels are unusable as training data (FR-12).
        self.raw_frame: ndarray | None = None
        self.rendered_frame_index: int | None = None
        self._force_render = False

    def reset(self) -> None:
        self.history.clear()
        self.overlay_complete = False
        self.position = 0.0
        self.cap = None
        self.video_path = None
        self.video_key = None
        self.total_frame_count = 0.0
        self.fps = 0.0
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
        return self.frame_ready()

    def set_mode(self, mode: ReviewMode) -> bool:
        """Show the video the way *mode* does from now on.

        Returns True if the caller should schedule a render now. Swapping the
        mode changes only what is drawn over the frame -- the recorded coverage
        is untouched, so going back brings the track return with nothing lost.
        """
        if mode is self.mode:
            return False
        if self.mode is not None:
            self.mode.left()
        self.mode = mode
        mode.entered()
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
            "frame_ready: %d/%d, playing: %s, overlay_complete: %s",
            last_frame,
            total,
            self.playing,
            self.overlay_complete,
        )
        if self.playing:
            if self.cap:
                return self.set_position(processed_position)
            return False
        if not self.overlay_complete and self.position < processed_position:
            return self._schedule_render()
        return False

    def force_repaint(self) -> None:
        """Redraw even though neither the position nor the coverage moved.

        Needed when what the *mode* can draw has changed under a still frame --
        a detection arriving for the frame on screen, say.
        """
        self._force_render = True

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
        """True when the frame on screen is missing something it could show.

        The answer is the active mode's: only it knows what it is waiting for.
        """
        if self.mode is None:
            return False
        return self.mode.repaint_needed(
            self.position_to_frame_index(self.position), self.overlay_complete
        )

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
        self.rendered_frame_index = frame_idx
        # Kept before the mode draws anything: a marked frame must be stored
        # without annotation.
        self.raw_frame = img.copy()
        self.overlay_complete = self.mode.draw(img, frame_idx) if self.mode is not None else True
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
