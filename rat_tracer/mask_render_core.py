"""Pure decision logic for VideoMasker's paused/playing render state machine.

This holds no Qt dependency (no QObject, QThread, QTimer, QVideoFrame) so it
can be driven and unit-tested directly, without an event loop, real threads,
or monkeypatching Qt's scheduling primitives. ``VideoMasker`` (rat_tracer.ui)
is the "humble" adapter: it owns the Qt wiring (signals, QThread, QTimer,
QVideoSink) and delegates every decision -- what to render, when a mask is
already applied, whether a re-render needs scheduling -- to a ``MaskRenderCore``
instance.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Protocol

from numpy import ndarray

from rat_tracer.coverage import CoverageHistory
from rat_tracer.paint import apply_red_mask

logger = getLogger(__name__)


class FrameCapture(Protocol):
    """Structural type for the subset of cv2.VideoCapture used here."""

    def frame_count(self) -> int: ...

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
    """Decides what frame (if any) to show as the background pass progresses."""

    def __init__(self) -> None:
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

    def reset(self) -> None:
        self.history.clear()
        self.mask_rendered = False
        self.position = 0.0
        self.cap = None
        self.total_frame_count = 0.0

    def open(self, cap: FrameCapture) -> None:
        self.cap = cap
        self.total_frame_count = cap.frame_count()

    def set_playing(self, value: bool) -> bool:
        """Returns True if the caller should schedule a render now."""
        self.playing = value
        return self.frame_ready()

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
            if self.position == self._pending_position and (
                self.mask_rendered
                or not self.history.contains(self.position_to_frame_index(self.position))
            ):
                logger.debug("render_now: nothing to render")
                return _NOTHING_TO_RENDER
            new_value = self._pending_position
            assert new_value is not None
            self.position = new_value
            return self._produce_frame(new_value)
        finally:
            self._render_pending = False

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
        if 0 <= frame_idx < len(self.history):
            apply_red_mask(img, self.history[frame_idx])
            self.mask_rendered = True
        else:
            logger.debug("_produce_frame: frame index %d is not processed yet", frame_idx)
        self.frame_count += 1
        return RenderOutcome(should_emit=True, image=img)

    def position_to_frame_index(self, position: float) -> int:
        return int(position * self.total_frame_count)
