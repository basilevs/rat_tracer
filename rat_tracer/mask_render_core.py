"""The video, where it is stopped, and getting that frame decoded.

This holds no Qt dependency (no QObject, QThread, QTimer, QVideoFrame) so it
can be driven and unit-tested directly, without an event loop, real threads,
or monkeypatching Qt's scheduling primitives.

Its remit stops at the frame: which one is displayed, the index and timestamp
that name it, and handing over its pixels once. It knows nothing about modes --
not which one is active, not that there is more than one, not that the concept
exists. Anything drawn over a frame happens after this file is done with it, so
adding a way of looking at a video does not touch this file at all.

The state behind that is private. Whether a decode is due is decided here, from
flags only :meth:`MaskRenderCore.render_now` may clear, and an outside
assignment to any of them silently costs a repaint; what callers get instead are
the questions they actually ask -- :attr:`~MaskRenderCore.position`,
:attr:`~MaskRenderCore.raw_frame`, :attr:`~MaskRenderCore.displayed_frame_index`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Protocol

from numpy import ndarray

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
    """Where the video is stopped, and the frame that belongs there."""

    def __init__(self) -> None:
        self._clear()

    def _clear(self) -> None:
        """Put everything a video owns back to its opening value.

        One place, used by both ``__init__`` and :meth:`reset`, because the
        fields that matter most here are the ones easiest to forget: the
        pending-render flags decide whether the *next* video gets its first
        repaint at all, and a stale ``_pending_position`` makes a seek to the
        position the last video was left at look like no seek at all.
        """
        self._video_path: Path | None = None
        self._video_key: str | None = None
        self._cap: FrameCapture | None = None
        self._total_frames = 0
        self._fps = 0.0
        self._position = 0.0
        self._pending_position: float | None = None
        self._render_pending = False
        self._playing = True
        # The frame as decoded. Every overlay mutates in place, and a marked
        # frame must be stored raw -- annotated pixels are unusable as training
        # data (FR-12) -- so what is handed out to be drawn on is a copy.
        self._raw_frame: ndarray | None = None
        self._rendered_frame_index: int | None = None

    def reset(self) -> None:
        self._clear()

    def open(self, cap: FrameCapture, video_path: Path) -> None:
        self._cap = cap
        self._video_path = video_path
        self._total_frames = int(cap.frame_count())
        self._fps = cap.fps()
        if self._fps <= 0:
            logger.warning("Video reports no frame rate; timestamps will read 00:00:00")

    def identify(self, video_key: str) -> None:
        """Name the open video by its content fingerprint.

        Separate from :meth:`open` because the fingerprint costs a full read of
        the file, so it is paid by the cumulative pass on its own thread rather
        than by whoever opened the video. Until it lands nothing can be marked,
        since a mark is stored under this key. Assigning one reference is all
        that crosses the thread boundary.
        """
        self._video_key = video_key

    # --- what is being looked at --------------------------------------------

    @property
    def video_open(self) -> bool:
        return self._cap is not None

    @property
    def video_path(self) -> Path | None:
        return self._video_path

    @property
    def video_key(self) -> str | None:
        return self._video_key

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def position(self) -> float:
        """Where the frame on screen is, as a fraction of the video."""
        return self._position

    @property
    def raw_frame(self) -> ndarray | None:
        """The displayed frame as decoded, before anything drew on it."""
        return self._raw_frame

    @property
    def rendered_frame_index(self) -> int | None:
        return self._rendered_frame_index

    @property
    def displayed_frame_index(self) -> int:
        """The frame currently on screen, as opposed to the one seeked to."""
        return self.position_to_frame_index(self._position)

    # --- what changes it ----------------------------------------------------

    def set_playing(self, value: bool) -> None:
        """Play or pause.

        Nothing is scheduled by itself: pausing leaves the same frame on screen,
        and what resuming should jump to is not this object's to know.
        """
        self._playing = value

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

    def render_now(
        self,
        paint: Callable[[ndarray, int], None] = lambda _image, _index: None,
        repaint_wanted: bool = False,
    ) -> RenderOutcome:
        """Produce the frame to display now, if anything has changed.

        *paint* draws over the decoded frame in place, and *repaint_wanted* says
        the painter has something to add that is not up yet -- both come from
        the caller, which is the only party that knows what is being drawn.

        Always clears the render-pending flag, even if nothing is rendered, so a
        later ``set_position`` can schedule again.
        """
        try:
            if self._position == self._pending_position and not repaint_wanted:
                logger.debug("render_now: nothing to render")
                return _NOTHING_TO_RENDER
            new_value = self._pending_position
            if new_value is None:
                # A wanted repaint can be the very first render of a video, so
                # there is no requested position yet -- repaint where we are.
                new_value = self._position
                self._pending_position = new_value
            self._position = new_value
            return self._produce_frame(new_value, paint)
        finally:
            self._render_pending = False

    def _produce_frame(
        self, position: float, paint: Callable[[ndarray, int], None]
    ) -> RenderOutcome:
        capture = self._cap
        if not capture:
            logger.warning("_produce_frame: no video capture available for rendering")
            return RenderOutcome(should_emit=True, image=None)
        frame_idx = self.position_to_frame_index(position)
        r = capture.read(frame_idx)
        if r is None:
            logger.warning("_produce_frame: cannot read frame %d", frame_idx)
            return RenderOutcome(should_emit=True, image=None)
        img: ndarray = r
        self._rendered_frame_index = frame_idx
        # Kept before anything draws: a marked frame must be stored unannotated.
        self._raw_frame = img.copy()
        paint(img, frame_idx)
        return RenderOutcome(should_emit=True, image=img)

    # --- naming a frame -----------------------------------------------------

    def position_to_frame_index(self, position: float) -> int:
        frame_idx = int(position * self._total_frames)
        # A slider at its maximum maps one past the last frame; clamping keeps
        # every reported index one a researcher and a technician can both name.
        last = self._total_frames - 1
        return max(0, min(frame_idx, last)) if last >= 0 else 0

    def frame_index_to_position(self, frame_index: int) -> float:
        """Inverse of :meth:`position_to_frame_index`, for frame stepping.

        Aims at the middle of the frame's slice of the slider so that rounding
        cannot land the result back on a neighbouring frame.
        """
        if self._total_frames <= 0:
            return 0.0
        return (frame_index + 0.5) / self._total_frames

    @property
    def current_position(self) -> float:
        """Where the researcher is now, seek included.

        ``position`` only catches up when a render happens, so it answers
        "what is on screen" rather than "where are we". Frame stepping,
        detection requests and the index readout all need the latter: a
        detection must be requested for the frame just seeked to, not for the
        one still being displayed.
        """
        return self._pending_position if self._pending_position is not None else self._position

    @property
    def current_frame_index(self) -> int:
        return self.position_to_frame_index(self.current_position)

    def timestamp_ms(self, frame_index: int) -> int:
        """Position of *frame_index* in the recording, in milliseconds.

        Derived from the frame index rather than read back from the capture:
        the capture's position is a side effect of the last read, which the
        detector and the renderer both perform.
        """
        if self._fps <= 0:
            return 0
        return int(frame_index / self._fps * 1000)

    def step_frame(self, delta: int) -> float | None:
        """Position *delta* frames away, or None when it would leave the video.

        Returns the new position without applying it; the caller pauses
        playback and seeks, so that stepping is meaningful in the first place.
        """
        if self._cap is None or self._total_frames <= 0:
            return None
        target = self.current_frame_index + delta
        if target < 0 or target >= self._total_frames:
            return None
        return self.frame_index_to_position(target)
