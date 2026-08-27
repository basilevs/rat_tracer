"""The open video, and the pixels in a frame of it.

No Qt, so it can be driven and unit-tested directly, without an event loop,
real threads, or monkeypatching Qt's scheduling primitives.

Nothing here is about reviewing. This knows which file is open, how to name a
frame -- by index, by position along the video, by timestamp -- and how to
decode one. Where the researcher is, whether they are playing, when a frame
needs drawing again and what gets drawn over it all belong to
:mod:`rat_tracer.video_review`, so none of it is here. Every question this
answers is a fact about the file.
"""

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


class VideoFile:
    """One video: which frames it has, and what is in them."""

    def __init__(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._path: Path | None = None
        self._key: str | None = None
        self._cap: FrameCapture | None = None
        self._frame_count = 0
        self._fps = 0.0
        # The frame as decoded. Every overlay mutates in place, and a marked
        # frame must be stored raw -- annotated pixels are unusable as training
        # data (FR-12) -- so this copy is kept before anything is drawn.
        self._raw_frame: ndarray | None = None
        self._decoded_frame_index: int | None = None

    def open(self, cap: FrameCapture, path: Path) -> None:
        self._cap = cap
        self._path = path
        self._frame_count = int(cap.frame_count())
        self._fps = cap.fps()
        if self._fps <= 0:
            logger.warning("Video reports no frame rate; timestamps will read 00:00:00")

    def close(self) -> None:
        """Forget the video entirely, so nothing of it reaches the next one."""
        self._clear()

    def identify(self, key: str) -> None:
        """Name the open video by its content fingerprint.

        Separate from :meth:`open` because the fingerprint costs a full read of
        the file, so it is paid by the cumulative pass on its own thread rather
        than by whoever opened the video. Until it lands nothing can be marked,
        since a mark is stored under this key. Assigning one reference is all
        that crosses the thread boundary.
        """
        self._key = key

    # --- which file ---------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def key(self) -> str | None:
        return self._key

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # --- naming a frame -----------------------------------------------------

    def frame_index_at(self, position: float) -> int:
        """The frame a fraction of the way through the video."""
        frame_idx = int(position * self._frame_count)
        # A slider at its maximum maps one past the last frame; clamping keeps
        # every reported index one a researcher and a technician can both name.
        last = self._frame_count - 1
        return max(0, min(frame_idx, last)) if last >= 0 else 0

    def position_of(self, frame_index: int) -> float:
        """Inverse of :meth:`frame_index_at`, for frame stepping.

        Aims at the middle of the frame's slice of the slider so that rounding
        cannot land the result back on a neighbouring frame.
        """
        if self._frame_count <= 0:
            return 0.0
        return (frame_index + 0.5) / self._frame_count

    def timestamp_ms(self, frame_index: int) -> int:
        """Position of *frame_index* in the recording, in milliseconds.

        Derived from the frame index rather than read back from the capture:
        the capture's position is a side effect of the last read, which the
        detector and the decoder both perform.
        """
        if self._fps <= 0:
            return 0
        return int(frame_index / self._fps * 1000)

    # --- the pixels ---------------------------------------------------------

    def decode(self, frame_index: int) -> ndarray | None:
        """Read one frame, or None if there is nothing to read.

        The returned array is the caller's to draw on. An untouched copy is
        kept as :attr:`raw_frame`, because a marked frame must be stored
        unannotated however it ends up looking on screen.
        """
        capture = self._cap
        if capture is None:
            logger.warning("decode: no video is open")
            return None
        image = capture.read(frame_index)
        if image is None:
            logger.warning("decode: cannot read frame %d", frame_index)
            return None
        self._decoded_frame_index = frame_index
        self._raw_frame = image.copy()
        return image

    @property
    def raw_frame(self) -> ndarray | None:
        """The last decoded frame, before anything drew on it."""
        return self._raw_frame

    @property
    def decoded_frame_index(self) -> int | None:
        """Which frame :attr:`raw_frame` is, or None if none has been read."""
        return self._decoded_frame_index
