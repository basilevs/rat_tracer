import zlib

from numpy import bool_, dtype, frombuffer, ndarray, packbits, uint8, unpackbits, zeros

from rat_tracer.lib import Synchronized

type MaskFrame = ndarray[tuple[int, int], dtype[bool_]]


class CoverageHistory(Synchronized):
    """Stores the history of visited pixels across frames."""

    def __init__(self):
        self.width: int | None = None
        self.height: int | None = None
        self.visited = None
        self.history: list[bytes] = []

    def clear(self):
        self.width = None
        self.height = None
        self.visited = None
        self.history.clear()

    def _ensure_initialized(self, width: int, height: int):
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.visited = zeros((height, width), dtype=bool)
            self.history = []

    def _encode(self, frame: MaskFrame) -> bytes:
        """Compress a binary mask into a bytes blob using zlib (DEFLATE).

        Bits are packed 8-per-byte before compression, so long runs of the
        contiguous blobs collapse into repeated bytes that DEFLATE encodes
        cheaply. The frame shape is taken from the history dimensions on decode.
        """
        packed = packbits(frame)
        return zlib.compress(packed.tobytes(), level=9)

    def _decode(self, blob: bytes) -> MaskFrame:
        """Inverse of :meth:`_encode`."""
        assert self.height is not None and self.width is not None
        packed = frombuffer(zlib.decompress(blob), dtype=uint8)
        bits = unpackbits(packed, count=self.height * self.width)
        return bits.view(bool_).reshape(self.height, self.width)

    def append(self, presence_frame: MaskFrame) -> MaskFrame:
        # TODO: release the read lock while doing computation
        height, width = presence_frame.shape[:2]
        self._ensure_initialized(width, height)
        self.visited |= presence_frame.astype(bool)
        self.history.append(self._encode(self.visited))
        return self.visited

    def __getitem__(self, frame_idx: int) -> MaskFrame:
        if frame_idx < 0 or frame_idx >= len(self.history):
            raise IndexError("Frame index out of range")
        return self._decode(self.history[frame_idx])

    def __len__(self) -> int:
        return len(self.history)
