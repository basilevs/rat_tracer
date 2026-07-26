import zlib
from threading import Lock

from numpy import bool_, dtype, frombuffer, ndarray, packbits, uint8, unpackbits, zeros

type MaskFrame = ndarray[tuple[int, int], dtype[bool_]]


class CoverageHistory:
    """Stores the history of visited pixels across frames."""

    def __init__(self):
        # A single lock guards the shared state (``visited``, dimensions and the
        # ``history`` list), but it is only held for the cheap in-memory
        # mutations. The expensive (de)compression steps run without the lock
        # held, so readers can decode past frames while a writer is encoding a
        # new one, and vice versa.
        self._lock = Lock()
        self.width: int | None = None
        self.height: int | None = None
        self.visited = None
        self.history: list[bytes] = []

    def __getstate__(self):
        with self._lock:
            state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        assert not getattr(self, "_lock", None), "unpickled instance already has a lock"
        self._lock = Lock()

    def replace_with(self, other: "CoverageHistory") -> None:
        """Atomically replace this instance's state with *other*'s state."""
        with other._lock:
            width = other.width
            height = other.height
            visited = other.visited
            history = list(other.history)
        with self._lock:
            self.width = width
            self.height = height
            self.visited = visited
            self.history = history

    def clear(self):
        with self._lock:
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

    def _decode(self, blob: bytes, height: int, width: int) -> MaskFrame:
        """Inverse of :meth:`_encode`."""
        packed = frombuffer(zlib.decompress(blob), dtype=uint8)
        bits = unpackbits(packed, count=height * width)
        return bits.view(bool_).reshape(height, width)

    def append(self, presence_frame: MaskFrame) -> MaskFrame:
        height, width = presence_frame.shape[:2]
        with self._lock:
            self._ensure_initialized(width, height)
            self.visited |= presence_frame.astype(bool)
            snapshot = self.visited.copy()
        # Compress outside the lock so readers can access the history meanwhile.
        blob = self._encode(snapshot)
        with self._lock:
            self.history.append(blob)
        return snapshot

    def __getitem__(self, frame_idx: int) -> MaskFrame:
        with self._lock:
            if frame_idx < 0 or frame_idx >= len(self.history):
                raise IndexError("Frame index out of range")
            assert self.height is not None and self.width is not None
            blob = self.history[frame_idx]
            height, width = self.height, self.width
        # Decompress outside the lock so a concurrent writer can keep encoding.
        return self._decode(blob, height, width)

    def __len__(self) -> int:
        with self._lock:
            return len(self.history)
