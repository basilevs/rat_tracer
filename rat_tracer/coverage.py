import contextlib
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from numpy import bool_, dtype, frombuffer, ndarray, packbits, uint8, unpackbits, zeros

type MaskFrame = ndarray[tuple[int, int], dtype[bool_]]
# A history slot holds either the compressed blob or, briefly, the raw mask
# while a worker thread is still compressing it in the background.
type HistorySlot = bytes | MaskFrame

# Highest zlib level: best ratio, used for the live history and pickled deltas.
# Compression runs on background workers (see ``_COMPRESS_WORKERS``), so a higher
# level no longer stalls the inference thread.
_MAX_ENCODE_LEVEL = 9
# Fast zlib level for rebuilding RAM blobs on unpickle, where the per-frame
# re-encode dominates load time and the ratio only affects transient RAM use.
_RAM_ENCODE_LEVEL = 1
# Worker threads that compress appended masks off the caller's (inference)
# thread. zlib and numpy both release the GIL during the heavy work, so threads
# genuinely run on other cores. Leave one core for the inference/main thread.
_COMPRESS_WORKERS = max(1, (os.cpu_count() or 2) - 1)


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
        self.history: list[HistorySlot] = []
        # ``append`` stores the raw mask immediately and offloads compression to
        # these workers so the inference thread is not blocked. ``_generation``
        # invalidates in-flight tasks after a ``clear``.
        self._generation = 0
        self._executor = ThreadPoolExecutor(
            max_workers=_COMPRESS_WORKERS,
            thread_name_prefix="coverage-compress",
        )

    def __getstate__(self):
        """Pickle the history as inter-frame deltas.

        The in-RAM ``history`` stores each frame as a full, independently
        decodable blob so ``__getitem__`` has O(1) random access. That is very
        redundant on disk because the cumulative mask barely changes between
        consecutive frames. For pickling we therefore XOR each frame against the
        previous one and store only those (sparse) deltas; ``visited`` is
        dropped and rebuilt on load. The RAM representation is unaffected.
        """
        with self._lock:
            state = self.__dict__.copy()
            height, width = self.height, self.width
            blobs = list(self.history)
        del state["_lock"]
        del state["_executor"]
        state.pop("_generation", None)
        state.pop("history", None)
        state.pop("visited", None)

        deltas: list[bytes] = []
        if blobs:
            assert height is not None and width is not None
            prev = zeros((height, width), dtype=bool)
            for slot in blobs:
                # A slot may still be a raw mask if its background compression
                # has not finished yet; use it directly in that case.
                curr = slot if isinstance(slot, ndarray) else self._decode(slot, height, width)
                deltas.append(self._encode(curr ^ prev))
                prev = curr
        state["_delta_history"] = deltas
        return state

    def __setstate__(self, state: dict):
        deltas = state.pop("_delta_history", None)
        self.__dict__.update(state)
        assert not getattr(self, "_lock", None), "unpickled instance already has a lock"
        self._lock = Lock()
        self._generation = 0
        self._executor = ThreadPoolExecutor(
            max_workers=_COMPRESS_WORKERS,
            thread_name_prefix="coverage-compress",
        )
        if deltas is None:
            # Legacy / non-delta pickle: ``history`` and ``visited`` are already
            # restored verbatim by the ``__dict__`` update above.
            return

        height, width = self.height, self.width
        history: list[HistorySlot] = []
        visited = None
        if deltas:
            assert height is not None and width is not None
            cum = zeros((height, width), dtype=bool)
            for delta_blob in deltas:
                cum ^= self._decode(delta_blob, height, width)
                # Rebuild the RAM blobs at a fast zlib level: this path runs once
                # per frame and dominates unpickling time, while the level only
                # affects RAM size (decode speed for random access is unchanged).
                history.append(self._encode(cum, level=_RAM_ENCODE_LEVEL))
            visited = cum
        elif height is not None and width is not None:
            visited = zeros((height, width), dtype=bool)
        self.history = history
        self.visited = visited

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
            # Invalidate any in-flight compression tasks and swap in a fresh
            # executor so their results are dropped instead of writing into the
            # cleared history.
            self._generation += 1
            self.width = None
            self.height = None
            self.visited = None
            self.history.clear()
            old_executor = self._executor
            self._executor = ThreadPoolExecutor(
                max_workers=_COMPRESS_WORKERS,
                thread_name_prefix="coverage-compress",
            )
        # Cancel queued tasks and stop the old workers without holding the lock.
        old_executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_initialized(self, width: int, height: int):
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.visited = zeros((height, width), dtype=bool)
            self.history = []

    def _encode(self, frame: MaskFrame, level: int = _MAX_ENCODE_LEVEL) -> bytes:
        """Compress a binary mask into a bytes blob using zlib (DEFLATE).

        Bits are packed 8-per-byte before compression, so long runs of the
        contiguous blobs collapse into repeated bytes that DEFLATE encodes
        cheaply. The frame shape is taken from the history dimensions on decode.
        ``level`` trades compression ratio for speed; any level decodes
        identically, so callers on hot paths may pick a faster one.
        """
        packed = packbits(frame)
        return zlib.compress(packed.tobytes(), level=level)

    def _decode(self, blob: bytes, height: int, width: int) -> MaskFrame:
        """Inverse of :meth:`_encode`."""
        packed = frombuffer(zlib.decompress(blob), dtype=uint8)
        bits = unpackbits(packed, count=height * width)
        return bits.view(bool_).reshape(height, width)

    def _compress_slot(self, generation: int, idx: int, mask: MaskFrame) -> None:
        """Compress *mask* on a worker thread and store it back in its slot.

        Runs off the caller's thread. The result is discarded if a ``clear``
        happened meanwhile (``generation`` mismatch) or the slot was already
        replaced, so a stale task can never corrupt the current history.
        """
        blob = self._encode(mask)
        with self._lock:
            if (
                generation == self._generation
                and idx < len(self.history)
                and self.history[idx] is mask
            ):
                self.history[idx] = blob

    def append(self, presence_frame: MaskFrame) -> MaskFrame:
        height, width = presence_frame.shape[:2]
        with self._lock:
            self._ensure_initialized(width, height)
            self.visited |= presence_frame.astype(bool)
            snapshot = self.visited.copy()
            # Store the raw mask immediately so the caller returns without
            # blocking on compression; a worker replaces it with the blob later.
            idx = len(self.history)
            self.history.append(snapshot)
            generation = self._generation
            executor = self._executor
        # Executor may be shut down by a concurrent clear(); the raw slot it
        # belonged to has been cleared too, so a failed submit is safe to drop.
        with contextlib.suppress(RuntimeError):
            executor.submit(self._compress_slot, generation, idx, snapshot)
        return snapshot

    def __getitem__(self, frame_idx: int) -> MaskFrame:
        with self._lock:
            if frame_idx < 0 or frame_idx >= len(self.history):
                raise IndexError("Frame index out of range")
            assert self.height is not None and self.width is not None
            slot = self.history[frame_idx]
            height, width = self.height, self.width
        # A slot not yet compressed is returned as an owned copy; otherwise
        # decompress outside the lock so a concurrent writer can keep encoding.
        if isinstance(slot, ndarray):
            return slot.copy()
        return self._decode(slot, height, width)

    def __len__(self) -> int:
        with self._lock:
            return len(self.history)
