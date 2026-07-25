
from rat_tracer.lib import Synchronized

from numpy import zeros
from numpy import ndarray, dtype, bool_

type MaskFrame = ndarray[tuple[int, int], dtype[bool_]]

class CoverageHistory(Synchronized):
    """ Stores the history of visited pixels across frames. """
    def __init__(self):
        self.width = None
        self.height = None
        self.visited = None
        self.history = []

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
    
    def append(self, presence_frame: MaskFrame) -> MaskFrame:
        # TODO: release the read lock while doing computation
        height, width = presence_frame.shape[:2]
        self._ensure_initialized(width, height)
        self.visited |= presence_frame.astype(bool)
        self.history.append(self.visited.copy())
        return self.visited
    
    def __getitem__(self, frame_idx: int) -> MaskFrame:
        if frame_idx < 0 or frame_idx >= len(self.history):
            raise IndexError("Frame index out of range")
        return self.history[frame_idx]
    
    def __len__(self) -> int:
        return len(self.history)
