"""Direct unit tests for MaskRenderCore -- the Humble Object extraction of
VideoMasker's paused/playing render decision logic (see rat_tracer.ui).

Unlike test_video_masker.py, none of this needs Qt at all: no QObject, no
QThread, no QTimer, no monkeypatching of Qt's scheduler. MaskRenderCore is a
plain class, so its logic is driven directly and synchronously by calling its
methods -- the same two scenarios from repro.log, without the caveat (noted
in test_video_masker.py's worker_harness docstring) that faking QThread/QTimer
to run synchronously changes real cross-thread timing semantics.
"""

from typing import override

import numpy as np
from numpy import ndarray
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore

_H, _W = 8, 12


class _FakeCapture(FrameCapture):
    """Stands in for cv2.VideoCapture: fixed frame count, solid frames."""

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.frame_idx = 0

    def frame_count(self) -> int:
        return self.total_frames

    @override
    def read(self, frame_idx: int) -> ndarray | None:
        self.frame_idx = frame_idx
        shade = min(255, self.frame_idx * 5)
        return np.full((_H, _W, 3), shade, dtype=np.uint8)


def _append_mask(core: MaskRenderCore) -> None:
    core.history.append(np.zeros((_H, _W), dtype=bool))


def test_seek_while_uncached_video_processes_then_mask_applies():
    """A seek landing before any frames are processed renders bare first,
    then re-renders with the mask once the background pass reaches that
    frame -- mirrors test_video_masker's uncached-video scenario."""
    core = MaskRenderCore()
    core.open(_FakeCapture(total_frames=10))
    core.playing = False

    assert core.set_position(0.55)  # frame index 5 of 10
    outcome = core.render_now()
    assert outcome.should_emit
    assert outcome.image is not None
    assert not core.mask_rendered, "frame 5 isn't processed yet -- must render bare"

    rescheduled = False
    for _ in range(10):
        _append_mask(core)
        if core.frame_ready():
            rescheduled = True
            break
    assert rescheduled, "processed_position should eventually overtake position 0.55"

    outcome = core.render_now()
    assert outcome.should_emit
    assert outcome.image is not None
    assert core.mask_rendered, "frame 5 is processed now -- must re-render with the mask"


def test_reset_clears_mask_rendered_for_next_video():
    """Regression test for the repro.log bug: opening a second, already
    fully cached video while paused must not inherit the first video's
    mask_rendered=True, or _on_frame_ready's paused branch never re-renders
    (see rat_tracer.ui.VideoMasker.reset -- ebd5674 fixed this)."""
    core = MaskRenderCore()
    core.open(_FakeCapture(total_frames=10))
    core.playing = False
    core.set_position(0.55)
    core.render_now()
    for _ in range(10):
        _append_mask(core)
    assert core.frame_ready()
    outcome = core.render_now()
    assert core.mask_rendered, "sanity check: video1's paused render should land masked"

    core.reset()
    assert not core.mask_rendered, "reset() must clear the stale flag from video1"

    core.open(_FakeCapture(total_frames=6))
    for _ in range(6):
        _append_mask(core)  # video2 was already fully processed (cached) before

    assert core.frame_ready(), (
        "paused, mask not rendered, position behind the processed frontier -- "
        "must ask the caller to schedule a render"
    )
    outcome = core.render_now()
    assert outcome.should_emit
    assert outcome.image is not None, (
        "opening a fully-cached video while paused must produce a real frame, "
        "not the empty placeholder left over from reset()"
    )
    assert core.mask_rendered
