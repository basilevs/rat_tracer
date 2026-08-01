"""Direct unit tests for MaskRenderCore -- the Humble Object extraction of
VideoMasker's paused/playing render decision logic (see rat_tracer.ui).

Unlike test_video_masker.py, none of this needs Qt at all: no QObject, no
QThread, no QTimer, no monkeypatching of Qt's scheduler. MaskRenderCore is a
plain class, so its logic is driven directly and synchronously by calling its
methods -- the same two scenarios from repro.log, without the caveat (noted
in test_video_masker.py's worker_harness docstring) that faking QThread/QTimer
to run synchronously changes real cross-thread timing semantics.

What is drawn over a frame belongs to the active mode, so most tests here use
a fake one: the core's job is deciding *which* frame is displayed and *when* it
must be drawn again, not what appears on it.
"""

from pathlib import Path
from typing import override

import numpy as np
from numpy import ndarray
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore
from rat_tracer.review_modes import CoverageMode

_H, _W = 8, 12
_VIDEO = Path("2026-07-30_run3.mp4")
_KEY = "cafe1234"


class _FakeCapture(FrameCapture):
    """Stands in for cv2.VideoCapture: fixed frame count, solid frames."""

    def __init__(self, total_frames: int, fps: float = 25.0):
        self.total_frames = total_frames
        self.fps_value = fps
        self.frame_idx = 0

    def frame_count(self) -> int:
        return self.total_frames

    def fps(self) -> float:
        return self.fps_value

    @override
    def read(self, frame_idx: int) -> ndarray | None:
        self.frame_idx = frame_idx
        shade = min(255, self.frame_idx * 5)
        return np.full((_H, _W, 3), shade, dtype=np.uint8)


class _FakeMode:
    """A mode that paints a fixed value, and can claim to be incomplete."""

    def __init__(self, complete: bool = True, paint: int = 255):
        self.complete = complete
        self.paint = paint
        self.drawn: list[int] = []
        self.entered_count = 0
        self.left_count = 0

    def draw(self, image: ndarray, frame_index: int) -> bool:
        self.drawn.append(frame_index)
        image[:] = self.paint
        return self.complete

    def repaint_needed(self, frame_index: int, drawn: bool) -> bool:
        return not drawn and self.complete

    def entered(self) -> None:
        self.entered_count += 1

    def left(self) -> None:
        self.left_count += 1


def _open(core: MaskRenderCore, cap: _FakeCapture) -> None:
    """Open a video the way a session does, identity included."""
    core.open(cap, _VIDEO, _KEY)


def _append_mask(core: MaskRenderCore) -> None:
    core.history.append(np.zeros((_H, _W), dtype=bool))


def test_seek_while_uncached_video_processes_then_mask_applies():
    """A seek landing before any frames are processed renders bare first,
    then re-renders with the mask once the background pass reaches that
    frame -- mirrors test_video_masker's uncached-video scenario."""
    core = MaskRenderCore()
    core.mode = CoverageMode(core.history)
    _open(core, _FakeCapture(total_frames=10))
    core.playing = False

    assert core.set_position(0.55)  # frame index 5 of 10
    outcome = core.render_now()
    assert outcome.should_emit
    assert outcome.image is not None
    assert not core.overlay_complete, "frame 5 isn't processed yet -- must render bare"

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
    assert core.overlay_complete, "frame 5 is processed now -- must re-render with the mask"


def test_reset_clears_the_overlay_flag_for_the_next_video():
    """Regression test for the repro.log bug: opening a second, already
    fully cached video while paused must not inherit the first video's
    completed overlay, or the paused branch never re-renders
    (see rat_tracer.ui.VideoMasker.reset -- ebd5674 fixed this)."""
    core = MaskRenderCore()
    core.mode = CoverageMode(core.history)
    _open(core, _FakeCapture(total_frames=10))
    core.playing = False
    core.set_position(0.55)
    core.render_now()
    for _ in range(10):
        _append_mask(core)
    assert core.frame_ready()
    core.render_now()
    assert core.overlay_complete, "sanity check: video1's paused render should land masked"

    core.reset()
    assert not core.overlay_complete, "reset() must clear the stale flag from video1"

    _open(core, _FakeCapture(total_frames=6))
    for _ in range(6):
        _append_mask(core)  # video2 was already fully processed (cached) before

    assert core.frame_ready(), (
        "paused, overlay not complete, position behind the processed frontier -- "
        "must ask the caller to schedule a render"
    )
    outcome = core.render_now()
    assert outcome.should_emit
    assert outcome.image is not None, (
        "opening a fully-cached video while paused must produce a real frame, "
        "not the empty placeholder left over from reset()"
    )
    assert core.overlay_complete


# --- the active mode --------------------------------------------------------


def test_the_frame_is_handed_to_the_active_mode_to_draw():
    core = MaskRenderCore(mode=_FakeMode(paint=7))
    _open(core, _FakeCapture(total_frames=10))
    core.playing = False

    core.set_position(0.5)
    outcome = core.render_now()

    assert outcome.image is not None
    assert np.all(outcome.image == 7), "the mode painted the frame"
    assert core.overlay_complete


def test_an_incomplete_overlay_leaves_the_core_expecting_a_repaint():
    core = MaskRenderCore(mode=_FakeMode(complete=False))
    _open(core, _FakeCapture(total_frames=10))
    core.playing = False
    core.set_position(0.5)

    core.render_now()

    assert not core.overlay_complete


def test_switching_mode_tells_both_modes_and_forces_a_repaint():
    first, second = _FakeMode(paint=1), _FakeMode(paint=2)
    core = MaskRenderCore(mode=first)
    _open(core, _FakeCapture(total_frames=10))
    core.playing = False
    core.set_position(0.5)
    core.render_now()

    assert core.set_mode(second), "a mode swap must ask for a repaint"
    outcome = core.render_now()

    assert first.left_count == 1
    assert second.entered_count == 1
    assert outcome.should_emit
    assert outcome.image is not None
    assert np.all(outcome.image == 2)


def test_switching_to_the_same_mode_changes_nothing():
    mode = _FakeMode()
    core = MaskRenderCore(mode=mode)
    _open(core, _FakeCapture(total_frames=10))

    assert not core.set_mode(mode)
    assert mode.entered_count == 0


def test_the_raw_frame_is_kept_before_the_mode_draws_on_it():
    """Every overlay mutates in place, and a marked frame must be stored
    without annotation -- masked or box-annotated pixels are unusable as
    training data."""
    core = MaskRenderCore(mode=_FakeMode(paint=255))
    _open(core, _FakeCapture(total_frames=100))
    core.playing = False
    core.set_position(0.5)

    outcome = core.render_now()

    assert core.raw_frame is not None
    assert outcome.image is not None
    assert not np.array_equal(core.raw_frame, outcome.image), "the mode drew on a copy"
    assert np.array_equal(core.raw_frame, np.full((_H, _W, 3), 50 * 5, dtype=np.uint8))


# --- position, index and stepping -------------------------------------------


def test_frame_index_and_timestamp_come_from_the_position():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=1000))  # 25 fps
    core.playing = False
    core.set_position(0.5)

    assert core.current_frame_index == 500
    assert core.timestamp_ms(500) == 20_000


def test_a_video_without_a_frame_rate_reports_a_zero_timestamp():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10, fps=0.0))

    assert core.timestamp_ms(5) == 0


def test_the_last_frame_is_reachable_and_never_out_of_range():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))

    assert core.position_to_frame_index(1.0) == 9
    assert core.position_to_frame_index(0.0) == 0


def test_stepping_moves_exactly_one_frame_and_stops_at_the_ends():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))
    core.playing = False
    core.set_position(core.frame_index_to_position(5))

    forward = core.step_frame(1)
    assert forward is not None
    core.set_position(forward)
    core.render_now()
    assert core.current_frame_index == 6

    back = core.step_frame(-1)
    assert back is not None
    core.set_position(back)
    core.render_now()
    assert core.current_frame_index == 5

    core.set_position(core.frame_index_to_position(0))
    assert core.step_frame(-1) is None
    core.set_position(core.frame_index_to_position(9))
    assert core.step_frame(1) is None


def test_stepping_without_a_video_does_nothing():
    assert MaskRenderCore().step_frame(1) is None
