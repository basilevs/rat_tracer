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
from rat_tracer.bad_frames import Detection
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore

_H, _W = 8, 12


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


# --- problem reporting mode -------------------------------------------------


def _paused_core(total_frames: int = 100) -> MaskRenderCore:
    core = MaskRenderCore()
    core.open(_FakeCapture(total_frames=total_frames))
    core.set_playing(False)
    return core


def _fill_history(core: MaskRenderCore, frames: int) -> None:
    """Give the cumulative pass a visible, non-empty mask over the whole frame."""
    for _ in range(frames):
        core.history.append(np.ones((_H, _W), dtype=bool))


def test_problem_mode_hides_the_cumulative_mask_and_shows_this_frame_alone():
    """A red region on screen is the union of every detection so far, so a
    single frame's detection cannot be judged from it at all."""
    core = _paused_core()
    _fill_history(core, 100)
    core.set_position(0.5)
    core.render_now()
    assert core.mask_rendered, "sanity check: the cumulative mask is on screen"

    assert core.set_problem_mode(True)
    outcome = core.render_now()

    assert outcome.should_emit
    assert not core.mask_rendered
    assert outcome.image is not None
    # Nothing has been detected for this frame yet, so it is shown clean.
    assert not core.detection_rendered


def test_leaving_problem_mode_brings_the_cumulative_mask_back_unchanged():
    core = _paused_core()
    _fill_history(core, 100)
    core.set_position(0.5)
    core.render_now()
    before = len(core.history)

    core.set_problem_mode(True)
    core.render_now()
    assert core.set_problem_mode(False)
    core.render_now()

    assert core.mask_rendered
    assert len(core.history) == before, "the mode is a display state, not a recording state"


def test_detection_is_shown_for_a_frame_the_cumulative_pass_has_not_reached():
    """The hardest requirement in the feature: a failure is usually noticed
    seconds after the frame that caused it, so the researcher seeks backwards
    -- but may also seek ahead of the pass."""
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_position(0.9)  # frame 90, nothing processed at all
    core.render_now()

    assert core.needs_detection() == 90
    assert core.set_detection(90, Detection([[0.5, 0.5, 0.2, 0.2]], [0.8]))
    outcome = core.render_now()

    assert outcome.should_emit
    assert core.detection_rendered
    assert not core.mask_rendered


def test_a_frame_the_detector_found_nothing_in_is_still_markable():
    """A missed detection is the most important defect to report, so an empty
    result enables the control rather than leaving it disabled."""
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_position(0.3)
    core.render_now()
    assert not core.can_mark, "no answer yet"

    core.set_detection(core.current_frame_index, Detection())
    core.render_now()

    assert core.can_mark
    assert core.current_detection == Detection()


def test_marking_is_blocked_until_the_detection_is_on_screen():
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_position(0.3)
    core.render_now()

    assert not core.can_mark
    core.set_detection(core.current_frame_index, Detection([[0.1, 0.1, 0.1, 0.1]], [0.5]))
    assert not core.can_mark, "the answer has arrived but is not drawn yet"
    core.render_now()
    assert core.can_mark


def test_marking_is_blocked_outside_problem_mode_and_during_playback():
    core = _paused_core()
    _fill_history(core, 100)
    core.set_position(0.5)
    core.render_now()
    assert not core.can_mark, "no detection is on screen to have judged"

    core.set_problem_mode(True)
    core.set_detection(50, Detection())
    core.render_now()
    assert core.can_mark

    core.set_playing(True)
    assert not core.can_mark


def test_resuming_playback_leaves_problem_mode():
    core = _paused_core()
    core.set_problem_mode(True)

    core.set_playing(True)

    assert not core.problem_mode


def test_no_detection_is_drawn_during_playback():
    core = _paused_core()
    _fill_history(core, 100)
    core.set_detection(50, Detection([[0.5, 0.5, 0.2, 0.2]], [0.9]))
    core.problem_mode = True  # bypass the toggle: playback must be safe anyway
    core.playing = True

    core.set_position(0.5)
    outcome = core.render_now()

    assert outcome.should_emit
    assert not core.detection_rendered
    assert core.mask_rendered, "playback keeps reporting how far the pass has got"


def test_a_detection_arriving_for_another_frame_does_not_repaint():
    """The researcher navigated on while inference was running; the answer is
    kept for their return, but the screen must not flicker back."""
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_position(0.3)
    core.render_now()

    assert not core.set_detection(90, Detection())
    assert core.detection_for(90) == Detection()
    assert not core.can_mark


def test_a_cached_detection_is_not_requested_twice():
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_position(0.3)

    assert core.needs_detection() == 30
    core.set_detection(30, Detection())
    assert core.needs_detection() is None


def test_no_detection_is_requested_outside_problem_mode():
    core = _paused_core()
    core.set_position(0.3)
    assert core.needs_detection() is None


def test_the_raw_frame_is_kept_unannotated_for_saving():
    """Masked or box-annotated pixels are unusable as training data."""
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_position(0.5)
    core.set_detection(50, Detection([[0.5, 0.5, 0.4, 0.4]], [0.9]))
    outcome = core.render_now()

    assert core.raw_frame is not None
    assert outcome.image is not None
    assert not np.array_equal(core.raw_frame, outcome.image), "the box was drawn on a copy"
    assert np.array_equal(core.raw_frame, np.full((_H, _W, 3), 50 * 5, dtype=np.uint8))


def test_frame_index_and_timestamp_come_from_the_position():
    core = _paused_core(total_frames=1000)  # 25 fps
    core.set_position(0.5)

    assert core.current_frame_index == 500
    assert core.timestamp_ms(500) == 20_000


def test_a_video_without_a_frame_rate_reports_a_zero_timestamp():
    core = MaskRenderCore()
    core.open(_FakeCapture(total_frames=10, fps=0.0))

    assert core.timestamp_ms(5) == 0


def test_the_last_frame_is_reachable_and_never_out_of_range():
    core = _paused_core(total_frames=10)

    assert core.position_to_frame_index(1.0) == 9
    assert core.position_to_frame_index(0.0) == 0


def test_stepping_moves_exactly_one_frame_and_stops_at_the_ends():
    core = _paused_core(total_frames=10)
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


def test_reset_clears_detections_from_the_previous_video():
    core = _paused_core()
    core.set_problem_mode(True)
    core.set_detection(50, Detection([[0.5, 0.5, 0.2, 0.2]], [0.9]))

    core.reset()
    core.open(_FakeCapture(total_frames=100))

    assert core.detection_for(50) is None
    assert core.raw_frame is None
    assert not core.can_mark
