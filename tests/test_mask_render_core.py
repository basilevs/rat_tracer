"""Direct unit tests for MaskRenderCore -- the Humble Object extraction of
VideoMasker's paused/playing render decision logic (see rat_tracer.ui).

Unlike test_video_masker.py, none of this needs Qt at all: no QObject, no
QThread, no QTimer, no monkeypatching of Qt's scheduler. MaskRenderCore is a
plain class, so its logic is driven directly and synchronously by calling its
methods.

It knows nothing about modes, so neither does anything here: what draws over a
frame arrives as a plain callable. The core's job is deciding *which* frame is
displayed and *when* it is decoded again, never what ends up on it -- the tests
for that live with the modes, in test_video_review.py.
"""

from pathlib import Path
from typing import override

import numpy as np
from numpy import ndarray
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore

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


class _Painter:
    """Stands in for whatever draws over a frame; records what it was given."""

    def __init__(self, paint: int = 255):
        self.paint = paint
        self.drawn: list[int] = []

    def __call__(self, image: ndarray, frame_index: int) -> None:
        self.drawn.append(frame_index)
        image[:] = self.paint


def _open(core: MaskRenderCore, cap: _FakeCapture) -> None:
    """Open a video the way a review does, identity included.

    The fingerprint arrives separately in production -- the cumulative pass
    computes it on its own thread -- but every test here wants a fully named
    video, so it is applied at once.
    """
    core.open(cap, _VIDEO)
    core.identify(_KEY)


# --- opening and closing ----------------------------------------------------


def test_reset_forgets_the_video_it_had_open():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))

    core.reset()

    assert not core.video_open
    assert core.video_key is None, "video1's identity must not be inherited"
    assert core.video_path is None


def test_reset_lets_the_next_video_be_seeked_to_the_same_position():
    """Regression: ``reset()`` used to clear the *applied* position and leave
    the *requested* one, so opening a second video and seeking to wherever the
    first was left looked like no seek at all -- and a seek that schedules
    nothing never repaints."""
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))
    core.set_playing(False)
    assert core.set_position(0.55)
    core.render_now()

    core.reset()
    _open(core, _FakeCapture(total_frames=10))
    core.set_playing(False)

    assert core.set_position(0.55), (
        "the position video1 was left at must still count as a seek in video2"
    )


# --- decoding and painting --------------------------------------------------


def test_the_decoded_frame_is_handed_to_the_painter():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))
    core.set_playing(False)
    painter = _Painter(paint=7)

    core.set_position(0.5)
    outcome = core.render_now(painter)

    assert outcome.image is not None
    assert np.all(outcome.image == 7), "the painter drew on the frame"
    assert painter.drawn == [5], "and was told which frame it was drawing"


def test_nothing_is_decoded_when_nothing_has_moved():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))
    core.set_playing(False)
    core.set_position(0.5)
    core.render_now()

    outcome = core.render_now()

    assert not outcome.should_emit


def test_a_wanted_repaint_decodes_again_where_it_is():
    """What a mode asks for when something it was waiting for has arrived: the
    position has not moved, but what belongs on screen has changed."""
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))
    core.set_playing(False)
    core.set_position(0.5)
    core.render_now()
    painter = _Painter()

    outcome = core.render_now(painter, repaint_wanted=True)

    assert outcome.should_emit
    assert painter.drawn == [5], "redrawn at the same frame"


def test_a_wanted_repaint_works_as_the_very_first_render():
    """There is no requested position yet, so the repaint has to happen where
    the video already is rather than being dropped."""
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=10))
    core.set_playing(False)

    outcome = core.render_now(_Painter(), repaint_wanted=True)

    assert outcome.should_emit
    assert outcome.image is not None


def test_the_raw_frame_is_kept_before_the_painter_draws_on_it():
    """Every overlay mutates in place, and a marked frame must be stored
    without annotation -- masked or box-annotated pixels are unusable as
    training data."""
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=100))
    core.set_playing(False)
    core.set_position(0.5)

    outcome = core.render_now(_Painter(paint=255))

    assert core.raw_frame is not None
    assert outcome.image is not None
    assert not np.array_equal(core.raw_frame, outcome.image), "the painter drew on a copy"
    assert np.array_equal(core.raw_frame, np.full((_H, _W, 3), 50 * 5, dtype=np.uint8))


def test_a_video_that_cannot_be_read_still_reports_something_to_show():
    core = MaskRenderCore()
    core.set_playing(False)

    outcome = core.render_now(_Painter(), repaint_wanted=True)

    assert outcome.should_emit
    assert outcome.image is None, "the empty placeholder, not a stale frame"


# --- position, index and stepping -------------------------------------------


def test_frame_index_and_timestamp_come_from_the_position():
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=1000))  # 25 fps
    core.set_playing(False)
    core.set_position(0.5)

    assert core.current_frame_index == 500
    assert core.timestamp_ms(500) == 20_000


def test_the_displayed_frame_lags_the_seeked_one_until_a_render():
    """``position`` answers "what is on screen"; ``current_position`` answers
    "where are we" -- a detection must be requested for the frame just seeked
    to, not the one still being displayed."""
    core = MaskRenderCore()
    _open(core, _FakeCapture(total_frames=1000))
    core.set_playing(False)
    core.set_position(0.1)
    core.render_now()

    core.set_position(0.5)

    assert core.displayed_frame_index == 100, "still showing the old frame"
    assert core.current_frame_index == 500, "but that is where we are"
    core.render_now()
    assert core.displayed_frame_index == 500


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
    core.set_playing(False)
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
