"""Direct unit tests for VideoFile -- naming a frame, and decoding one.

None of this needs Qt: no QObject, no QThread, no QTimer, no monkeypatching of
Qt's scheduler. VideoFile is a plain class with no collaborators at all, so
every question it answers is driven directly and synchronously.

Nothing here is about reviewing. When a frame needs decoding again, what gets
drawn over it and where the researcher is are all the review's, and their tests
live in test_video_review.py.
"""

from pathlib import Path
from typing import override

import numpy as np
from numpy import ndarray
from rat_tracer.video_file import FrameCapture, VideoFile

_H, _W = 8, 12
_VIDEO = Path("2026-07-30_run3.mp4")
_KEY = "cafe1234"


class _FakeCapture(FrameCapture):
    """Stands in for cv2.VideoCapture: fixed frame count, solid frames."""

    def __init__(self, total_frames: int, fps: float = 25.0, readable: bool = True):
        self.total_frames = total_frames
        self.fps_value = fps
        self.readable = readable
        self.reads: list[int] = []

    def frame_count(self) -> int:
        return self.total_frames

    def fps(self) -> float:
        return self.fps_value

    @override
    def read(self, frame_idx: int) -> ndarray | None:
        self.reads.append(frame_idx)
        if not self.readable:
            return None
        return np.full((_H, _W, 3), min(255, frame_idx * 5), dtype=np.uint8)


def _open(video: VideoFile, cap: _FakeCapture) -> None:
    """Open a video the way a review does, identity included.

    The fingerprint arrives separately in production -- the cumulative pass
    computes it on its own thread -- but every test here wants a fully named
    video, so it is applied at once.
    """
    video.open(cap, _VIDEO)
    video.identify(_KEY)


# --- opening and closing ----------------------------------------------------


def test_an_opened_video_reports_what_it_is():
    video = VideoFile()

    _open(video, _FakeCapture(total_frames=10))

    assert video.is_open
    assert video.path == _VIDEO
    assert video.key == _KEY
    assert video.frame_count == 10


def test_nothing_of_a_closed_video_reaches_the_next_one():
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=10))
    video.decode(5)

    video.close()

    assert not video.is_open
    assert video.path is None
    assert video.key is None, "video1's identity must not be inherited"
    assert video.frame_count == 0
    assert video.raw_frame is None
    assert video.decoded_frame_index is None


def test_a_video_is_unnamed_until_the_pass_fingerprints_it():
    """The fingerprint costs a full read of the file, so opening does not wait
    for it -- and nothing can be marked in the meantime."""
    video = VideoFile()
    video.open(_FakeCapture(total_frames=10), _VIDEO)

    assert video.is_open
    assert video.key is None

    video.identify(_KEY)
    assert video.key == _KEY


# --- decoding ---------------------------------------------------------------


def test_decoding_returns_the_frame_and_keeps_an_untouched_copy():
    """Every overlay mutates in place, and a marked frame must be stored
    without annotation -- masked or box-annotated pixels are unusable as
    training data."""
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=100))

    image = video.decode(50)

    assert image is not None
    assert video.decoded_frame_index == 50
    image[:] = 255  # the caller draws all over it
    assert video.raw_frame is not None
    assert not np.array_equal(video.raw_frame, image), "the caller drew on a copy"
    assert np.array_equal(video.raw_frame, np.full((_H, _W, 3), 50 * 5, dtype=np.uint8))


def test_decoding_without_a_video_answers_nothing():
    assert VideoFile().decode(0) is None


def test_a_frame_that_cannot_be_read_answers_nothing():
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=10, readable=False))

    assert video.decode(5) is None
    assert video.decoded_frame_index is None, "nothing was decoded, so nothing is named"


# --- naming a frame ---------------------------------------------------------


def test_a_position_names_the_frame_it_lands_on():
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=1000))

    assert video.frame_index_at(0.5) == 500
    assert video.timestamp_ms(500) == 20_000  # 25 fps


def test_the_last_frame_is_reachable_and_never_out_of_range():
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=10))

    assert video.frame_index_at(1.0) == 9, "a slider at its maximum must still name a frame"
    assert video.frame_index_at(0.0) == 0


def test_naming_a_frame_of_no_video_is_harmless():
    video = VideoFile()

    assert video.frame_index_at(0.5) == 0
    assert video.position_of(5) == 0.0
    assert video.timestamp_ms(5) == 0


def test_a_frame_index_round_trips_through_its_position():
    """Stepping converts both ways, so rounding must not land on a neighbour."""
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=1000))

    for frame_index in (0, 1, 499, 500, 998, 999):
        assert video.frame_index_at(video.position_of(frame_index)) == frame_index


def test_a_video_without_a_frame_rate_reports_a_zero_timestamp():
    video = VideoFile()
    _open(video, _FakeCapture(total_frames=10, fps=0.0))

    assert video.timestamp_ms(5) == 0
