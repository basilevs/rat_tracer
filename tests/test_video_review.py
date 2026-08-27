"""Tests for a video review -- the behaviour the UI merely displays.

Nothing here touches Qt, threads, model weights or the filesystem. The review
reaches its collaborators through protocols, so detection and storage are
fakes that answer when the test says so: the asynchrony is real (a request is
made, a completion is reported later) but its timing is the test's to choose.

That is the point of the class. These decisions used to be split between
``VideoMasker`` and ``MaskRenderCore`` and could only be exercised through a
Qt harness with faked threads.
"""

from pathlib import Path
from typing import override

import numpy as np
from numpy import ndarray
from rat_tracer.bad_frames import Detection, MarkRequest
from rat_tracer.mask_render_core import FrameCapture
from rat_tracer.video_review import ReviewListener, VideoReview

_H, _W = 8, 12
_VIDEO = Path("2026-07-30_run3.mp4")
_KEY = "cafe1234"
_BOX = [[0.5, 0.5, 0.2, 0.2]]


class _FakeCapture(FrameCapture):
    """Solid frames whose shade encodes the frame index."""

    def __init__(self, total_frames: int = 100, fps: float = 25.0):
        self.total_frames = total_frames
        self.fps_value = fps

    def frame_count(self) -> int:
        return self.total_frames

    def fps(self) -> float:
        return self.fps_value

    @override
    def read(self, frame_idx: int) -> ndarray | None:
        return np.full((_H, _W, 3), frame_idx % 256, dtype=np.uint8)


class _FakeDetection:
    """Answers only when the test says so, so the wait is observable."""

    def __init__(self, detection: Detection | None = None):
        self.detection = detection if detection is not None else Detection(_BOX, [0.9])
        self.requests: list[tuple[int, ndarray]] = []

    @property
    def model_id(self) -> str:
        return "test-model:v1"

    def request(self, frame_index: int, image: ndarray) -> None:
        self.requests.append((frame_index, image))


class _FakeStorage:
    """Records what it was asked to do; completions are reported by the test."""

    def __init__(self):
        self.marked: set[tuple[str, int]] = set()
        self.stored: list[MarkRequest] = []
        self.removed: list[tuple[str, int, str]] = []

    def is_marked(self, video_key: str, frame_index: int) -> bool:
        return (video_key, frame_index) in self.marked

    def store(self, request: MarkRequest) -> None:
        self.stored.append(request)

    def remove(self, video_key: str, frame_index: int, video_stem: str) -> None:
        self.removed.append((video_key, frame_index, video_stem))


class _Events:
    """Counts what the review told the UI."""

    def __init__(self):
        self.renders = 0
        self.states = 0
        self.stored: list[int] = []
        self.failed: list[int] = []

    def listener(self) -> ReviewListener:
        return ReviewListener(
            schedule_render=self._render,
            state_changed=self._state,
            mark_stored=self.stored.append,
            mark_failed=self.failed.append,
        )

    def _render(self) -> None:
        self.renders += 1

    def _state(self) -> None:
        self.states += 1


def _review(
    total_frames: int = 100,
) -> tuple[VideoReview, _FakeDetection, _FakeStorage, _Events]:
    events = _Events()
    detection, storage = _FakeDetection(), _FakeStorage()
    review = VideoReview(listener=events.listener(), storage=storage, detection=detection)
    review.open_video(_FakeCapture(total_frames), _VIDEO, _KEY)
    review.set_playing(False)
    return review, detection, storage, events


def _reach_judged_frame(review: VideoReview, detection: _FakeDetection, position=0.5) -> int:
    """Do what a researcher does: enter the mode, stop on a frame, get an answer."""
    review.seek(position)
    review.set_problem_mode(True)
    review.render_frame()  # draws the frame, which is when the request can go out
    frame_index, _image = detection.requests[-1]
    review.detection_ready(frame_index, detection.detection)
    review.render_frame()  # draws the answer
    return frame_index


# --- mode selection ---------------------------------------------------------


def test_entering_problem_mode_pauses():
    """A frame can only be judged if the researcher has stopped on it."""
    review, _detection, _storage, _events = _review()
    review.set_playing(True)

    review.set_problem_mode(True)

    assert not review.playing
    assert review.problem_mode


def test_resuming_playback_leaves_problem_mode():
    """The two answer unrelated questions, and playback draws no detections."""
    review, detection, _storage, _events = _review()
    _reach_judged_frame(review, detection)

    review.set_playing(True)

    assert not review.problem_mode
    assert not review.can_mark


def test_no_detection_is_requested_before_the_mode_is_entered():
    """A researcher who never reports a problem never pays for a second model."""
    review, detection, _storage, _events = _review()
    review.seek(0.5)
    review.render_frame()

    assert detection.requests == []


# --- detection --------------------------------------------------------------


def test_the_displayed_frame_is_asked_about_once():
    review, detection, _storage, _events = _review()
    _reach_judged_frame(review, detection)
    asked = len(detection.requests)

    review.render_frame()
    review.request_detection()

    assert len(detection.requests) == asked, "the same frame must not be asked about twice"


def test_the_image_sent_for_detection_is_a_copy_of_the_raw_frame():
    """Detection runs later, while rendering keeps mutating its own frame."""
    review, detection, _storage, _events = _review()
    _reach_judged_frame(review, detection)

    _frame_index, image = detection.requests[-1]

    assert review.render.raw_frame is not None
    assert np.array_equal(image, review.render.raw_frame)
    assert image is not review.render.raw_frame


def test_a_failed_detection_can_be_asked_about_again():
    """One failure must not disable the frame for the rest of the review."""
    review, detection, _storage, _events = _review()
    review.seek(0.5)
    review.set_problem_mode(True)
    review.render_frame()
    frame_index, _image = detection.requests[-1]

    review.detection_failed(frame_index)
    review.request_detection()

    assert len(detection.requests) == 2
    assert not review.can_mark


def test_an_answer_for_a_frame_already_left_does_not_repaint():
    review, detection, _storage, events = _review()
    _reach_judged_frame(review, detection, position=0.5)
    review.seek(0.9)
    renders = events.renders

    review.detection_ready(12, Detection())

    assert events.renders == renders


# --- what may be marked -----------------------------------------------------


def test_a_judged_frame_may_be_marked():
    review, detection, _storage, _events = _review()
    _reach_judged_frame(review, detection)

    assert review.can_mark
    assert not review.frame_marked


def test_a_frame_the_detector_found_nothing_in_may_be_marked():
    """A missed detection is the most important defect to report."""
    review, detection, _storage, _events = _review()
    detection.detection = Detection()
    _reach_judged_frame(review, detection)

    assert review.can_mark


def test_a_frame_may_not_be_marked_before_its_answer_arrives():
    review, detection, _storage, _events = _review()
    review.seek(0.5)
    review.set_problem_mode(True)
    review.render_frame()

    assert not review.can_mark, "the answer has not arrived"
    frame_index, _image = detection.requests[-1]
    review.detection_ready(frame_index, detection.detection)
    assert not review.can_mark, "the answer has arrived but is not drawn yet"
    review.render_frame()
    assert review.can_mark


def test_nothing_may_be_marked_without_storage():
    review = VideoReview(detection=_FakeDetection())
    review.open_video(_FakeCapture(), _VIDEO, _KEY)

    assert not review.can_mark
    assert not review.frame_marked


# --- marking ----------------------------------------------------------------


def test_marking_describes_the_frame_on_screen():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)

    review.mark()

    assert len(storage.stored) == 1
    request = storage.stored[0]
    assert request.frame_index == frame_index
    assert request.video_key == _KEY
    assert request.video_stem == "2026-07-30_run3"
    assert request.model_id == "test-model:v1"
    assert request.detection == Detection(_BOX, [0.9])
    assert request.timestamp_ms == int(frame_index / 25.0 * 1000)
    assert review.render.raw_frame is not None
    assert np.array_equal(request.image, review.render.raw_frame)
    assert request.image is not review.render.raw_frame, "storage runs later"


def test_marking_does_not_move_the_position():
    review, detection, _storage, _events = _review()
    _reach_judged_frame(review, detection)
    position, frame_index = review.position, review.frame_index

    review.mark()

    assert review.position == position
    assert review.frame_index == frame_index


def test_a_frame_being_stored_may_not_be_marked_again():
    """Storage is asynchronous; a second click would otherwise queue a second
    write before the first has landed."""
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)

    review.mark()
    assert not review.can_mark
    review.mark()
    assert len(storage.stored) == 1

    storage.marked.add((_KEY, frame_index))
    review.mark_stored(frame_index)
    assert review.frame_marked


def test_a_stored_frame_is_not_marked_twice():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)
    storage.marked.add((_KEY, frame_index))

    review.mark()

    assert storage.stored == []


def test_a_refused_mark_still_refreshes_the_control():
    """A click flips the control's own tick, so even a refused mark has to send
    it back to reporting what is on disk."""
    review, _detection, _storage, events = _review()
    states = events.states

    review.mark()  # not in problem mode: nothing to mark

    assert events.states > states


def test_a_failed_write_frees_the_frame_to_be_marked_again():
    review, detection, _storage, events = _review()
    frame_index = _reach_judged_frame(review, detection)
    review.mark()

    review.mark_failed(frame_index)

    assert events.failed == [frame_index]
    assert not review.frame_marked
    assert review.can_mark


# --- withdrawing ------------------------------------------------------------


def test_toggling_a_stored_frame_withdraws_it():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)
    storage.marked.add((_KEY, frame_index))

    review.toggle_mark()

    assert storage.removed == [(_KEY, frame_index, "2026-07-30_run3")]
    assert storage.stored == []


def test_toggling_an_unstored_frame_marks_it():
    review, detection, storage, _events = _review()
    _reach_judged_frame(review, detection)

    review.toggle_mark()

    assert len(storage.stored) == 1
    assert storage.removed == []


def test_a_removal_in_flight_blocks_a_second_one():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)
    storage.marked.add((_KEY, frame_index))

    review.unmark()
    review.unmark()

    assert len(storage.removed) == 1, "one removal must not append two rows"
    review.mark_removed(frame_index)
    storage.marked.discard((_KEY, frame_index))
    assert review.can_mark


def test_undo_removes_the_most_recent_mark():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)
    review.mark()
    review.mark_stored(frame_index)

    review.undo()

    assert storage.removed == [(_KEY, frame_index, "2026-07-30_run3")]


def test_undo_after_withdrawing_by_hand_removes_nothing_more():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)
    review.mark()
    review.mark_stored(frame_index)
    storage.marked.add((_KEY, frame_index))
    review.unmark()
    review.mark_removed(frame_index)
    storage.marked.discard((_KEY, frame_index))

    review.undo()

    assert len(storage.removed) == 1, "Undo had nothing left to remove"


def test_undo_without_a_mark_does_nothing():
    review, _detection, storage, _events = _review()

    review.undo()

    assert storage.removed == []


# --- navigation and readouts ------------------------------------------------


def test_stepping_moves_one_frame_and_pauses():
    review, _detection, _storage, _events = _review(total_frames=100)
    review.set_playing(True)
    review.seek(0.5)
    review.render_frame()
    start = review.frame_index

    review.step(1)

    assert review.frame_index == start + 1
    assert not review.playing
    review.step(-1)
    assert review.frame_index == start


def test_stepping_stops_at_the_ends():
    review, _detection, _storage, _events = _review(total_frames=10)
    review.seek(0.0)

    review.step(-1)

    assert review.frame_index == 0


def test_the_readouts_name_the_displayed_frame():
    review, _detection, _storage, _events = _review(total_frames=10_000)
    review.seek(0.5)

    assert review.frame_index == 5000
    assert review.time_text == "00:03:20"


def test_the_readouts_are_available_before_a_video_is_opened():
    review = VideoReview()

    assert review.time_text == "00:00:00"
    assert review.frame_index == 0
    assert not review.can_mark
    assert not review.video_open


def test_closing_a_video_forgets_its_marks_and_answers():
    review, detection, storage, _events = _review()
    frame_index = _reach_judged_frame(review, detection)
    review.mark()

    review.close_video()

    assert not review.video_open
    assert not review.can_mark
    assert not review.frame_marked
    review.undo()
    assert storage.removed == [], "a closed video's Undo has nothing to act on"
    assert review.problem.detection_for(frame_index) is None


# --- what each mode shows ---------------------------------------------------


def test_problem_mode_hides_the_cumulative_track():
    """A red region is the union of every detection so far, so a single
    frame's detection cannot be judged from it at all."""
    review, detection, _storage, _events = _review()
    for _ in range(100):
        review.history.append(np.ones((_H, _W), dtype=bool))  # everything visited
    detection.detection = Detection()  # nothing found here, so nothing is drawn
    review.seek(0.5)
    tracked = review.render_frame()
    assert tracked.image is not None
    assert review.render.raw_frame is not None
    assert not np.array_equal(tracked.image, review.render.raw_frame), (
        "sanity check: the cumulative track is painted over the frame"
    )

    _reach_judged_frame(review, detection)

    assert review.problem_mode
    review.render.force_repaint()
    shown = review.render.render_now().image
    assert shown is not None
    assert review.render.raw_frame is not None
    assert np.array_equal(shown, review.render.raw_frame), (
        "with the track hidden and nothing detected, the frame is shown clean"
    )


def test_leaving_problem_mode_brings_the_track_back_with_nothing_lost():
    review, detection, _storage, _events = _review()
    for _ in range(100):
        review.history.append(np.ones((_H, _W), dtype=bool))
    _reach_judged_frame(review, detection)
    before = len(review.history)

    review.set_problem_mode(False)
    review.render_frame()

    assert not review.problem_mode
    assert review.render.overlay_complete, "the cumulative track is drawn again"
    assert len(review.history) == before, "the mode is a display state, not a recording state"


def test_a_frame_the_pass_has_not_reached_still_gets_its_detection():
    """The hardest requirement in the feature: a failure is usually noticed
    seconds after the frame that caused it, and the researcher may also seek
    ahead of the pass."""
    review, detection, _storage, _events = _review()
    # Nothing processed at all -- the coverage mode would have nothing to draw.
    frame_index = _reach_judged_frame(review, detection, position=0.9)

    assert frame_index == 90
    assert review.can_mark
    assert review.render.overlay_complete


def test_the_cumulative_pass_does_not_repaint_over_problem_mode():
    review, detection, _storage, events = _review()
    _reach_judged_frame(review, detection)
    renders = events.renders

    review.history.append(np.ones((_H, _W), dtype=bool))
    review.frame_processed()

    assert events.renders == renders, "its progress is not on screen in this mode"
