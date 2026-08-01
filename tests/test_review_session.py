"""Tests for the reviewing session -- the behaviour the UI merely displays.

Nothing here touches Qt, threads, model weights or the filesystem. The session
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
from rat_tracer.review_session import ReviewSession, SessionListener

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
    """Counts what the session told the UI."""

    def __init__(self):
        self.renders = 0
        self.states = 0
        self.stored: list[int] = []
        self.failed: list[int] = []

    def listener(self) -> SessionListener:
        return SessionListener(
            schedule_render=self._render,
            state_changed=self._state,
            mark_stored=self.stored.append,
            mark_failed=self.failed.append,
        )

    def _render(self) -> None:
        self.renders += 1

    def _state(self) -> None:
        self.states += 1


def _session(
    total_frames: int = 100,
) -> tuple[ReviewSession, _FakeDetection, _FakeStorage, _Events]:
    events = _Events()
    detection, storage = _FakeDetection(), _FakeStorage()
    session = ReviewSession(listener=events.listener(), storage=storage, detection=detection)
    session.open_video(_FakeCapture(total_frames), _VIDEO, _KEY)
    session.set_playing(False)
    return session, detection, storage, events


def _reach_judged_frame(session: ReviewSession, detection: _FakeDetection, position=0.5) -> int:
    """Do what a researcher does: enter the mode, stop on a frame, get an answer."""
    session.seek(position)
    session.set_problem_mode(True)
    session.render_frame()  # draws the frame, which is when the request can go out
    frame_index, _image = detection.requests[-1]
    session.detection_ready(frame_index, detection.detection)
    session.render_frame()  # draws the answer
    return frame_index


# --- mode selection ---------------------------------------------------------


def test_entering_problem_mode_pauses():
    """A frame can only be judged if the researcher has stopped on it."""
    session, _detection, _storage, _events = _session()
    session.set_playing(True)

    session.set_problem_mode(True)

    assert not session.playing
    assert session.problem_mode


def test_resuming_playback_leaves_problem_mode():
    """The two answer unrelated questions, and playback draws no detections."""
    session, detection, _storage, _events = _session()
    _reach_judged_frame(session, detection)

    session.set_playing(True)

    assert not session.problem_mode
    assert not session.can_mark


def test_no_detection_is_requested_before_the_mode_is_entered():
    """A researcher who never reports a problem never pays for a second model."""
    session, detection, _storage, _events = _session()
    session.seek(0.5)
    session.render_frame()

    assert detection.requests == []


# --- detection --------------------------------------------------------------


def test_the_displayed_frame_is_asked_about_once():
    session, detection, _storage, _events = _session()
    _reach_judged_frame(session, detection)
    asked = len(detection.requests)

    session.render_frame()
    session.request_detection()

    assert len(detection.requests) == asked, "the same frame must not be asked about twice"


def test_the_image_sent_for_detection_is_a_copy_of_the_raw_frame():
    """Detection runs later, while rendering keeps mutating its own frame."""
    session, detection, _storage, _events = _session()
    _reach_judged_frame(session, detection)

    _frame_index, image = detection.requests[-1]

    assert session.render.raw_frame is not None
    assert np.array_equal(image, session.render.raw_frame)
    assert image is not session.render.raw_frame


def test_a_failed_detection_can_be_asked_about_again():
    """One failure must not disable the frame for the rest of the session."""
    session, detection, _storage, _events = _session()
    session.seek(0.5)
    session.set_problem_mode(True)
    session.render_frame()
    frame_index, _image = detection.requests[-1]

    session.detection_failed(frame_index)
    session.request_detection()

    assert len(detection.requests) == 2
    assert not session.can_mark


def test_an_answer_for_a_frame_already_left_does_not_repaint():
    session, detection, _storage, events = _session()
    _reach_judged_frame(session, detection, position=0.5)
    session.seek(0.9)
    renders = events.renders

    session.detection_ready(12, Detection())

    assert events.renders == renders


# --- what may be marked -----------------------------------------------------


def test_a_judged_frame_may_be_marked():
    session, detection, _storage, _events = _session()
    _reach_judged_frame(session, detection)

    assert session.can_mark
    assert not session.frame_marked


def test_a_frame_the_detector_found_nothing_in_may_be_marked():
    """A missed detection is the most important defect to report."""
    session, detection, _storage, _events = _session()
    detection.detection = Detection()
    _reach_judged_frame(session, detection)

    assert session.can_mark


def test_a_frame_may_not_be_marked_before_its_answer_arrives():
    session, detection, _storage, _events = _session()
    session.seek(0.5)
    session.set_problem_mode(True)
    session.render_frame()

    assert not session.can_mark, "the answer has not arrived"
    frame_index, _image = detection.requests[-1]
    session.detection_ready(frame_index, detection.detection)
    assert not session.can_mark, "the answer has arrived but is not drawn yet"
    session.render_frame()
    assert session.can_mark


def test_nothing_may_be_marked_without_storage():
    session = ReviewSession(detection=_FakeDetection())
    session.open_video(_FakeCapture(), _VIDEO, _KEY)

    assert not session.can_mark
    assert not session.frame_marked


# --- marking ----------------------------------------------------------------


def test_marking_describes_the_frame_on_screen():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)

    session.mark()

    assert len(storage.stored) == 1
    request = storage.stored[0]
    assert request.frame_index == frame_index
    assert request.video_key == _KEY
    assert request.video_stem == "2026-07-30_run3"
    assert request.model_id == "test-model:v1"
    assert request.detection == Detection(_BOX, [0.9])
    assert request.timestamp_ms == int(frame_index / 25.0 * 1000)
    assert session.render.raw_frame is not None
    assert np.array_equal(request.image, session.render.raw_frame)
    assert request.image is not session.render.raw_frame, "storage runs later"


def test_marking_does_not_move_the_position():
    session, detection, _storage, _events = _session()
    _reach_judged_frame(session, detection)
    position, frame_index = session.position, session.frame_index

    session.mark()

    assert session.position == position
    assert session.frame_index == frame_index


def test_a_frame_being_stored_may_not_be_marked_again():
    """Storage is asynchronous; a second click would otherwise queue a second
    write before the first has landed."""
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)

    session.mark()
    assert not session.can_mark
    session.mark()
    assert len(storage.stored) == 1

    storage.marked.add((_KEY, frame_index))
    session.mark_stored(frame_index)
    assert session.frame_marked


def test_a_stored_frame_is_not_marked_twice():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)
    storage.marked.add((_KEY, frame_index))

    session.mark()

    assert storage.stored == []


def test_a_refused_mark_still_refreshes_the_control():
    """A click flips the control's own tick, so even a refused mark has to send
    it back to reporting what is on disk."""
    session, _detection, _storage, events = _session()
    states = events.states

    session.mark()  # not in problem mode: nothing to mark

    assert events.states > states


def test_a_failed_write_frees_the_frame_to_be_marked_again():
    session, detection, _storage, events = _session()
    frame_index = _reach_judged_frame(session, detection)
    session.mark()

    session.mark_failed(frame_index)

    assert events.failed == [frame_index]
    assert not session.frame_marked
    assert session.can_mark


# --- withdrawing ------------------------------------------------------------


def test_toggling_a_stored_frame_withdraws_it():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)
    storage.marked.add((_KEY, frame_index))

    session.toggle_mark()

    assert storage.removed == [(_KEY, frame_index, "2026-07-30_run3")]
    assert storage.stored == []


def test_toggling_an_unstored_frame_marks_it():
    session, detection, storage, _events = _session()
    _reach_judged_frame(session, detection)

    session.toggle_mark()

    assert len(storage.stored) == 1
    assert storage.removed == []


def test_a_removal_in_flight_blocks_a_second_one():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)
    storage.marked.add((_KEY, frame_index))

    session.unmark()
    session.unmark()

    assert len(storage.removed) == 1, "one removal must not append two rows"
    session.mark_removed(frame_index)
    storage.marked.discard((_KEY, frame_index))
    assert session.can_mark


def test_undo_removes_the_most_recent_mark():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)
    session.mark()
    session.mark_stored(frame_index)

    session.undo()

    assert storage.removed == [(_KEY, frame_index, "2026-07-30_run3")]


def test_undo_after_withdrawing_by_hand_removes_nothing_more():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)
    session.mark()
    session.mark_stored(frame_index)
    storage.marked.add((_KEY, frame_index))
    session.unmark()
    session.mark_removed(frame_index)
    storage.marked.discard((_KEY, frame_index))

    session.undo()

    assert len(storage.removed) == 1, "Undo had nothing left to remove"


def test_undo_without_a_mark_does_nothing():
    session, _detection, storage, _events = _session()

    session.undo()

    assert storage.removed == []


# --- navigation and readouts ------------------------------------------------


def test_stepping_moves_one_frame_and_pauses():
    session, _detection, _storage, _events = _session(total_frames=100)
    session.set_playing(True)
    session.seek(0.5)
    session.render_frame()
    start = session.frame_index

    session.step(1)

    assert session.frame_index == start + 1
    assert not session.playing
    session.step(-1)
    assert session.frame_index == start


def test_stepping_stops_at_the_ends():
    session, _detection, _storage, _events = _session(total_frames=10)
    session.seek(0.0)

    session.step(-1)

    assert session.frame_index == 0


def test_the_readouts_name_the_displayed_frame():
    session, _detection, _storage, _events = _session(total_frames=10_000)
    session.seek(0.5)

    assert session.frame_index == 5000
    assert session.time_text == "00:03:20"


def test_the_readouts_are_available_before_a_video_is_opened():
    session = ReviewSession()

    assert session.time_text == "00:00:00"
    assert session.frame_index == 0
    assert not session.can_mark
    assert not session.video_open


def test_closing_a_video_forgets_its_marks_and_answers():
    session, detection, storage, _events = _session()
    frame_index = _reach_judged_frame(session, detection)
    session.mark()

    session.close_video()

    assert not session.video_open
    assert not session.can_mark
    assert not session.frame_marked
    session.undo()
    assert storage.removed == [], "a closed video's Undo has nothing to act on"
    assert session.problem.detection_for(frame_index) is None


# --- what each mode shows ---------------------------------------------------


def test_problem_mode_hides_the_cumulative_track():
    """A red region is the union of every detection so far, so a single
    frame's detection cannot be judged from it at all."""
    session, detection, _storage, _events = _session()
    for _ in range(100):
        session.history.append(np.ones((_H, _W), dtype=bool))  # everything visited
    detection.detection = Detection()  # nothing found here, so nothing is drawn
    session.seek(0.5)
    tracked = session.render_frame()
    assert tracked.image is not None
    assert session.render.raw_frame is not None
    assert not np.array_equal(tracked.image, session.render.raw_frame), (
        "sanity check: the cumulative track is painted over the frame"
    )

    _reach_judged_frame(session, detection)

    assert session.problem_mode
    session.render.force_repaint()
    shown = session.render.render_now().image
    assert shown is not None
    assert session.render.raw_frame is not None
    assert np.array_equal(shown, session.render.raw_frame), (
        "with the track hidden and nothing detected, the frame is shown clean"
    )


def test_leaving_problem_mode_brings_the_track_back_with_nothing_lost():
    session, detection, _storage, _events = _session()
    for _ in range(100):
        session.history.append(np.ones((_H, _W), dtype=bool))
    _reach_judged_frame(session, detection)
    before = len(session.history)

    session.set_problem_mode(False)
    session.render_frame()

    assert not session.problem_mode
    assert session.render.overlay_complete, "the cumulative track is drawn again"
    assert len(session.history) == before, "the mode is a display state, not a recording state"


def test_a_frame_the_pass_has_not_reached_still_gets_its_detection():
    """The hardest requirement in the feature: a failure is usually noticed
    seconds after the frame that caused it, and the researcher may also seek
    ahead of the pass."""
    session, detection, _storage, _events = _session()
    # Nothing processed at all -- the coverage mode would have nothing to draw.
    frame_index = _reach_judged_frame(session, detection, position=0.9)

    assert frame_index == 90
    assert session.can_mark
    assert session.render.overlay_complete


def test_the_cumulative_pass_does_not_repaint_over_problem_mode():
    session, detection, _storage, events = _session()
    _reach_judged_frame(session, detection)
    renders = events.renders

    session.history.append(np.ones((_H, _W), dtype=bool))
    session.frame_processed()

    assert events.renders == renders, "its progress is not on screen in this mode"
