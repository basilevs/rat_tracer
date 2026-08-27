"""Tests for a video review -- the behaviour the UI merely displays.

Nothing here touches Qt, threads, model weights or the filesystem. The slow
work goes to a fake executor that holds each job until the test runs it, so the
asynchrony is real -- a request is made, a completion arrives later -- but its
timing is the test's to choose. The cumulative pass is the same idea one level
up: ``process_video`` is called directly, with its collaborators faked, rather
than given a thread.

That is the point of the class. These decisions used to be split between
``VideoMasker`` and ``MaskRenderCore`` and could only be exercised through a
Qt harness with faked threads.
"""

from collections.abc import Callable
from pathlib import Path
from typing import override

import numpy as np
import pytest
from numpy import ndarray
from rat_tracer import review_modes, video_review
from rat_tracer.bad_frames import BadFrameStore, Detection, MarkRequest
from rat_tracer.mask_render_core import FrameCapture
from rat_tracer.video_review import ReviewListener, VideoReview

from queued_executor import QueuedExecutor

_H, _W = 8, 12
_VIDEO = Path("2026-07-30_run3.mp4")
_KEY = "cafe1234"
_STEM = "2026-07-30_run3"
_BOX = [[0.5, 0.5, 0.2, 0.2]]


def _frame(frame_index: int) -> ndarray:
    """What ``_FakeCapture`` decodes for *frame_index*, unannotated."""
    return np.full((_H, _W, 3), frame_index % 256, dtype=np.uint8)


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
        return _frame(frame_idx)


class _FakeDetector:
    """A detector with no model behind it, and no network."""

    def __init__(self, detection: Detection | None = None):
        self.detection = detection if detection is not None else Detection(_BOX, [0.9])
        self.fail = False
        self.images: list[ndarray] = []
        self.prewarmed = 0

    @property
    def model_id(self) -> str:
        return "test-model:v1"

    def prewarm(self) -> None:
        self.prewarmed += 1

    def detect(self, image: ndarray) -> Detection:
        self.images.append(image)
        if self.fail:
            raise RuntimeError("no model on this machine")
        return self.detection


class _FakeStore(BadFrameStore):
    """Keeps the index in memory; writes nothing and reads nothing.

    A real :class:`BadFrameStore` with a root that is never touched, because
    every method that would touch it is overridden -- so the review is driven
    against the type it actually collaborates with.
    """

    def __init__(self):
        super().__init__(Path("nowhere"))
        self.stored: list[MarkRequest] = []
        self.removed: list[tuple[str, int, str]] = []
        self.fail = False

    @override
    def is_marked(self, video_key: str, frame_index: int) -> bool:
        return (video_key, frame_index) in self._marked

    @override
    def mark(self, request: MarkRequest) -> str:
        if self.fail:
            raise OSError("disk full")
        self.stored.append(request)
        self._marked.add((request.video_key, request.frame_index))
        return "stored"

    @override
    def retract(self, video_key: str, frame_index: int, video_stem: str) -> None:
        self.removed.append((video_key, frame_index, video_stem))
        self._marked.discard((video_key, frame_index))


class _Events:
    """Counts what the review told the UI."""

    def __init__(self):
        self.changes = 0
        self.stored: list[int] = []
        self.failed: list[int] = []

    def listener(self) -> ReviewListener:
        return ReviewListener(
            changed=self._changed,
            mark_stored=self.stored.append,
            mark_failed=self.failed.append,
        )

    def _changed(self) -> None:
        self.changes += 1


class _Review:
    """A review and the fakes behind it, so a test can name any of them."""

    def __init__(self, total_frames: int = 100):
        self.events = _Events()
        self.detector = _FakeDetector()
        self.store = _FakeStore()
        self.executor = QueuedExecutor()
        self.total_frames = total_frames
        self.review = VideoReview(
            listener=self.events.listener(),
            executor=self.executor,
            detector=self.detector,
            store=self.store,
        )
        self.review.open_video(_FakeCapture(total_frames), _VIDEO)
        # In production the fingerprint is the pass's first act; every test
        # below wants a video that can already be marked.
        self.review._render.identify(_KEY)
        self.review.set_playing(False)

    def reach_judged_frame(self, position: float = 0.5) -> int:
        """Do what a researcher does: enter the mode, stop, get an answer."""
        review = self.review
        review.seek(position)
        review.set_problem_mode(True)
        review.render_frame()  # draws the frame, which is when the request goes out
        self.executor.pump()  # the detector answers
        review.render_frame()  # draws the answer
        return review.frame_index


@pytest.fixture
def fixture() -> _Review:
    return _Review()


@pytest.fixture
def pass_collaborators(monkeypatch) -> Callable[[int], None]:
    """Fake everything the cumulative pass reaches for outside the review."""
    monkeypatch.setattr(video_review, "video_key", lambda path: _KEY)
    monkeypatch.setattr(video_review, "YOLO", lambda *a, **k: None)
    monkeypatch.setattr(video_review, "model_path", lambda: Path("fake-model.pt"))
    # The resume cache belongs to the coverage track, not to the review.
    monkeypatch.setattr(review_modes, "load_progress", lambda key: None)
    monkeypatch.setattr(review_modes, "save_progress", lambda history, key: None)

    def produce(total: int) -> None:
        def frames(input_video, model, start_frame=0):
            for _ in range(start_frame, total):
                yield None, np.ones((_H, _W), dtype=bool)  # everything visited

        monkeypatch.setattr(video_review, "presence_frames", frames)

    return produce


def _run_pass(fixture: _Review, pass_collaborators, frames: int | None = None) -> None:
    """Run the cumulative pass to completion, the way its thread would."""
    pass_collaborators(fixture.total_frames if frames is None else frames)
    fixture.review.process_video(lambda: False)


# --- mode selection ---------------------------------------------------------


def test_entering_problem_mode_pauses(fixture):
    """A frame can only be judged if the researcher has stopped on it."""
    fixture.review.set_playing(True)

    fixture.review.set_problem_mode(True)

    assert not fixture.review.playing
    assert fixture.review.problem_mode


def test_resuming_playback_leaves_problem_mode(fixture):
    """The two answer unrelated questions, and playback draws no detections."""
    fixture.reach_judged_frame()

    fixture.review.set_playing(True)

    assert not fixture.review.problem_mode
    assert not fixture.review.can_mark


def test_no_detection_is_requested_before_the_mode_is_entered(fixture):
    """A researcher who never reports a problem never pays for a second model."""
    fixture.review.seek(0.5)
    fixture.review.render_frame()
    fixture.executor.pump()

    assert fixture.detector.images == []
    assert fixture.detector.prewarmed == 0


def test_the_model_is_loaded_before_a_frame_waits_on_it(fixture):
    """The first inference costs seconds and the queue is serial, so the load
    goes in ahead of the request it would otherwise delay."""
    fixture.review.set_problem_mode(True)
    fixture.review.set_problem_mode(False)
    fixture.review.set_problem_mode(True)
    fixture.executor.pump()

    assert fixture.detector.prewarmed == 1, "paid once, not once per entry"


# --- detection --------------------------------------------------------------


def test_the_displayed_frame_is_asked_about_once(fixture):
    fixture.reach_judged_frame()
    asked = len(fixture.detector.images)

    fixture.review.render_frame()
    fixture.executor.pump()

    assert len(fixture.detector.images) == asked, "the same frame must not be asked about twice"


def test_the_image_sent_for_detection_is_a_copy_of_the_raw_frame(fixture):
    """Detection runs later, while rendering keeps mutating its own frame."""
    frame_index = fixture.reach_judged_frame()

    sent = fixture.detector.images[-1]

    assert np.array_equal(sent, _frame(frame_index)), "the frame as decoded, unannotated"


def test_a_failed_detection_can_be_asked_about_again(fixture):
    """One failure must not disable the frame for the rest of the review."""
    fixture.detector.fail = True
    fixture.reach_judged_frame()
    assert not fixture.review.can_mark

    fixture.detector.fail = False
    fixture.review.render_frame()
    fixture.executor.pump()
    fixture.review.render_frame()

    assert len(fixture.detector.images) == 2
    assert fixture.review.can_mark


def test_seeking_past_a_request_abandons_it(fixture):
    """Seeking outruns inference, and the researcher is looking at the newest
    frame -- so the older request is dropped rather than queued in front."""
    review = fixture.review
    review.set_problem_mode(True)
    review.seek(0.5)
    review.render_frame()
    stale = fixture.executor.jobs[-1]

    review.seek(0.9)
    review.render_frame()

    assert stale.cancelled, "the frame that was seeked past must not be inferred"
    fixture.executor.pump()
    review.render_frame()
    assert review.frame_index == 90
    assert review.can_mark


def test_an_abandoned_frame_is_asked_about_again_on_return(fixture):
    review = fixture.review
    review.set_problem_mode(True)
    review.seek(0.5)
    review.render_frame()
    review.seek(0.9)
    review.render_frame()

    review.seek(0.5)
    review.render_frame()
    fixture.executor.pump()
    review.render_frame()

    assert review.frame_index == 50
    assert review.can_mark, "coming back to an abandoned frame must ask again"


def test_an_answer_for_a_frame_already_left_does_not_repaint(fixture):
    fixture.reach_judged_frame(position=0.5)
    fixture.review.seek(0.9)
    fixture.review.render_frame()
    before = fixture.review.render_frame()

    assert not before.should_emit, "sanity check: nothing is outstanding here"


# --- what may be marked -----------------------------------------------------


def test_a_judged_frame_may_be_marked(fixture):
    fixture.reach_judged_frame()

    assert fixture.review.can_mark
    assert not fixture.review.frame_marked


def test_a_frame_the_detector_found_nothing_in_may_be_marked(fixture):
    """A missed detection is the most important defect to report."""
    fixture.detector.detection = Detection()
    fixture.reach_judged_frame()

    assert fixture.review.can_mark


def test_a_frame_may_not_be_marked_before_its_answer_arrives(fixture):
    review = fixture.review
    review.seek(0.5)
    review.set_problem_mode(True)
    review.render_frame()

    assert not review.can_mark, "the answer has not arrived"
    fixture.executor.pump()
    assert not review.can_mark, "the answer has arrived but is not drawn yet"
    review.render_frame()
    assert review.can_mark


def test_nothing_may_be_marked_without_storage():
    review = VideoReview(detector=_FakeDetector())
    review.open_video(_FakeCapture(), _VIDEO)

    assert not review.can_mark
    assert not review.frame_marked


def test_nothing_may_be_marked_until_the_pass_has_named_the_video(fixture):
    """A mark is stored under the video's fingerprint, and computing it reads
    the whole file -- so the pass does it, and marking waits for it."""
    review = VideoReview(
        listener=fixture.events.listener(),
        executor=fixture.executor,
        detector=fixture.detector,
        store=fixture.store,
    )
    review.open_video(_FakeCapture(), _VIDEO)
    review.set_playing(False)
    review.seek(0.5)
    review.set_problem_mode(True)
    review.render_frame()
    fixture.executor.pump()
    review.render_frame()

    assert not review.can_mark, "no fingerprint yet, so nowhere to file the mark"
    review._render.identify(_KEY)
    assert review.can_mark


# --- marking ----------------------------------------------------------------


def test_marking_describes_the_frame_on_screen(fixture):
    frame_index = fixture.reach_judged_frame()

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert len(fixture.store.stored) == 1
    request = fixture.store.stored[0]
    assert request.frame_index == frame_index
    assert request.video_key == _KEY
    assert request.video_stem == _STEM
    assert request.model_id == "test-model:v1"
    assert request.detection == Detection(_BOX, [0.9])
    assert request.timestamp_ms == int(frame_index / 25.0 * 1000)
    assert np.array_equal(request.image, _frame(frame_index)), "stored raw, not annotated"


def test_marking_does_not_move_the_position(fixture):
    fixture.reach_judged_frame()
    position, frame_index = fixture.review.position, fixture.review.frame_index

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert fixture.review.position == position
    assert fixture.review.frame_index == frame_index


def test_a_frame_being_stored_may_not_be_marked_again(fixture):
    """Storage is asynchronous; a second click would otherwise queue a second
    write before the first has landed."""
    fixture.reach_judged_frame()

    fixture.review.toggle_mark()
    assert not fixture.review.can_mark
    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert len(fixture.store.stored) == 1
    assert fixture.review.frame_marked


def test_a_stored_frame_is_not_marked_twice(fixture):
    frame_index = fixture.reach_judged_frame()
    fixture.store._marked.add((_KEY, frame_index))

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert fixture.store.stored == []


def test_a_refused_click_still_refreshes_the_control(fixture):
    """A click flips the control's own tick, so even a refused one has to send
    it back to reporting what is on disk."""
    changes = fixture.events.changes

    fixture.review.toggle_mark()  # not in problem mode: nothing to mark

    assert fixture.events.changes > changes


def test_a_failed_write_frees_the_frame_to_be_marked_again(fixture):
    frame_index = fixture.reach_judged_frame()
    fixture.store.fail = True

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert fixture.events.failed == [frame_index]
    assert not fixture.review.frame_marked
    assert fixture.review.can_mark


def test_a_stored_frame_is_confirmed_by_index(fixture):
    frame_index = fixture.reach_judged_frame()

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert fixture.events.stored == [frame_index]


# --- withdrawing ------------------------------------------------------------


def test_toggling_a_stored_frame_withdraws_it(fixture):
    frame_index = fixture.reach_judged_frame()
    fixture.store._marked.add((_KEY, frame_index))

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert fixture.store.removed == [(_KEY, frame_index, _STEM)]
    assert fixture.store.stored == []


def test_toggling_an_unstored_frame_marks_it(fixture):
    fixture.reach_judged_frame()

    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert len(fixture.store.stored) == 1
    assert fixture.store.removed == []


def test_a_removal_in_flight_blocks_a_second_one(fixture):
    frame_index = fixture.reach_judged_frame()
    fixture.store._marked.add((_KEY, frame_index))

    fixture.review.toggle_mark()
    assert not fixture.review.can_mark, "the control is disabled while it is in flight"
    fixture.review.toggle_mark()
    fixture.executor.pump()

    assert len(fixture.store.removed) == 1, "one removal must not append two rows"
    assert fixture.review.can_mark


def test_undo_removes_the_most_recent_mark(fixture):
    frame_index = fixture.reach_judged_frame()
    fixture.review.toggle_mark()
    fixture.executor.pump()

    fixture.review.undo()
    fixture.executor.pump()

    assert fixture.store.removed == [(_KEY, frame_index, _STEM)]


def test_undo_after_withdrawing_by_hand_removes_nothing_more(fixture):
    fixture.reach_judged_frame()
    fixture.review.toggle_mark()
    fixture.executor.pump()
    fixture.review.toggle_mark()
    fixture.executor.pump()

    fixture.review.undo()
    fixture.executor.pump()

    assert len(fixture.store.removed) == 1, "Undo had nothing left to remove"


def test_undo_without_a_mark_does_nothing(fixture):
    fixture.review.undo()
    fixture.executor.pump()

    assert fixture.store.removed == []


# --- navigation and readouts ------------------------------------------------


def test_stepping_moves_one_frame_and_pauses(fixture):
    fixture.review.set_playing(True)
    fixture.review.seek(0.5)
    fixture.review.render_frame()
    start = fixture.review.frame_index

    fixture.review.step(1)

    assert fixture.review.frame_index == start + 1
    assert not fixture.review.playing
    fixture.review.step(-1)
    assert fixture.review.frame_index == start


def test_stepping_stops_at_the_ends():
    fixture = _Review(total_frames=10)
    fixture.review.seek(0.0)

    fixture.review.step(-1)

    assert fixture.review.frame_index == 0


def test_the_readouts_name_the_displayed_frame():
    fixture = _Review(total_frames=10_000)
    fixture.review.seek(0.5)

    assert fixture.review.frame_index == 5000
    assert fixture.review.time_text == "00:03:20"


def test_the_readouts_are_available_before_a_video_is_opened():
    review = VideoReview()

    assert review.time_text == "00:00:00"
    assert review.frame_index == 0
    assert not review.can_mark
    assert not review.video_open


def test_closing_a_video_forgets_its_marks_and_answers(fixture):
    fixture.reach_judged_frame()
    fixture.review.toggle_mark()
    fixture.executor.pump()
    asked = len(fixture.detector.images)

    fixture.review.close_video()

    assert not fixture.review.video_open
    assert not fixture.review.can_mark
    assert not fixture.review.frame_marked
    fixture.review.undo()
    fixture.executor.pump()
    assert fixture.store.removed == [], "a closed video's Undo has nothing to act on"

    fixture.review.open_video(_FakeCapture(fixture.total_frames), _VIDEO)
    fixture.review._render.identify(_KEY)
    fixture.review.set_playing(False)
    fixture.reach_judged_frame()
    assert len(fixture.detector.images) > asked, "the answers went with the video"


# --- the cumulative pass ----------------------------------------------------


def test_the_pass_records_coverage_and_shows_it(fixture, pass_collaborators):
    _run_pass(fixture, pass_collaborators)
    fixture.review.seek(0.5)

    shown = fixture.review.render_frame()

    assert shown.image is not None
    assert not np.array_equal(shown.image, _frame(50)), "the cumulative track is painted on"


def test_the_pass_names_the_video_so_frames_can_be_marked(pass_collaborators):
    """The fingerprint is the pass's first act, before any inference."""
    fixture = _Review()
    review = VideoReview(
        listener=fixture.events.listener(),
        executor=fixture.executor,
        detector=fixture.detector,
        store=fixture.store,
    )
    review.open_video(_FakeCapture(), _VIDEO)
    review.set_playing(False)

    _run_pass(fixture, pass_collaborators)
    review.process_video(lambda: False)

    review.seek(0.5)
    review.set_problem_mode(True)
    review.render_frame()
    fixture.executor.pump()
    review.render_frame()
    assert review.can_mark


def test_an_interrupted_pass_stops_where_it_was_asked_to(fixture, pass_collaborators):
    pass_collaborators(fixture.total_frames)
    stop_after = 5
    seen = 0

    def is_interrupted() -> bool:
        nonlocal seen
        seen += 1
        return seen >= stop_after

    fixture.review.process_video(is_interrupted)

    assert seen == stop_after, "the pass must stop being asked once it has stopped"


def test_a_seek_ahead_of_the_pass_renders_bare_then_masked(fixture, pass_collaborators):
    """From repro.log: a seek landing before the pass has reached that frame
    shows it bare, and the track appears when the pass catches up."""
    review = fixture.review
    review.seek(0.55)
    bare = review.render_frame()

    assert bare.image is not None
    assert np.array_equal(bare.image, _frame(55)), "nothing processed there yet -- shown bare"

    _run_pass(fixture, pass_collaborators)
    masked = review.render_frame()

    assert masked.image is not None
    assert not np.array_equal(masked.image, _frame(55)), "the track lands once the pass arrives"


def test_reopening_a_processed_video_while_paused_still_draws(fixture, pass_collaborators):
    """From repro.log: opening a second, already processed video while paused
    used to leave the placeholder frame from closing the first one on screen --
    the flag saying the overlay was complete belonged to the renderer and
    survived the reset. Nothing outside a mode records that any more."""
    _run_pass(fixture, pass_collaborators)
    fixture.review.seek(0.55)
    fixture.review.render_frame()

    fixture.review.close_video()
    fixture.review.open_video(_FakeCapture(fixture.total_frames), _VIDEO)
    fixture.review.set_playing(False)
    _run_pass(fixture, pass_collaborators)
    fixture.review.seek(0.55)
    shown = fixture.review.render_frame()

    assert shown.image is not None, "the second video left the placeholder on screen"
    assert not np.array_equal(shown.image, _frame(55)), "and its own track is drawn"


def test_problem_mode_hides_the_cumulative_track(fixture, pass_collaborators):
    """A red region is the union of every detection so far, so a single
    frame's detection cannot be judged from it at all."""
    _run_pass(fixture, pass_collaborators)
    fixture.detector.detection = Detection()  # nothing found here, so nothing is drawn
    fixture.review.seek(0.5)
    tracked = fixture.review.render_frame()
    assert tracked.image is not None
    assert not np.array_equal(tracked.image, _frame(50)), (
        "sanity check: the cumulative track is painted over the frame"
    )

    frame_index = fixture.reach_judged_frame()
    shown = fixture.review.render_frame()

    assert fixture.review.problem_mode
    assert shown.image is None or np.array_equal(shown.image, _frame(frame_index)), (
        "with the track hidden and nothing detected, the frame is shown clean"
    )


def test_leaving_problem_mode_brings_the_track_back_with_nothing_lost(fixture, pass_collaborators):
    _run_pass(fixture, pass_collaborators)
    frame_index = fixture.reach_judged_frame()

    fixture.review.set_problem_mode(False)
    shown = fixture.review.render_frame()

    assert not fixture.review.problem_mode
    assert shown.image is not None
    assert not np.array_equal(shown.image, _frame(frame_index)), (
        "the cumulative track is drawn again, so nothing was lost by leaving it"
    )


def test_returning_to_the_track_redraws_a_frame_it_had_already_drawn(fixture, pass_collaborators):
    """The mode coming back has drawn this frame before, but the other one has
    painted over it since. Being told it ``left`` is the only thing that tells
    it so -- there is no forced repaint anywhere to fall back on."""
    _run_pass(fixture, pass_collaborators)
    review = fixture.review
    review.seek(0.5)
    tracked = review.render_frame()
    assert tracked.image is not None
    assert not np.array_equal(tracked.image, _frame(50)), "sanity check: the track is up"

    fixture.detector.detection = Detection()  # nothing to draw, so the track's absence shows
    fixture.reach_judged_frame()
    over = fixture.review.render_frame()
    assert over.image is None or np.array_equal(over.image, _frame(50))

    review.set_problem_mode(False)
    back = review.render_frame()

    assert back.image is not None, "coverage never redrew the frame it thought it still had up"
    assert not np.array_equal(back.image, _frame(50)), "the track is back"


def test_leaving_problem_mode_while_paused_disables_marking(fixture):
    """Its detection is no longer what is on screen, so nothing is judged --
    and nothing but the mode itself is keeping track of that."""
    fixture.reach_judged_frame()
    assert fixture.review.can_mark

    fixture.review.set_problem_mode(False)

    assert not fixture.review.can_mark


def test_a_frame_the_pass_has_not_reached_still_gets_its_detection(fixture):
    """The hardest requirement in the feature: a failure is usually noticed
    seconds after the frame that caused it, and the researcher may also seek
    ahead of the pass."""
    # Nothing processed at all -- the coverage mode would have nothing to draw.
    frame_index = fixture.reach_judged_frame(position=0.9)

    assert frame_index == 90
    assert fixture.review.can_mark


def test_the_cumulative_pass_does_not_repaint_over_problem_mode(fixture, pass_collaborators):
    fixture.reach_judged_frame()
    changes = fixture.events.changes

    _run_pass(fixture, pass_collaborators)

    assert fixture.events.changes == changes + 1, (
        "only the fingerprint is worth reporting here; the track is not on screen"
    )
