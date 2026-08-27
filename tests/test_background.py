"""Tests for how the review's slow work is actually run.

Unlike the rest of the UI tests these use *real* threads: what is under test is
ordering, completion delivery and shutdown, and faking the pool to run inline
would remove the thing being checked. Each test is bounded by an explicit
timeout so a hang fails rather than blocks.

The completions come back through a queued signal, so a test has to let the
event loop deliver them -- which is exactly what the application does.
"""

import os
import threading
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QDeadlineTimer, QEventLoop
from PySide6.QtGui import QGuiApplication
from rat_tracer import review_modes
from rat_tracer import video_review as review_module
from rat_tracer.background import InlineExecutor
from rat_tracer.mask_render_core import FrameCapture
from rat_tracer.ui import CoveragePass, QtBackgroundExecutor
from rat_tracer.video_review import VideoReview

_TIMEOUT_S = 5.0
_H, _W = 8, 12


@pytest.fixture(scope="session")
def qapp():
    return QGuiApplication.instance() or QGuiApplication([])


def _appender(sink: list[int], value: int) -> Callable[[], None]:
    """A job that records *value* -- bound now, not when the job runs."""

    def job() -> None:
        sink.append(value)

    return job


def _settle(predicate, message: str) -> None:
    """Pump the event loop until *predicate* holds, or fail."""
    deadline = QDeadlineTimer(int(_TIMEOUT_S * 1000))
    app = QGuiApplication.instance()
    assert app is not None
    while not predicate():
        if deadline.hasExpired():
            raise AssertionError(message)
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)


# --- the inline executor ----------------------------------------------------


def test_the_inline_executor_has_already_finished_when_it_returns():
    done: list[int] = []

    InlineExecutor().submit(lambda: 7, on_done=done.append)

    assert done == [7], "a review driven by this needs no pumping at all"


def test_the_inline_executor_reports_a_failure_instead_of_raising():
    failures: list[BaseException] = []

    def explode() -> int:
        raise RuntimeError("no model on this machine")

    InlineExecutor().submit(explode, on_error=failures.append)

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)


def test_a_cancelled_inline_job_is_already_over():
    job = InlineExecutor().submit(lambda: None)

    assert not job.cancel(), "nothing to abandon once the work has run"


# --- the Qt executor --------------------------------------------------------


def test_a_result_comes_back_on_the_thread_that_submitted_it(qapp):
    executor = QtBackgroundExecutor()
    caller = threading.get_ident()
    ran_on: list[int] = []
    reported_on: list[int] = []

    executor.submit(
        lambda: ran_on.append(threading.get_ident()),
        on_done=lambda _none: reported_on.append(threading.get_ident()),
    )

    _settle(lambda: bool(reported_on), "the completion never arrived")
    assert ran_on[0] != caller, "the work itself must be off the caller's thread"
    assert reported_on[0] == caller, "its result must come back on the caller's"
    executor.stop()


def test_jobs_run_one_at_a_time_in_submission_order(qapp):
    """A mark and its retraction must not interleave, and a stale detection
    must not overwrite a fresh one."""
    executor = QtBackgroundExecutor()
    order: list[int] = []

    for number in range(10):
        executor.submit(_appender(order, number))

    executor.stop()
    assert order == list(range(10))


def test_a_failing_job_is_reported_and_the_worker_survives(qapp):
    executor = QtBackgroundExecutor()
    failures: list[BaseException] = []
    done: list[str] = []

    def explode() -> None:
        raise OSError("disk full")

    executor.submit(explode, on_error=failures.append)
    executor.submit(lambda: "after", on_done=done.append)

    _settle(lambda: bool(failures) and bool(done), "the worker died with its job")
    assert isinstance(failures[0], OSError)
    assert done == ["after"]
    executor.stop()


def test_a_queued_job_can_be_abandoned_before_it_runs(qapp):
    """What replaces the detector's queue-draining: an outstanding request for
    a frame the researcher has seeked past is cancelled, not queued in front."""
    executor = QtBackgroundExecutor()
    gate = threading.Event()
    ran: list[str] = []

    executor.submit(lambda: gate.wait(_TIMEOUT_S))
    stale = executor.submit(lambda: ran.append("stale"))

    assert stale.cancel(), "a job that has not started must be abandonable"
    gate.set()
    executor.stop()
    assert ran == [], "the cancelled job must never have run"


def test_stopping_finishes_the_writes_that_are_still_queued(qapp):
    """An abandoned write loses the observation, which is the only copy of it."""
    executor = QtBackgroundExecutor()
    gate = threading.Event()
    written: list[int] = []

    executor.submit(lambda: gate.wait(_TIMEOUT_S))
    for number in range(5):
        executor.submit(_appender(written, number))

    gate.set()
    executor.stop()

    assert written == list(range(5))


def test_stopping_twice_is_harmless(qapp):
    executor = QtBackgroundExecutor()
    executor.submit(lambda: None)

    executor.stop()
    executor.stop()


def test_an_executor_that_was_never_used_starts_no_thread(qapp):
    """A researcher who only ever watches a video pays for no worker at all."""
    executor = QtBackgroundExecutor()

    assert executor._pool is None
    executor.stop()


# --- the cumulative pass ----------------------------------------------------


class _FakeCapture(FrameCapture):
    def frame_count(self) -> int:
        return 1000

    def fps(self) -> float:
        return 25.0

    def read(self, frame_idx: int) -> np.ndarray | None:
        return np.zeros((_H, _W, 3), dtype=np.uint8)


def _fake_pass_collaborators(monkeypatch, frames: int, on_save=None, gate=None) -> None:
    def produce(input_video, model, start_frame=0):
        for number in range(start_frame, frames):
            if gate is not None and number == 1:
                assert gate.wait(_TIMEOUT_S), "the gate was never released"
            yield None, np.zeros((_H, _W), dtype=bool)

    monkeypatch.setattr(review_module, "video_key", lambda path: "cafe1234")
    monkeypatch.setattr(review_module, "YOLO", lambda *a, **k: None)
    monkeypatch.setattr(review_module, "model_path", lambda: Path("fake-model.pt"))
    # The resume cache belongs to the coverage track, not to the review.
    monkeypatch.setattr(review_modes, "load_progress", lambda key: None)
    monkeypatch.setattr(
        review_modes,
        "save_progress",
        lambda history, key: on_save(key) if on_save is not None else None,
    )
    monkeypatch.setattr(review_module, "presence_frames", produce)


def test_a_real_pass_on_a_real_thread_reaches_the_screen(qapp, monkeypatch, tmp_path):
    """The one test that fakes no threading at all.

    Everywhere else the pass runs inline and the timer fires inline, which
    cannot catch a notification that never crosses back to the GUI thread -- a
    ``QTimer`` started on the pass's thread would simply never fire.

    The pass is held after its first frame until the render that opening the
    video scheduled has already run. From then on nothing on the GUI thread
    asks for anything, so the position advancing to the last frame can only be
    the pass's own notifications crossing over and turning into renders.

    Waiting for a first frame instead would prove nothing: binding the sink
    renders one from the GUI thread anyway. Neither would letting the pass run
    free -- one late render picks up the whole finished pass at once.
    """
    from PySide6.QtMultimedia import QVideoSink
    from rat_tracer.bad_frames import STORAGE_ENV_VAR
    from rat_tracer.ui import VideoMasker

    monkeypatch.setenv(STORAGE_ENV_VAR, str(tmp_path / "bad_frames"))
    gate = threading.Event()
    _fake_pass_collaborators(monkeypatch, frames=20, gate=gate)

    class _Capture:
        def get(self, prop):
            import cv2

            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 20.0
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            return 0.0

        def set(self, prop, value):
            return True

        def read(self):
            return True, np.zeros((_H, _W, 3), dtype=np.uint8)

    monkeypatch.setattr("rat_tracer.ui.VideoCapture", lambda path: _Capture())

    masker = VideoMasker()
    sink = QVideoSink()
    masker.video_output = sink  # type: ignore[assignment]
    try:
        masker.video = "twenty.mp4"  # type: ignore[assignment]
        _settle(
            lambda: sink.videoFrame().isValid(),
            "opening the video never rendered anything at all",
        )
        assert masker.frame_index != 19, "sanity check: the pass is still held at frame 1"

        gate.set()

        _settle(
            lambda: masker.frame_index == 19,
            "the pass ran on its own thread but its progress never reached the UI",
        )
    finally:
        gate.set()
        masker.reset()


def test_the_pass_stops_when_it_is_asked_to_and_is_joined(qapp, monkeypatch):
    """``reset()`` runs on ``aboutToQuit``, so stopping the pass has to both
    interrupt it and wait for it -- a thread still running at interpreter
    shutdown is what used to abort the process."""
    started = threading.Event()
    saved: list[str] = []

    def frames(input_video, model, start_frame=0):
        started.set()
        while True:  # never ends on its own; only the interruption stops it
            yield None, np.zeros((_H, _W), dtype=bool)

    monkeypatch.setattr(review_module, "video_key", lambda path: "cafe1234")
    monkeypatch.setattr(review_module, "YOLO", lambda *a, **k: None)
    monkeypatch.setattr(review_module, "model_path", lambda: Path("fake-model.pt"))
    monkeypatch.setattr(review_modes, "load_progress", lambda key: None)
    monkeypatch.setattr(review_modes, "save_progress", lambda history, key: saved.append(key))
    monkeypatch.setattr(review_module, "presence_frames", frames)

    review = VideoReview()
    review.open_video(_FakeCapture(), Path("endless.mp4"))
    coverage_pass = CoveragePass(review)
    coverage_pass.start()
    assert started.wait(_TIMEOUT_S), "the pass never got going"

    coverage_pass.stop()

    assert not coverage_pass._thread.is_alive(), "stop() must join, not just ask"
    assert saved == ["cafe1234"], "an interrupted pass saves where it got to"
