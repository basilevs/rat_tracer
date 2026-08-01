"""Tests for the background workers' job queues and shutdown.

Unlike the rest of the UI tests these use *real* threads: how a worker stops is
precisely what is under test, and faking QThread.start to run inline would
remove the thing being checked. Each test is bounded by an explicit timeout so
a hang fails rather than blocks.
"""

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from rat_tracer import ui as ui_module
from rat_tracer.bad_frames import Detection
from rat_tracer.ui import QtDetectionService, QtMarkStorage, _Worker

_TIMEOUT_S = 5.0
_TIMEOUT_MS = 5000


@pytest.fixture(scope="session")
def qapp():
    return QGuiApplication.instance() or QGuiApplication([])


class _KeepsQueuedJobs(_Worker):
    """Stops like the storage worker: queued work is finished first."""

    def __init__(self):
        super().__init__("test-keeps")
        self.done: list[int] = []


class _DropsQueuedJobs(_Worker):
    """Stops like the detector: queued work is abandoned."""

    discards_queued_jobs = True

    def __init__(self):
        super().__init__("test-drops")
        self.done: list[int] = []


def _record(worker, number: int, gate: threading.Event | None = None, started=None):
    def job():
        if started is not None:
            started.set()
        if gate is not None:
            assert gate.wait(_TIMEOUT_S), "gate was never released"
        worker.done.append(number)

    return job


def test_a_worker_stops_when_its_queue_is_shut_down(qapp):
    worker = _KeepsQueuedJobs()
    worker.start()
    ran = threading.Event()
    worker.submit(ran.set)
    assert ran.wait(_TIMEOUT_S)

    worker.stop()

    assert worker.wait(_TIMEOUT_MS), "the thread did not exit"
    assert worker.isFinished()


def test_queued_writes_are_finished_before_the_worker_stops(qapp):
    """A mark the researcher has already made must not be lost because they
    closed the video a moment later."""
    worker = _KeepsQueuedJobs()
    gate, started = threading.Event(), threading.Event()
    worker.submit(_record(worker, 0, gate, started))  # holds the thread in job 0
    worker.start()
    assert started.wait(_TIMEOUT_S)
    for number in range(1, 5):
        worker.submit(_record(worker, number))

    worker.stop()  # ... while four jobs are still queued
    gate.set()

    assert worker.wait(_TIMEOUT_MS)
    assert worker.done == [0, 1, 2, 3, 4]


def test_queued_detections_are_abandoned_when_the_worker_stops(qapp):
    """A detection is only interesting for a frame being looked at, so work
    still queued when the video closes is already worthless."""
    worker = _DropsQueuedJobs()
    gate, started = threading.Event(), threading.Event()
    worker.submit(_record(worker, 0, gate, started))
    worker.start()
    assert started.wait(_TIMEOUT_S), "job 0 must be running before the queue is shut down"
    for number in range(1, 5):
        worker.submit(_record(worker, number))

    worker.stop()
    gate.set()

    assert worker.wait(_TIMEOUT_MS)
    assert worker.done == [0], "only the job already running should have finished"


def test_a_job_submitted_after_shutdown_is_dropped_without_raising(qapp):
    """Teardown races with the UI thread; a late job is dropped, not an error."""
    worker = _KeepsQueuedJobs()
    worker.start()
    worker.stop()
    assert worker.wait(_TIMEOUT_MS)

    worker.submit(_record(worker, 99))  # must not raise

    assert worker.done == []


def test_stopping_twice_is_harmless(qapp):
    """reset() runs on both closing a video and quitting the application."""
    worker = _KeepsQueuedJobs()
    worker.start()
    worker.stop()
    assert worker.wait(_TIMEOUT_MS)

    worker.stop()  # must not raise

    assert worker.isFinished()


class _InstantDetector:
    """Answers without a model, so the thread under test is the only slow part."""

    @property
    def model_id(self) -> str:
        return "test-model:v1"

    def prewarm(self) -> None:
        pass

    def detect(self, image):
        return Detection()


def _record_workers(monkeypatch, name: str) -> list:
    """Capture the worker instances a service starts, so the test can join them."""
    started: list = []
    real = getattr(ui_module, name)

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(ui_module, name, Recording)
    return started


def test_the_detection_service_leaves_no_thread_running(qapp, monkeypatch):
    """Closing the video -- or the window -- must join the detector's thread.

    A QThread still running when the interpreter tears PySide down is destroyed
    while running, which Qt reports with qFatal and aborts the process.
    """
    monkeypatch.setattr(ui_module, "YoloFrameDetector", lambda *a, **k: _InstantDetector())
    workers = _record_workers(monkeypatch, "FrameDetectionWorker")
    service = QtDetectionService(on_ready=lambda *a: None, on_failed=lambda *a: None)
    service.request(0, np.zeros((4, 4, 3), dtype=np.uint8))
    assert workers, "requesting a detection starts the worker"

    service.stop()

    assert workers[0].wait(_TIMEOUT_MS), "the detector's thread is still running"
    assert workers[0].isFinished()


def test_the_mark_storage_leaves_no_thread_running(qapp, monkeypatch):
    class _FakeStore:
        def is_marked(self, video_key: str, frame_index: int) -> bool:
            return False

        def retract(self, video_key: str, frame_index: int, video_stem: str) -> None:
            pass

    monkeypatch.setattr(ui_module, "BadFrameStore", lambda *a, **k: _FakeStore())
    workers = _record_workers(monkeypatch, "MarkStorageWorker")
    storage = QtMarkStorage(
        on_stored=lambda *a: None, on_failed=lambda *a: None, on_removed=lambda *a: None
    )
    storage.remove("cafe1234", 7, "experiment")
    assert workers, "removing a mark starts the worker"

    storage.stop()

    assert workers[0].wait(_TIMEOUT_MS), "the storage thread is still running"
    assert workers[0].isFinished()


def test_a_failing_job_does_not_stop_the_worker(qapp):
    """The researcher keeps navigating, and the next job still runs."""
    worker = _KeepsQueuedJobs()
    worker.start()

    def explode():
        raise RuntimeError("job failed")

    worker.submit(explode)
    ran = threading.Event()
    worker.submit(ran.set)

    assert ran.wait(_TIMEOUT_S), "a later job must still run"
    worker.stop()
    assert worker.wait(_TIMEOUT_MS)
