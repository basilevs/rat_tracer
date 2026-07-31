"""End-to-end tests for problem reporting mode, driven through VideoMasker.

Like ``test_video_masker.py`` these go through the public Qt surface only --
properties, slots and signals -- and fake the heavy collaborators at the module
boundary: the video capture, the cumulative pass and the detection model. The
storage tree is real, in a temp directory, because its behaviour under marking
and undoing is exactly what is being tested.

Threads are faked rather than started: the two background workers are real
objects whose jobs the test runs inline via ``pump``. That keeps ordering
deterministic while still exercising the jobs ``VideoMasker`` actually submits.
"""

import os
import pickle
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtMultimedia import QVideoSink
from rat_tracer import ui as ui_module
from rat_tracer.bad_frames import STORAGE_ENV_VAR, BadFrameStore, Detection
from rat_tracer.ui import VideoMasker

_H, _W = 16, 24
_TOTAL_FRAMES = 100
_VIDEO = "experiment.mp4"
_BOX = [[0.5, 0.5, 0.4, 0.4]]


class _FakeCapture:
    """Stands in for ``cv2.VideoCapture``: solid frames, shade encodes index."""

    def __init__(self, total_frames: int, fps: float = 25.0):
        self.total_frames = total_frames
        self.fps = fps
        self.frame_idx = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.frame_idx / self.fps * 1000.0
        return 0.0

    def set(self, prop, value):
        self.frame_idx = int(value)
        return True

    def read(self):
        return True, np.full((_H, _W, 3), self.frame_idx % 256, dtype=np.uint8)


class _FakeDetector:
    """A detector that answers instantly, with no model and no network."""

    def __init__(self, detection: Detection | None = None, fail: bool = False):
        self.detection = detection if detection is not None else Detection(_BOX, [0.9])
        self.fail = fail
        self.calls: list = []
        self.prewarmed = False

    @property
    def model_id(self) -> str:
        return "test-model:v1"

    def prewarm(self) -> None:
        self.prewarmed = True

    def detect(self, image):
        self.calls.append(image)
        if self.fail:
            raise RuntimeError("no model on this machine")
        return self.detection


@pytest.fixture(scope="session")
def qapp():
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Fake the video, the cumulative pass, the model and thread scheduling."""
    monkeypatch.setenv(STORAGE_ENV_VAR, str(tmp_path / "bad_frames"))

    def fake_presence_frames(input_video, model, start_frame=0):
        for _ in range(start_frame, _TOTAL_FRAMES):
            yield None, np.zeros((_H, _W), dtype=bool)

    cache: dict[str, object] = {}
    monkeypatch.setattr(ui_module, "VideoCapture", lambda path: _FakeCapture(_TOTAL_FRAMES))
    monkeypatch.setattr(ui_module, "video_key", lambda path: "cafe1234")
    monkeypatch.setattr(ui_module, "YOLO", lambda *a, **k: None)
    monkeypatch.setattr(ui_module, "model_path", lambda: Path("fake-model.pt"))
    monkeypatch.setattr(ui_module, "presence_frames", fake_presence_frames)
    monkeypatch.setattr(ui_module, "load_progress", cache.get)
    monkeypatch.setattr(
        ui_module,
        "save_progress",
        lambda history, key: cache.__setitem__(key, pickle.loads(pickle.dumps(history))),
    )
    monkeypatch.setattr(QThread, "start", lambda self, priority=None: self.run())
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda _msec, cb: cb()))
    # The queue-driven workers would block forever if run inline; the test
    # pumps their jobs instead, so their threads never start.
    monkeypatch.setattr(ui_module._Worker, "start", lambda self, priority=None: None)
    monkeypatch.setattr(ui_module._Worker, "wait", lambda self, *a, **k: True)

    detector = _FakeDetector()
    monkeypatch.setattr(ui_module, "YoloFrameDetector", lambda *a, **k: detector)

    class Harness:
        root = tmp_path / "bad_frames"
        detector_stub = detector

        def open(self) -> VideoMasker:
            masker = VideoMasker()
            masker.video_output = QVideoSink()  # type: ignore[assignment]
            masker.video = _VIDEO  # type: ignore[assignment]
            masker.playing = False  # type: ignore[assignment]
            return masker

        def pump(self, masker: VideoMasker) -> None:
            """Run every queued background job, then let renders settle."""
            for _ in range(10):
                ran = False
                for worker in (masker._detector_worker, masker._storage_worker):
                    if worker is None:
                        continue
                    while True:
                        try:
                            job = worker._queue.get_nowait()
                        except Exception:
                            break
                        if job is None:
                            continue
                        # Through the worker's own error handling, so a failing
                        # job behaves here exactly as it would on the thread.
                        worker.run_job(job)
                        ran = True
                if not ran:
                    return

        def store(self) -> BadFrameStore:
            return BadFrameStore(self.root)

    return Harness()


def _enter_problem_mode(masker: VideoMasker, harness, position: float = 0.5) -> None:
    masker.position = position  # type: ignore[assignment]
    masker.problem_mode = True  # type: ignore[assignment]
    harness.pump(masker)


def test_entering_problem_mode_pauses_and_asks_for_this_frame(qapp, harness):
    masker = harness.open()
    masker.playing = True

    _enter_problem_mode(masker, harness, 0.5)

    assert not masker.playing
    assert masker.problem_mode
    assert harness.detector_stub.calls, "the displayed frame's detection was requested"
    assert masker.can_mark
    masker.reset()


def test_a_frame_the_detector_found_nothing_in_is_markable(qapp, harness, monkeypatch):
    """A missed detection is the most important defect to report."""
    harness.detector_stub.detection = Detection()
    masker = harness.open()

    _enter_problem_mode(masker, harness)

    assert masker.can_mark
    masker.reset()


def test_marking_stores_the_raw_frame_and_leaves_the_position_alone(qapp, harness):
    masker = harness.open()
    _enter_problem_mode(masker, harness, 0.5)
    position_before = masker.position
    frame_index = masker.frame_index

    masker.markBadFrame()
    harness.pump(masker)

    assert masker.position == position_before, "marking reads the frame, it does not navigate"
    stored = harness.root / "images" / f"experiment_{frame_index:06d}.png"
    assert stored.is_file()
    image = cv2.imread(str(stored))
    assert image is not None
    # A box was drawn on screen; the stored frame must carry no annotation, so
    # every pixel is still the flat shade the fake capture produced.
    assert len(np.unique(image)) == 1, "the stored frame must be raw, not annotated"
    masker.reset()


def test_a_marked_frame_shows_as_marked_and_cannot_be_marked_twice(qapp, harness):
    masker = harness.open()
    _enter_problem_mode(masker, harness, 0.5)
    assert not masker.frame_marked

    masker.markBadFrame()
    harness.pump(masker)
    assert masker.frame_marked

    masker.markBadFrame()  # the control would be disabled; belt and braces
    harness.pump(masker)

    assert len(list((harness.root / "images").glob("*.png"))) == 1
    masker.reset()


def test_undo_removes_every_file_and_records_the_retraction(qapp, harness):
    masker = harness.open()
    _enter_problem_mode(masker, harness, 0.5)
    masker.markBadFrame()
    harness.pump(masker)

    masker.undoLastMark()
    harness.pump(masker)

    assert list((harness.root / "images").glob("*.png")) == []
    assert list((harness.root / "meta").glob("*.json")) == []
    assert not masker.frame_marked
    events = [
        line.split('"event": "')[1].split('"')[0]
        for line in (harness.root / "index.jsonl").read_text().splitlines()
    ]
    assert events == ["mark", "retract"]
    masker.reset()


def test_marking_is_impossible_outside_problem_mode_and_while_playing(qapp, harness):
    masker = harness.open()
    assert not masker.can_mark, "no detection is on screen to have judged"

    _enter_problem_mode(masker, harness, 0.5)
    assert masker.can_mark

    masker.playing = True
    assert not masker.can_mark
    assert not masker.problem_mode, "resuming playback leaves the mode"
    masker.reset()


def test_marking_is_impossible_without_a_video(qapp, harness):
    masker = VideoMasker()
    masker.video_output = QVideoSink()  # type: ignore[assignment]

    assert not masker.can_mark
    masker.markBadFrame()  # must not raise

    assert not (harness.root / "images").exists()


def test_the_frame_readout_is_available_before_a_video_is_opened(qapp, harness):
    """Regression: time_text used to raise AttributeError at startup."""
    masker = VideoMasker()

    assert masker.time_text == "00:00:00"
    assert masker.frame_index == 0


def test_stepping_moves_one_frame_and_pauses(qapp, harness):
    masker = harness.open()
    masker.playing = True
    masker.position = 0.5
    harness.pump(masker)
    start = masker.frame_index

    masker.stepFrame(1)
    harness.pump(masker)

    assert masker.frame_index == start + 1
    assert not masker.playing

    masker.stepFrame(-1)
    harness.pump(masker)
    assert masker.frame_index == start
    masker.reset()


def test_a_failed_save_is_reported_and_the_application_keeps_running(qapp, harness, monkeypatch):
    masker = harness.open()
    _enter_problem_mode(masker, harness, 0.5)
    failures: list[int] = []
    masker.mark_failed.connect(failures.append)

    def explode(self, request):
        raise OSError("disk full")

    monkeypatch.setattr(BadFrameStore, "mark", explode)
    masker.markBadFrame()
    harness.pump(masker)

    assert failures == [masker.frame_index]
    assert not masker.frame_marked
    # The researcher's position is untouched and navigation still works.
    masker.stepFrame(1)
    harness.pump(masker)
    masker.reset()


def test_a_detection_failure_leaves_the_control_disabled_and_retryable(qapp, harness):
    harness.detector_stub.fail = True
    masker = harness.open()

    _enter_problem_mode(masker, harness, 0.5)

    assert not masker.can_mark
    # Seeking away and back must ask again rather than staying stuck.
    harness.detector_stub.fail = False
    masker.position = 0.6
    harness.pump(masker)
    masker.position = 0.5
    harness.pump(masker)
    assert masker.can_mark
    masker.reset()


def test_the_sidecar_records_what_the_model_produced(qapp, harness):
    import json

    masker = harness.open()
    _enter_problem_mode(masker, harness, 0.5)
    frame_index = masker.frame_index

    masker.markBadFrame()
    harness.pump(masker)

    meta = json.loads(
        (harness.root / "meta" / f"experiment_{frame_index:06d}.json").read_text(encoding="utf-8")
    )
    assert meta["detection"] == {"boxes": _BOX, "conf": [0.9]}
    assert meta["model_id"] == "test-model:v1"
    assert meta["video_key"] == "cafe1234"
    assert meta["frame_index"] == frame_index
    assert meta["timestamp_ms"] == int(frame_index / 25.0 * 1000)
    masker.reset()
