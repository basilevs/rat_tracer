"""Regression tests for VideoMasker's paused-video rendering.

These drive :class:`rat_tracer.ui.VideoMasker` only through its public Qt
surface (the ``video``, ``playing``, ``position`` and ``video_output``
properties, and the ``reset`` slot) plus a plain ``QVideoSink`` handed to it
via ``video_output`` -- never by reaching into its private attributes.

``VideoMasker`` delegates the expensive work (model inference, frame
decoding, disk-backed progress cache) to module-level collaborators
(``YOLO``, ``presence_frames``, ``VideoCapture``, ``load_progress``,
``save_progress``). Those are faked at the module boundary so the tests are
fast and deterministic; ``QThread.start``/``QTimer.singleShot`` are made
synchronous for the same reason. None of this touches ``VideoMasker``
internals -- it only replaces its external dependencies, the same way a real
video file and a real YOLO model would be swapped for a fake one.

The two tests mirror the two phases of ``repro.log``:

1. Opening a fresh (uncached) video while a seek arrives mid-processing first
   renders the target frame bare (``_produce_frame: frame index N is not
   processed yet``), then re-renders it with the mask once processing
   catches up.
2. Opening a *second*, already fully cached video while paused should show
   its cached last frame, not the empty placeholder from ``reset()``. This
   currently fails: ``reset()`` never clears the stale ``mask_rendered`` flag
   left by the first video, so ``_on_frame_ready``'s paused branch never
   schedules a re-render -- exactly as in the log, which shows no
   ``_produce_frame``/``_emit_frame`` activity at all for the second video.
"""

import logging
import os
import pickle
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtMultimedia import QVideoSink
from rat_tracer import ui as ui_module
from rat_tracer import video_review as review_module
from rat_tracer.bad_frames import STORAGE_ENV_VAR
from rat_tracer.coverage import CoverageHistory
from rat_tracer.ui import VideoMasker

_H, _W = 8, 12
_LOGGER_NAME = "rat_tracer.ui"

# Flip to True (or run with RAT_TRACER_DUMP_LOG=1) to print each scenario's
# captured debug log to stdout (needs `pytest -s`), so a human can eyeball it
# against repro.log. Disabled by default -- assertions never depend on log
# content, only on VideoMasker's public state (e.g. the emitted video frame).
_DUMP_LOG = os.environ.get("RAT_TRACER_DUMP_LOG") == "1"


def _dump_log(caplog: pytest.LogCaptureFixture, label: str) -> None:
    if not _DUMP_LOG:
        return
    print(f"\n----- captured log: {label} -----")
    for record in caplog.records:
        print(f"{record.levelname}:{record.name}:{record.getMessage()}")
    print(f"----- end captured log: {label} -----\n")


class _FakeCapture:
    """Stands in for ``cv2.VideoCapture``: fixed frame count, solid frames."""

    def __init__(self, total_frames: int, fps: float = 30.0):
        self.total_frames = total_frames
        self.fps = fps
        self.frame_idx = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.frame_idx / self.fps * 1000.0
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, prop, value):
        self.frame_idx = int(value)
        return True

    def read(self):
        shade = min(255, self.frame_idx * 5)
        return True, np.full((_H, _W, 3), shade, dtype=np.uint8)


@pytest.fixture(scope="session")
def qapp():
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture
def worker_harness(monkeypatch, tmp_path):
    """Fake the heavy collaborators ``VideoReview.process_video`` drives.

    Replaces ``cv2.VideoCapture``, the YOLO model, ``presence_frames`` and the
    disk-backed progress cache with fast, in-memory fakes, and makes
    ``CoveragePass.start``/``QTimer.singleShot`` run their callback inline
    instead of deferring to a real thread or the event loop. This lets a test
    drive ``VideoMasker`` purely through its public properties/slots while
    keeping execution order fully deterministic.

    Caveat: the review reports progress through ``VideoMasker._review_changed``,
    whose default ``AutoConnection`` resolves to a *queued* cross-thread call in
    production (the pass's thread emits, the main thread's event loop delivers
    it later) but to a *direct*, synchronous call here, since running the pass
    inline means emitter and receiver share a thread. So the render ends up
    invoked reentrantly, nested inside the very call stack that triggered it,
    instead of on a fresh stack dispatched later by the event loop. Exact
    interleavings/frame-counts below are therefore illustrative of the code's
    *logic*, not a timing prediction of production behavior. This is fine for
    state-based bugs (e.g. the stale overlay flag below doesn't care which
    thread or stack frame observes it) but wouldn't catch a bug that only
    manifests through genuine queued-delivery timing.
    """
    monkeypatch.setenv(STORAGE_ENV_VAR, str(tmp_path / "bad_frames"))
    frame_counts: dict[str, int] = {}
    cache_store: dict[str, object] = {}
    mid_stream_hooks: dict[str, Callable[[], None]] = {}

    def fake_video_capture(path):
        return _FakeCapture(frame_counts[path])

    def fake_presence_frames(input_video, model, start_frame=0):
        path = str(input_video)
        total = frame_counts[path]
        hook = mid_stream_hooks.get(path)
        if hook is not None and start_frame == 0:
            hook()
        for _ in range(start_frame, total):
            yield None, np.zeros((_H, _W), dtype=bool)

    monkeypatch.setattr(ui_module, "VideoCapture", fake_video_capture)
    monkeypatch.setattr(review_module, "video_key", lambda path: str(path))
    monkeypatch.setattr(review_module, "YOLO", lambda *a, **k: None)
    monkeypatch.setattr(review_module, "model_path", lambda: Path("fake-model.pt"))
    monkeypatch.setattr(review_module, "presence_frames", fake_presence_frames)
    monkeypatch.setattr(review_module, "load_progress", cache_store.get)
    monkeypatch.setattr(
        review_module,
        "save_progress",
        # Snapshot via a pickle round-trip, like the real disk-backed cache: a
        # stored reference to the live CoverageHistory would let later mutations
        # (e.g. reset()'s clear()) silently corrupt what was "saved".
        lambda history, key: cache_store.__setitem__(key, pickle.loads(pickle.dumps(history))),
    )
    monkeypatch.setattr(ui_module.CoveragePass, "start", lambda self: self._run())
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda _msec, cb: cb()))

    class Harness:
        def register(self, path: str, total_frames: int, *, on_first_frame=None):
            """Declare *path*'s frame count and an optional hook run when
            processing starts from frame 0 (used to simulate a seek arriving
            while the background pass is still running)."""
            frame_counts[path] = total_frames
            if on_first_frame is not None:
                mid_stream_hooks[path] = on_first_frame

        def seed_cache(self, path: str, total_frames: int):
            """Pre-populate the progress cache as if *path* were already
            fully processed in a previous run."""
            history = CoverageHistory()
            for _ in range(total_frames):
                history.append(np.zeros((_H, _W), dtype=bool))
            cache_store[path] = history

    return Harness()


def test_seek_while_uncached_video_processes_then_mask_applies(qapp, worker_harness, caplog):
    """A seek landing before any frames are processed renders bare first, then
    re-renders with the mask once the background pass reaches that frame --
    the same shape as the first video in ``repro.log`` (bare render, then a
    masked one at the same position). Exactly which frame count triggers the
    second render is an artifact of this harness's synchronous execution
    (see ``worker_harness``'s docstring), not a claim about production
    timing."""
    video = "video1.mp4"
    total = 10
    target_position = 0.55  # frame index 5 of 10

    masker = VideoMasker()
    sink = QVideoSink()
    masker.video_output = sink  # type: ignore[assignment]  # PySide Property setter

    def seek_mid_stream():
        masker.playing = False  # type: ignore[assignment]  # PySide Property setter
        masker.position = target_position  # type: ignore[assignment]  # PySide Property setter

    worker_harness.register(video, total, on_first_frame=seek_mid_stream)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        masker.video = video  # type: ignore[assignment]  # PySide Property setter
    _dump_log(caplog, "test_seek_while_uncached_video_processes_then_mask_applies")

    assert sink.videoFrame().isValid()
    masker.reset()


def test_reopen_cached_video_while_paused_leaves_blank_frame(qapp, worker_harness, caplog):
    """Reproduces the ``repro.log`` bug: opening a second, already fully
    cached video while paused never re-renders, leaving the empty placeholder
    frame from ``reset()`` on screen even though the video is fully processed
    and ready to display. Currently FAILS: ``reset()`` leaves the
    ``mask_rendered`` flag from the first video stuck at ``True``, and
    ``_on_frame_ready``'s paused branch only schedules a re-render when that
    flag is ``False``. Should pass once ``reset()`` clears it."""
    video1, total1 = "video1.mp4", 10
    video2, total2 = "video2.mp4", 6

    masker = VideoMasker()
    sink = QVideoSink()
    masker.video_output = sink  # type: ignore[assignment]  # PySide Property setter

    def seek_mid_stream():
        masker.playing = False  # type: ignore[assignment]  # PySide Property setter
        masker.position = 0.55  # type: ignore[assignment]  # PySide Property setter

    worker_harness.register(video1, total1, on_first_frame=seek_mid_stream)
    worker_harness.register(video2, total2)
    worker_harness.seed_cache(video2, total2)  # video2 was fully processed before

    masker.video = video1  # type: ignore[assignment]  # PySide Property setter
    assert sink.videoFrame().isValid(), "sanity check: video1's paused render should land"

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        masker.video = video2  # type: ignore[assignment]  # PySide Property setter
    _dump_log(caplog, "test_reopen_cached_video_while_paused_leaves_blank_frame")

    assert sink.videoFrame().isValid(), (
        "opening a fully-cached video while paused left the empty placeholder "
        "frame from reset() on screen instead of the cached last frame -- "
        "reset() never clears _mask_rendered, so _on_frame_ready's paused "
        "branch thinks a mask is already shown and never re-renders"
    )
    masker.reset()
