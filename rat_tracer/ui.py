import argparse
import os
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from queue import Empty, Queue
from signal import SIGINT, signal
from sys import argv, exit
from time import time
from typing import TypeVar, override

import cv2
from cv2 import VideoCapture
from cv2.typing import MatLike
from numpy import ndarray
from PySide6.QtCore import (
    Property,
    QLocale,
    QObject,
    QSize,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat, QVideoSink
from PySide6.QtQml import QmlElement, QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication
from ultralytics import YOLO

from rat_tracer.bad_frames import (
    BadFrameStore,
    Detection,
    MarkRequest,
    configure_application_identity,
)
from rat_tracer.coverage import CoverageHistory
from rat_tracer.frame_detector import FrameDetector, YoloFrameDetector
from rat_tracer.lib import model_path
from rat_tracer.mask_render_core import FrameCapture, MaskRenderCore
from rat_tracer.paint import presence_frames
from rat_tracer.progress_cache import load_progress, save_progress, video_key
from rat_tracer.translations import resolve_translations

T = TypeVar("T")


logger = getLogger(__name__)
logger.setLevel(DEBUG)

masker_logger = logger.getChild("VideoMasker")
masker_logger.setLevel(DEBUG)

QML_IMPORT_NAME = "MyBackend"
QML_IMPORT_MAJOR_VERSION = 1


class CoverageComputer(QThread):
    frameReady = Signal()

    def __init__(self, history: CoverageHistory, video: Path, parent=None):
        super().__init__(parent)
        self._history = history
        self._video = video
        self._key = video_key(video)

    @property
    def key(self) -> str:
        """The video's content fingerprint, already paid for by the cache.

        Marked frames are keyed by it too, so the same physical video marked
        from different paths or machines deduplicates.
        """
        return self._key

    def run(self):
        start = time()
        logger.info("Processing video: %s", self._video)
        loaded = load_progress(self._key)
        if loaded is not None:
            self._history.replace_with(loaded)
            self.frameReady.emit()
        start_frame = len(self._history)
        logger.info("Starting from frame %d", start_frame)
        for _, mask in presence_frames(
            self._video, model=YOLO(model_path()), start_frame=start_frame
        ):
            self._history.append(mask)
            if self.isInterruptionRequested():
                save_progress(self._history, self._key)
                return
            self.frameReady.emit()
        else:
            self.frameReady.emit()
        save_progress(self._history, self._key)
        logger.info("Finished processing video: %s in %.2f seconds", self._video, time() - start)


class _Worker(QThread):
    """A thread that runs queued jobs one at a time until it is stopped.

    Both background jobs in problem reporting mode -- inference and saving --
    must stay off the UI thread and must stay ordered with respect to
    themselves: a mark and its undo cannot be allowed to interleave, and a
    stale detection must not overwrite a fresh one.
    """

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._queue: Queue = Queue()
        self._log = logger.getChild(name)

    def submit(self, job) -> None:
        self._queue.put(job)

    def stop(self) -> None:
        self._queue.put(None)

    def _drain(self, job):
        """Hook for workers that only care about the most recent job."""
        return job

    def run(self):
        self._startup()
        while True:
            job = self._queue.get()
            if job is None:
                self._log.debug("stopping")
                return
            job = self._drain(job)
            if job is None:
                continue
            self.run_job(job)

    def run_job(self, job) -> None:
        """Run one job, absorbing its failure.

        A failed job must never take the thread down with it: the researcher
        keeps navigating, and the next job still runs. Public so tests can
        drive the queue without a real thread and still exercise this.
        """
        try:
            job()
        except Exception:
            self._log.exception("job failed")

    def _startup(self) -> None:
        pass


class FrameDetectionWorker(_Worker):
    """Runs on-demand detection for whichever frame is being looked at."""

    detectionReady = Signal(int, object)
    detectionFailed = Signal(int)

    def __init__(self, detector: FrameDetector, parent=None):
        super().__init__("detector", parent)
        self._detector = detector

    def _startup(self) -> None:
        # The first inference in a process costs seconds while later ones cost
        # a fraction of one. Paying it here keeps it off the first frame the
        # researcher judges.
        self._detector.prewarm()

    def _drain(self, job):
        """Keep only the newest request.

        Seeking produces requests faster than inference answers them, and the
        researcher is looking at the newest frame -- so older ones are dropped
        rather than queued. A dropped frame is re-requested if they come back.
        """
        while True:
            try:
                newer = self._queue.get_nowait()
            except Empty:
                return job
            if newer is None:
                self._queue.put(None)  # let run() see the stop signal
                return None
            job = newer

    def request(self, frame_index: int, image: ndarray) -> None:
        def job():
            detection = self._detector.detect(image)
            self.detectionReady.emit(frame_index, detection)

        def guarded():
            try:
                job()
            except Exception:
                self.detectionFailed.emit(frame_index)
                raise

        self.submit(guarded)

    @property
    def model_id(self) -> str:
        return self._detector.model_id


class MarkStorageWorker(_Worker):
    """Writes and deletes marked frames without blocking defect hunting."""

    markSaved = Signal(int)
    markFailed = Signal(int)

    def __init__(self, store: BadFrameStore, parent=None):
        super().__init__("storage", parent)
        self._store = store

    def mark(self, request: MarkRequest) -> None:
        def job():
            try:
                self._store.mark(request)
            except Exception:
                # Surfaced to the researcher as "not saved"; the application
                # keeps running and their position is untouched.
                self.markFailed.emit(request.frame_index)
                raise
            self.markSaved.emit(request.frame_index)

        self.submit(job)

    def retract(self, video_key_value: str, frame_index: int, video_stem: str) -> None:
        self.submit(lambda: self._store.retract(video_key_value, frame_index, video_stem))


@QmlElement
class VideoMasker(QObject):
    # 1. Define a signal to notify QML when the property changes
    position_changed = Signal(float)
    video_changed = Signal(str)
    problem_mode_changed = Signal(bool)
    mark_state_changed = Signal()
    #: Emitted with the frame index once a mark is safely on disk.
    mark_saved = Signal(int)
    #: Emitted with the frame index when a mark could *not* be stored.
    mark_failed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._core = MaskRenderCore()
        self._video = None
        self._cap = None
        self._video_key: str | None = None
        self._thread_connection = None
        self._thread = None
        self._video_output = None
        self._video_sink = QVideoSink()
        self._strings = resolve_translations(QLocale.system().name())
        self._detector_worker: FrameDetectionWorker | None = None
        self._storage_worker: MarkStorageWorker | None = None
        self._store: BadFrameStore | None = None
        self._requested_detections: set[int] = set()
        self._last_mark: tuple[str, int, str] | None = None
        masker_logger.debug("__init__: VideoMasker initialized")

    def _emit_frame(self, frame: QVideoFrame) -> None:
        masker_logger.debug("_emit_frame: emitting frame (empty=%s)", not frame.isValid())
        self._video_sink.setVideoFrame(frame)

    def _get_video(self) -> str:
        masker_logger.debug("_get_video: %s", self._video)
        return str(self._video) if self._video else ""

    def _set_video(self, new_video: str) -> None:
        masker_logger.debug("_set_video: %s", new_video)
        self.reset()
        if not new_video:
            return
        self._video = Path(new_video)
        self.video_changed.emit(str(self._video))
        cap = VideoCapture(str(self._video))

        class FrameCaptureAdapter(FrameCapture):
            """Adapter to make cv2.VideoCapture compatible with the FrameCapture protocol."""

            def frame_count(self) -> int:
                return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def fps(self) -> float:
                return float(cap.get(cv2.CAP_PROP_FPS))

            @override
            def read(self, frame_idx: int) -> ndarray | None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    return None
                return frame

        self._core.open(FrameCaptureAdapter())
        self._cap = cap
        t = CoverageComputer(self._core.history, self._video)
        self._thread = t
        self._video_key = t.key
        self._thread_connection = t.frameReady.connect(self._on_frame_ready)
        t.start()
        self.mark_state_changed.emit()

    video = Property(str, _get_video, _set_video, notify=video_changed)

    @Slot(QUrl)
    def openVideo(self, url: QUrl) -> None:
        """Load a video from a QML file URL (from a file dialog or drag-and-drop)."""
        masker_logger.debug("openVideo: %s", url)
        local_path = url.toLocalFile()
        if local_path:
            self.video = local_path  # type: ignore[assignment]  # PySide Property setter

    @Slot()
    def _on_frame_ready(self):
        masker_logger.debug("_on_frame_ready")
        if self._core.frame_ready():
            self._schedule_render()

    def _get_video_output(self) -> QObject:
        masker_logger.debug("_get_video_output: %s", self._video_output)
        return self._video_output  # type: ignore[return-value]  # None until QML binds it

    def _set_video_output(self, video_output: QObject) -> None:
        masker_logger.debug("_set_video_output: %s", video_output)
        self._video_output = video_output
        if isinstance(video_output, QVideoSink):
            self._video_sink = video_output
        else:
            sink = video_output.findChild(QVideoSink)
            if not sink:
                raise ValueError("video_output must be a QVideoSink or contain one as a child")
            self._video_sink = sink
        self._on_frame_ready()

    video_output = Property(QObject, _get_video_output, _set_video_output)

    @Slot()
    def reset(self):
        masker_logger.debug("reset: resetting VideoMasker")
        t = self._thread
        if t:
            t.frameReady.disconnect(self._thread_connection)
            t.requestInterruption()
            t.wait()
        self._thread = None
        self._video = None
        self._cap = None
        self._video_key = None
        self._last_mark = None
        self._requested_detections.clear()
        self._stop_workers()
        self._core.reset()
        self._emit_frame(QVideoFrame())
        self.mark_state_changed.emit()

    def _stop_workers(self) -> None:
        for worker in (self._detector_worker, self._storage_worker):
            if worker is not None:
                worker.stop()
                worker.wait()
        self._detector_worker = None
        self._storage_worker = None

    def _get_playing(self) -> bool:
        masker_logger.debug("_get_playing: %s", self._core.playing)
        return self._core.playing

    def _set_playing(self, value: bool) -> None:
        masker_logger.debug("_set_playing: %s", value)
        was_problem_mode = self._core.problem_mode
        if self._core.set_playing(value):
            self._schedule_render()
        if self._core.problem_mode != was_problem_mode:
            # Resuming playback leaves problem reporting mode (see
            # MaskRenderCore.set_playing); QML has to hear about it.
            self.problem_mode_changed.emit(self._core.problem_mode)
        self.mark_state_changed.emit()

    playing = Property(bool, _get_playing, _set_playing)

    @Property(str, notify=position_changed)
    def time_text(self):
        masker_logger.debug("time_text")
        if self._cap is None:
            return "00:00:00"
        # Derived from the frame index, not from the capture's own cursor:
        # that cursor is a side effect of the last read, which the renderer and
        # the detector both perform, so it is not a reliable clock.
        elapsed_seconds = self._core.timestamp_ms(self._core.current_frame_index) // 1000
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @Property(int, notify=position_changed)
    def frame_index(self) -> int:
        """The displayed frame's index, so a researcher and a technician can
        refer to the same frame unambiguously."""
        if self._cap is None:
            return 0
        return self._core.current_frame_index

    def _get_position(self) -> float:
        masker_logger.debug("_get_position: %.3f", self._core.position)
        return self._core.position

    def _set_position(self, new_value: float) -> None:
        masker_logger.debug("_set_position: %.3f", new_value)
        if self._core.set_position(new_value):
            self._schedule_render()

    position = Property(float, _get_position, _set_position, notify=position_changed)

    def _schedule_render(self):
        masker_logger.debug("_schedule_render")
        QTimer.singleShot(1, self._rerender_if_needed)

    @Slot()
    def _rerender_if_needed(self):
        outcome = self._core.render_now()
        masker_logger.debug("_rerender_if_needed: should_emit=%s", outcome.should_emit)
        if not outcome.should_emit:
            self._request_detection_if_needed()
            return
        frame = (
            bgr_array_to_qvideoframe(outcome.image) if outcome.image is not None else QVideoFrame()
        )
        self._emit_frame(frame)
        self.position_changed.emit(self._core.position)
        self.mark_state_changed.emit()
        self._request_detection_if_needed()

    # --- problem reporting mode ---------------------------------------------

    def _get_problem_mode(self) -> bool:
        return self._core.problem_mode

    def _set_problem_mode(self, value: bool) -> None:
        masker_logger.debug("_set_problem_mode: %s", value)
        if self._core.problem_mode == value:
            return
        if value:
            # Judging a frame requires stopping on it, so entering the mode
            # pauses rather than leaving the researcher chasing the playhead.
            self._set_playing(False)
            self._start_detection()
        if self._core.set_problem_mode(value):
            self._schedule_render()
        self.problem_mode_changed.emit(self._core.problem_mode)
        self.mark_state_changed.emit()

    problem_mode = Property(bool, _get_problem_mode, _set_problem_mode, notify=problem_mode_changed)

    def _start_detection(self) -> None:
        if self._detector_worker is not None:
            return
        worker = FrameDetectionWorker(YoloFrameDetector())
        worker.detectionReady.connect(self._on_detection_ready)
        worker.detectionFailed.connect(self._on_detection_failed)
        self._detector_worker = worker
        worker.start()

    def _request_detection_if_needed(self) -> None:
        """Ask for the displayed frame's detection, once."""
        worker = self._detector_worker
        if worker is None:
            return
        frame_index = self._core.needs_detection()
        if frame_index is None or frame_index in self._requested_detections:
            return
        if self._core.rendered_frame_index != frame_index or self._core.raw_frame is None:
            # The frame is not on screen yet; the render that puts it there
            # asks again.
            return
        masker_logger.debug("_request_detection_if_needed: requesting frame %d", frame_index)
        self._requested_detections.add(frame_index)
        worker.request(frame_index, self._core.raw_frame.copy())

    @Slot(int, object)
    def _on_detection_ready(self, frame_index: int, detection: Detection) -> None:
        masker_logger.debug(
            "_on_detection_ready: frame %d, %d box(es)", frame_index, len(detection.boxes)
        )
        if self._core.set_detection(frame_index, detection):
            self._schedule_render()
        else:
            self.mark_state_changed.emit()

    @Slot(int)
    def _on_detection_failed(self, frame_index: int) -> None:
        # Allow a retry: the researcher can seek away and back, or re-enter the
        # mode, rather than being stuck with a permanently disabled control.
        self._requested_detections.discard(frame_index)

    # --- marking ------------------------------------------------------------

    @Property(bool, notify=mark_state_changed)
    def can_mark(self) -> bool:
        return self._core.can_mark and self._video is not None and self._video_key is not None

    @Property(bool, notify=mark_state_changed)
    def frame_marked(self) -> bool:
        """Whether the displayed frame is already stored.

        The control is stateful so the researcher sees the answer before
        acting, rather than discovering it by pressing.
        """
        if self._video_key is None:
            return False
        return self._ensure_store().is_marked(self._video_key, self._core.current_frame_index)

    def _ensure_store(self) -> BadFrameStore:
        if self._store is None:
            self._store = BadFrameStore()
        return self._store

    def _ensure_storage_worker(self) -> MarkStorageWorker:
        if self._storage_worker is None:
            worker = MarkStorageWorker(self._ensure_store())
            worker.markSaved.connect(self._on_mark_saved)
            worker.markFailed.connect(self._on_mark_failed)
            self._storage_worker = worker
            worker.start()
        return self._storage_worker

    @Slot()
    def markBadFrame(self) -> None:
        """Store the frame on screen as a detection failure.

        Reads the current frame; it never navigates, so the position and the
        recorded coverage are untouched.
        """
        if not self.can_mark:
            masker_logger.debug("markBadFrame: ignored, nothing judged on screen")
            return
        assert self._video is not None and self._video_key is not None
        frame_index = self._core.current_frame_index
        image = self._core.raw_frame
        detection = self._core.detection_for(frame_index)
        if image is None or detection is None:
            masker_logger.warning("markBadFrame: no raw frame or detection for %d", frame_index)
            return
        if self._ensure_store().is_marked(self._video_key, frame_index):
            masker_logger.debug("markBadFrame: frame %d is already marked", frame_index)
            return
        worker = self._ensure_storage_worker()
        detector = self._detector_worker
        request = MarkRequest(
            image=image.copy(),
            video_path=self._video,
            video_key=self._video_key,
            frame_index=frame_index,
            timestamp_ms=self._core.timestamp_ms(frame_index),
            detection=detection,
            model_id=detector.model_id if detector is not None else "unknown",
        )
        self._last_mark = (self._video_key, frame_index, self._video.stem)
        worker.mark(request)

    @Slot()
    def undoLastMark(self) -> None:
        """Delete everything stored for the most recent mark."""
        if self._last_mark is None:
            return
        key, frame_index, stem = self._last_mark
        self._last_mark = None
        self._ensure_storage_worker().retract(key, frame_index, stem)
        self.mark_state_changed.emit()

    @Slot(int)
    def stepFrame(self, delta: int) -> None:
        """Move exactly *delta* frames, pausing playback."""
        target = self._core.step_frame(delta)
        if target is None:
            return
        self._set_playing(False)
        self.position = target  # type: ignore[assignment]  # PySide Property setter

    @Slot(int)
    def _on_mark_saved(self, frame_index: int) -> None:
        masker_logger.info("Marked frame %d", frame_index)
        self.mark_state_changed.emit()
        self.mark_saved.emit(frame_index)

    @Slot(int)
    def _on_mark_failed(self, frame_index: int) -> None:
        masker_logger.error("Could not store frame %d", frame_index)
        self._last_mark = None
        self.mark_state_changed.emit()
        self.mark_failed.emit(frame_index)


def bgr_array_to_qvideoframe(bgr_arr: MatLike) -> QVideoFrame:
    """Converts a BGR NumPy array to a PySide6 QVideoFrame."""

    # 1. Convert BGR to BGRA for reliable memory alignment in Qt
    bgra_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2BGRA)
    height, width, _ = bgra_arr.shape

    # 2. Define the video frame format
    size = QSize(width, height)
    pixel_format = QVideoFrameFormat.PixelFormat.Format_BGRA8888
    frame_format = QVideoFrameFormat(size, pixel_format)

    # 3. Instantiate the empty frame
    frame = QVideoFrame(frame_format)

    # 4. Map the memory, copy the bytes, and unmap
    if frame.map(QVideoFrame.MapMode.WriteOnly):
        # frame.bits(0) returns a Python memoryview of the buffer
        frame_data = frame.bits(0)

        # Convert the NumPy array to raw bytes
        arr_bytes = bgra_arr.tobytes()

        # Reassign the memoryview slice with our array data
        frame_data[: len(arr_bytes)] = arr_bytes  # type: ignore[index]  # bits() returns a writable memoryview

        # Always unmap when finished to lock the data into the frame!
        frame.unmap()

    return frame


def format_qobject_children(obj: QObject, indent: str = "") -> str:
    """Recursively formats the QObject tree as an indented string for debugging."""
    lines = [f"{indent}{obj.__class__.__name__} (objectName='{obj.objectName()}')"]
    for child in obj.children():
        lines.append(format_qobject_children(child, indent + "  "))
    return "\n".join(lines)


def handleIntSignal(signum, frame):
    print("SIGINT received, quitting application...")
    QApplication.quit()


def main():
    if os.environ.get("RAT_TRACER_LOG_TIMING"):
        basicConfig(format="%(relativeCreated)d ms %(levelname)s:%(name)s: %(message)s")
    else:
        basicConfig()
    strings = resolve_translations(QLocale.system().name())
    parser = argparse.ArgumentParser(description=strings["cli_description"])
    parser.add_argument("-v", "--video", type=Path, default=None, help=strings["cli_video_help"])
    args, _ = parser.parse_known_args(argv[1:])
    if args.video is not None and not args.video.is_file():
        parser.error(strings["cli_video_not_found"].format(path=args.video))

    app = QGuiApplication(argv)
    # Marked frames live under a per-application data directory, which Qt
    # derives from this name; without it the directory would follow argv[0]
    # and differ per launch method.
    configure_application_identity()

    signal(SIGINT, handleIntSignal)

    QQuickStyle.setStyle("Material")
    engine = QQmlApplicationEngine()
    # VideoMasker is registered as the "MyBackend" QML module via @QmlElement,
    # so no import path is needed. Load Main.qml directly by absolute path.
    engine.rootContext().setContextProperty("tr", strings)
    engine.load(Path(__file__).parent / "Main.qml")

    if not engine.rootObjects():
        exit(-1)

    root = engine.rootObjects()[0]
    if logger.isEnabledFor(DEBUG):
        logger.debug("QObject tree:\n%s", format_qobject_children(root))

    masker = root.findChild(VideoMasker)

    assert masker is not None
    app.aboutToQuit.connect(masker.reset)

    if args.video is not None:
        masker.video = str(args.video)  # type: ignore[assignment]  # PySide Property setter

    exit_code = app.exec()  # exit immediately to investigate QML binding issues

    del engine
    exit(exit_code)


if __name__ == "__main__":
    main()
