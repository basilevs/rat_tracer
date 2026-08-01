import argparse
import os
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from queue import Empty, Queue, ShutDown
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
from rat_tracer.mask_render_core import FrameCapture
from rat_tracer.paint import presence_frames
from rat_tracer.progress_cache import load_progress, save_progress, video_key
from rat_tracer.review_session import ReviewSession, SessionListener
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

    #: Whether ``stop`` abandons jobs that are still queued. Work whose result
    #: is worthless once the video is closed says yes; work that must not be
    #: lost says no and is drained first.
    discards_queued_jobs = False

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._queue: Queue = Queue()
        self._log = logger.getChild(name)

    def submit(self, job) -> None:
        try:
            self._queue.put(job)
        except ShutDown:
            # Raced with teardown: the thread is on its way out and nothing is
            # left to run this.
            self._log.debug("dropping a job submitted after shutdown")

    def stop(self) -> None:
        self._queue.shutdown(immediate=self.discards_queued_jobs)

    def _drain(self, job):
        """Hook for workers that only care about the most recent job."""
        return job

    def run(self):
        self._startup()
        while True:
            try:
                job = self._queue.get()
            except ShutDown:
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

    # A detection is only interesting for a frame the researcher is looking at,
    # so anything still queued when the video closes is already worthless.
    discards_queued_jobs = True

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
            except ShutDown:
                # Teardown began while requests were still stacked up; let the
                # main loop see the shutdown rather than running one more.
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
    markRetracted = Signal(int)

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
        def job():
            self._store.retract(video_key_value, frame_index, video_stem)
            self.markRetracted.emit(frame_index)

        self.submit(job)


class QtDetectionService:
    """Runs detection on a worker thread, for the session to ask.

    Owns the thread's whole life: the session holds one of these for as long as
    it exists, while the thread underneath is created on first use and replaced
    after a video is closed.
    """

    def __init__(self, on_ready, on_failed):
        self._on_ready = on_ready
        self._on_failed = on_failed
        self._worker: FrameDetectionWorker | None = None

    def _ensure(self) -> FrameDetectionWorker:
        if self._worker is None:
            worker = FrameDetectionWorker(YoloFrameDetector())
            worker.detectionReady.connect(self._on_ready)
            worker.detectionFailed.connect(self._on_failed)
            self._worker = worker
            worker.start()
        return self._worker

    @property
    def model_id(self) -> str:
        return self._ensure().model_id

    def request(self, frame_index: int, image: ndarray) -> None:
        self._ensure().request(frame_index, image)

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
            self._worker = None


class QtMarkStorage:
    """Writes and removes marked frames on a worker thread.

    ``is_marked`` answers immediately from the store, because the control shows
    whether the frame on screen is stored every time the researcher moves.
    """

    def __init__(self, on_stored, on_failed, on_removed):
        self._on_stored = on_stored
        self._on_failed = on_failed
        self._on_removed = on_removed
        self._store: BadFrameStore | None = None
        self._worker: MarkStorageWorker | None = None

    def _ensure_store(self) -> BadFrameStore:
        # Opened on first use so that resolving the storage directory -- and
        # replaying its index -- is not paid by an application that never marks
        # anything.
        if self._store is None:
            self._store = BadFrameStore()
        return self._store

    def _ensure(self) -> MarkStorageWorker:
        if self._worker is None:
            worker = MarkStorageWorker(self._ensure_store())
            worker.markSaved.connect(self._on_stored)
            worker.markFailed.connect(self._on_failed)
            worker.markRetracted.connect(self._on_removed)
            self._worker = worker
            worker.start()
        return self._worker

    def is_marked(self, video_key: str, frame_index: int) -> bool:
        return self._ensure_store().is_marked(video_key, frame_index)

    def store(self, request: MarkRequest) -> None:
        self._ensure().mark(request)

    def remove(self, video_key: str, frame_index: int, video_stem: str) -> None:
        self._ensure().retract(video_key, frame_index, video_stem)

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
            self._worker = None


@QmlElement
class VideoMasker(QObject):
    """Qt's half of the application: properties, signals, timers and threads.

    Every decision belongs to :class:`~rat_tracer.review_session.ReviewSession`,
    which this class drives and reports completions to. What is left here is
    the part that genuinely needs Qt -- exposing state to QML, scheduling a
    render on the event loop, running detection and storage on threads, and
    turning an image into a ``QVideoFrame``.
    """

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
        # Named handlers rather than lambdas over ``self._session``: the
        # services are built before it exists, and each of these is a one-line
        # relay from a worker's signal into the session.
        self._detection = QtDetectionService(
            on_ready=self._detection_ready,
            on_failed=self._detection_failed,
        )
        self._storage = QtMarkStorage(
            on_stored=self._mark_stored,
            on_failed=self._mark_failed,
            on_removed=self._mark_removed,
        )
        self._session = ReviewSession(
            listener=SessionListener(
                schedule_render=self._schedule_render,
                state_changed=self._on_state_changed,
                mark_stored=self.mark_saved.emit,
                mark_failed=self.mark_failed.emit,
            ),
            storage=self._storage,
            detection=self._detection,
        )
        self._video = None
        self._cap = None
        self._thread_connection = None
        self._thread = None
        self._video_output = None
        self._video_sink = QVideoSink()
        self._strings = resolve_translations(QLocale.system().name())
        masker_logger.debug("__init__: VideoMasker initialized")

    @Slot(int, object)
    def _detection_ready(self, frame_index: int, detection: Detection) -> None:
        self._session.detection_ready(frame_index, detection)

    @Slot(int)
    def _detection_failed(self, frame_index: int) -> None:
        self._session.detection_failed(frame_index)

    @Slot(int)
    def _mark_stored(self, frame_index: int) -> None:
        self._session.mark_stored(frame_index)

    @Slot(int)
    def _mark_failed(self, frame_index: int) -> None:
        self._session.mark_failed(frame_index)

    @Slot(int)
    def _mark_removed(self, frame_index: int) -> None:
        self._session.mark_removed(frame_index)

    def _on_state_changed(self) -> None:
        self.mark_state_changed.emit()
        self.problem_mode_changed.emit(self._session.problem_mode)

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

        self._cap = cap
        t = CoverageComputer(self._session.history, self._video)
        self._thread = t
        self._session.open_video(FrameCaptureAdapter(), self._video, t.key)
        self._thread_connection = t.frameReady.connect(self._on_frame_ready)
        t.start()

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
        self._session.frame_processed()

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
        # Every thread this object started has to be joined here, not just the
        # cumulative pass: reset() is what ``aboutToQuit`` runs, and a worker
        # still parked on its queue at interpreter shutdown is destroyed while
        # running, which Qt treats as fatal and aborts the process.
        self._detection.stop()
        self._storage.stop()
        self._video = None
        self._cap = None
        self._session.close_video()
        self._emit_frame(QVideoFrame())

    def _get_playing(self) -> bool:
        return self._session.playing

    def _set_playing(self, value: bool) -> None:
        masker_logger.debug("_set_playing: %s", value)
        self._session.set_playing(value)

    playing = Property(bool, _get_playing, _set_playing)

    @Property(str, notify=position_changed)
    def time_text(self) -> str:
        return self._session.time_text

    @Property(int, notify=position_changed)
    def frame_index(self) -> int:
        """The displayed frame's index, so a researcher and a technician can
        refer to the same frame unambiguously."""
        return self._session.frame_index

    def _get_position(self) -> float:
        return self._session.position

    def _set_position(self, new_value: float) -> None:
        masker_logger.debug("_set_position: %.3f", new_value)
        self._session.seek(new_value)

    position = Property(float, _get_position, _set_position, notify=position_changed)

    def _schedule_render(self):
        masker_logger.debug("_schedule_render")
        QTimer.singleShot(1, self._rerender_if_needed)

    @Slot()
    def _rerender_if_needed(self):
        outcome = self._session.render_frame()
        masker_logger.debug("_rerender_if_needed: should_emit=%s", outcome.should_emit)
        if not outcome.should_emit:
            return
        frame = (
            bgr_array_to_qvideoframe(outcome.image) if outcome.image is not None else QVideoFrame()
        )
        self._emit_frame(frame)
        self.position_changed.emit(self._session.position)

    def _get_problem_mode(self) -> bool:
        return self._session.problem_mode

    def _set_problem_mode(self, value: bool) -> None:
        masker_logger.debug("_set_problem_mode: %s", value)
        self._session.set_problem_mode(value)

    problem_mode = Property(bool, _get_problem_mode, _set_problem_mode, notify=problem_mode_changed)

    @Property(bool, notify=mark_state_changed)
    def can_mark(self) -> bool:
        return self._session.can_mark

    @Property(bool, notify=mark_state_changed)
    def frame_marked(self) -> bool:
        return self._session.frame_marked

    @Slot()
    def toggleMark(self) -> None:
        """Store the frame on screen, or withdraw it if it is already stored."""
        self._session.toggle_mark()

    @Slot()
    def markBadFrame(self) -> None:
        self._session.mark()

    @Slot()
    def unmarkFrame(self) -> None:
        self._session.unmark()

    @Slot()
    def undoLastMark(self) -> None:
        self._session.undo()

    @Slot(int)
    def stepFrame(self, delta: int) -> None:
        """Move exactly *delta* frames, pausing playback."""
        self._session.step(delta)


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
