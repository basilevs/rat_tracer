import argparse
import os
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
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

from rat_tracer.coverage import CoverageHistory
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


@QmlElement
class VideoMasker(QObject):
    # 1. Define a signal to notify QML when the property changes
    position_changed = Signal(float)
    video_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._core = MaskRenderCore()
        self._video = None
        self._thread_connection = None
        self._thread = None
        self._video_output = None
        self._video_sink = QVideoSink()
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
        self._core.reset()
        self._emit_frame(QVideoFrame())

    def _get_playing(self) -> bool:
        masker_logger.debug("_get_playing: %s", self._core.playing)
        return self._core.playing

    def _set_playing(self, value: bool) -> None:
        masker_logger.debug("_set_playing: %s", value)
        if self._core.set_playing(value):
            self._schedule_render()

    playing = Property(bool, _get_playing, _set_playing)

    @Property(str, notify=position_changed)
    def time_text(self):
        masker_logger.debug("time_text")
        cap = self._cap
        if not cap:
            return "00:00:00"
        elapsed_seconds = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

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
            return
        frame = (
            bgr_array_to_qvideoframe(outcome.image) if outcome.image is not None else QVideoFrame()
        )
        self._emit_frame(frame)
        self.position_changed.emit(self._core.position)


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
