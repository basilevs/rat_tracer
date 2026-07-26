from logging import getLogger, basicConfig, DEBUG
from sys import argv, exit
from pathlib import Path
from time import time
from typing import TypeVar
from signal import signal, SIGINT

import cv2
from cv2 import CAP_PROP_POS_FRAMES, VideoCapture
from cv2.typing import MatLike

from PySide6.QtCore import Property, QObject, QThread, Slot, Signal, QSize, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtMultimedia import QVideoSink, QVideoFrame, QVideoFrameFormat
from PySide6.QtWidgets import QApplication


from ultralytics import YOLO

from rat_tracer.coverage import CoverageHistory
from rat_tracer.paint import apply_red_mask, presence_frames
from rat_tracer.lib import model_path


T = TypeVar('T')


logger = getLogger(__name__)
logger.setLevel(DEBUG)

QML_IMPORT_NAME = "MyBackend"
QML_IMPORT_MAJOR_VERSION = 1

class CoverageComputer(QThread):
    frameReady = Signal()

    def __init__(self, history: CoverageHistory, video: Path, parent=None):
        super().__init__(parent)
        self._history = history
        self._video = video

    def run(self):
        start = time()
        logger.info("Processing video: %s", self._video)
        for _, mask in presence_frames(self._video, model=YOLO(model_path())):
            self._history.append(mask)
            if self.isInterruptionRequested():
                return
            self.frameReady.emit()
        logger.info("Finished processing video: %s in %.2f seconds", self._video, time() - start)


@QmlElement
class VideoMasker(QObject):
    # 1. Define a signal to notify QML when the property changes
    position_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = CoverageHistory()
        self._position = 0.0
        self._video = Path("input/2026-02-07-2.mp4")
        self._thread_connection = None
        self._thread = None
        self._playing = True
        self._video_output = None
        self._video_sink = QVideoSink()
        self._cap = None
        self._total_frame_count = 0.0
        self._frame_count = 0
        self._pending_position = None
        self._do_render_pending = False

    def _get_video(self) -> str:
        return str(self._video)

    def _set_video(self, new_video: str) -> None:
        self.reset()
        self._video = Path(new_video)
        self._cap = VideoCapture(str(self._video))
        self._total_frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        t = CoverageComputer(self._history, self._video)
        self._thread  = t
        self._thread_connection =  t.frameReady.connect(self._on_frame_ready)
        self._thread.start()

    video = Property(str, _get_video, _set_video)

    @Slot()
    def _on_frame_ready(self):
        if self._playing:
            cap = self._cap
            if cap:
                self.position = float(len(self._history)-1) / self._total_frame_count

    def _get_video_output(self) -> QObject:
        return self._video_output

    def _set_video_output(self, video_output: QObject) -> None:
        self._video_output = video_output
        if isinstance(video_output, QVideoSink):
            self._video_sink = video_output
        else:
            self._video_sink = video_output.findChild(QVideoSink)
        if not self._video_sink:
            raise ValueError("video_output must be a QVideoSink or contain one as a child")
        self._on_frame_ready()

    video_output = Property(QObject, _get_video_output, _set_video_output)

    @Slot()
    def reset(self):
        logger.debug("Resetting VideoMasker")
        t = self._thread
        if t:
            t.frameReady.disconnect(self._thread_connection)
            t.requestInterruption()
            t.wait()
        self._thread = None
        self._video = None
        self._history.clear()
        self._position = 0.0
        self._cap = None
        self._total_frame_count = 0.0

    def _get_playing(self) -> bool:
        return self._playing

    def _set_playing(self, value: bool) -> None:
        self._playing = value
        self._on_frame_ready()

    playing = Property(bool, _get_playing, _set_playing)

    @Property(str, notify=position_changed)
    def time_text(self):
        cap = self._cap
        if not cap:
            return "00:00:00"
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1.0
        elapsed_seconds = int(self._position * total_frames / fps)
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _get_position(self) -> float:
        return self._position

    def _set_position(self, new_value: float) -> None:
        # Prevent infinite binding loops by only updating if the value actually changed
        if self._pending_position != new_value:
            # Coalesce: overwrite the pending value; schedule a single render if not already queued
            self._pending_position = new_value
            if not self._do_render_pending:
                self._do_render_pending = True
                QTimer.singleShot(0, self._do_render)

    position = Property(float, _get_position, _set_position, notify=position_changed)

    @Slot()
    def _do_render(self):
        try: 
            frame = QVideoFrame()
            while self._position != self._pending_position:
                new_value = self._pending_position
                self._position = new_value
                capture = self._cap
                if not capture:
                    self._video_sink.setVideoFrame(QVideoFrame())
                    return
                frame_idx = int(new_value * capture.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.debug("Sliding to frame %d", frame_idx)
                capture.set(CAP_PROP_POS_FRAMES, frame_idx)
                ok, img = capture.read()
                if new_value != self._pending_position:
                    logger.debug("Position changed during render; skipping frame %d", frame_idx)
                    continue
                if not ok:
                    logger.warning("Cannot read frame %d", frame_idx)
                    return
                if frame_idx < 0 or frame_idx >= len(self._history):
                    logger.debug("Frame index %d out of range", frame_idx)
                else:
                    apply_red_mask(img, self._history[frame_idx])
                frame = bgr_array_to_qvideoframe(img)
            self._frame_count += 1
            self._video_sink.setVideoFrame(frame)
            self.position_changed.emit(self._position)
        finally:
            self._do_render_pending = False

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
        frame_data[:len(arr_bytes)] = arr_bytes

        # Always unmap when finished to lock the data into the frame!
        frame.unmap()

    return frame



def print_qobject_children(obj: QObject, indent: str = ""):
    """Recursively prints the QObject tree for debugging purposes."""
    prefix = indent
    print(f"{prefix}{obj.__class__.__name__} (objectName='{obj.objectName()}')")
    for child in obj.children():
        print_qobject_children(child, indent + "  ")




def handleIntSignal():  # pylint: disable=unused-argument
    print("SIGINT received, quitting application...")
    QApplication.quit()

def main():
    basicConfig()
    app = QGuiApplication(argv)

    signal(SIGINT, handleIntSignal)

    QQuickStyle.setStyle("Material")
    engine = QQmlApplicationEngine()
    # VideoMasker is registered as the "MyBackend" QML module via @QmlElement,
    # so no import path is needed. Load Main.qml directly by absolute path.
    engine.load(Path(__file__).parent / "Main.qml")

    if not engine.rootObjects():
        exit(-1)

    root = engine.rootObjects()[0]
    print_qobject_children(root)

    video = Path("input/2026-02-07-2.mp4")

    masker = root.findChild(VideoMasker)
    masker.video = str(video)

    app.aboutToQuit.connect(masker.reset)

    exit_code = app.exec() # exit immediately to investigate QML binding issues

    del engine
    exit(exit_code)


if __name__ == "__main__":
    main()