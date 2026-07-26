from logging import getLogger, basicConfig, DEBUG, root
from sys import argv, path, exit
from pathlib import Path
from time import time
from typing import TypeVar

from cv2 import CAP_PROP_POS_FRAMES, VideoCapture

from signal import signal, SIGINT

from PySide6.QtCore import Property, QObject, QThread, Slot, Signal, QSize
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtMultimedia import QVideoSink, QVideoFrame, QVideoFrameFormat
from PySide6.QtWidgets import QApplication
from PySide6.QtQuick import QQuickItem

import cv2
from cv2.typing import MatLike
import numpy as np


from ultralytics import YOLO

from rat_tracer.coverage import CoverageHistory
from rat_tracer.paint import apply_red_mask, presence_frames
from rat_tracer.lib import best_model_path


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
        for _, mask in presence_frames(self._video, model=YOLO(best_model_path)):
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
        self._threadConnection = None
        self._thread = None
        self._playing = True
        self._video_sink = QVideoSink()
        self._cap = None

    @Property(str)
    def video(self):
        return str(self._video)

    @video.setter
    def video(self, new_video: str):
        self.reset()
        self._video = Path(new_video)
        self._cap = VideoCapture(str(self._video))
        t = CoverageComputer(self._history, self._video)
        self._thread  = t
        self._threadConnection =  t.frameReady.connect(self._on_frame_ready)
        self._thread.start()

    @Slot()
    def _on_frame_ready(self):
        if self._playing:
            cap = self._cap
            if cap:
                self.position = float(len(self._history)-1) / self._cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @Property(QQuickItem)
    def video_output(self):
        return self._video_output

    @video_output.setter
    def video_output(self, video_output: QQuickItem):
        self._video_output = video_output
        self._video_sink = video_output.property("videoSink")
        self._on_frame_ready()

    @Slot()
    def reset(self):
        logger.debug("Resetting VideoMasker")
        t = self._thread
        if t:
            t.frameReady.disconnect(self._threadConnection)
            t.requestInterruption()
            t.wait()
        self._thread = None
        self._video = None
        self._history.clear()
        self._position = 0.0
        self._cap = None

    @Property(bool)
    def playing(self):
        return self._playing

    @playing.setter
    def playing(self, value: bool):
        self._playing = value
        self._on_frame_ready()

    @Property(float, notify=position_changed)
    def position(self):
        return self._position

    @position.setter
    def position(self, new_value: float):
        # Prevent infinite binding loops by only updating if the value actually changed
        if self._position != new_value:
            self._position = new_value
        capture = self._cap
        if not capture:
            self._video_sink.setVideoFrame(QVideoFrame())
            return
        new_value = int(new_value * capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug("Sliding to frame %d", new_value)
        capture.set(CAP_PROP_POS_FRAMES, int(new_value))
        ok, img = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {new_value}")
        if new_value < 0 or new_value >= len(self._history):
            logger.debug("Frame index %d out of range", new_value)
        else:
            apply_red_mask(img, self._history[int(new_value)])
        frame = bgr_array_to_qvideoframe(img)
        self._video_sink.setVideoFrame(frame)

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
    # Add the current directory to the import paths and load the main module.
    engine.addImportPath(path[0])
    engine.loadFromModule(".", "Main")

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