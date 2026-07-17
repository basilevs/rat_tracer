from pprint import pprint
from sys import argv, path, exit
from pathlib import Path
import random
import sys
from threading import Thread
from typing import Generic, TypeVar

from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtCore import Qt, Signal
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink, QVideoFrame
from ultralytics import YOLO

from rat_tracer.paint import apply_red_mask, presence_frames
from rat_tracer.lib import best_model_path


T = TypeVar('T')

import cv2
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat

def bgr_array_to_qvideoframe(bgr_arr: np.ndarray) -> QVideoFrame:
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

class Throttle(QObject):
    """ A QObject that emits a signal with a new frame. """
    _newFrame = Signal()
    ready = Signal(QVideoFrame)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None
        self.count = 0
        self.processed = 0
        self._newFrame.connect(self._processQueue)

    @Slot(QVideoFrame)
    def set(self, frame):
        self.frame = frame
        self.count += 1
        self._newFrame.emit()
    
    def _processQueue(self):
        if self.processed < self.count: # ignore thread safety for now
            self.processed = self.count
            self.ready.emit(self.frame)



if __name__ == "__main__":
    app = QGuiApplication(argv)
    QQuickStyle.setStyle("Material")
    engine = QQmlApplicationEngine()
    # Add the current directory to the import paths and load the main module.
    engine.addImportPath(path[0])
    engine.loadFromModule(".", "Main")

    if not engine.rootObjects():
        exit(-1)

    frameThrottle = Throttle()

    class BackgroundWorker(QThread):
        def run(self):
            for img, mask in presence_frames(Path("input/2026-02-07-2.mp4"), model=YOLO(best_model_path)):
                apply_red_mask(img, mask)
                frame = bgr_array_to_qvideoframe(img)
                frameThrottle.set(frame)
                if self.isInterruptionRequested():
                    return

    thread = BackgroundWorker()
    app.aboutToQuit.connect(thread.requestInterruption)
    app.aboutToQuit.connect(thread.wait)

    root = engine.rootObjects()[0]

    videoOutput = root.findChild(QVideoSink)
    assert videoOutput is not None, "QVideoSink not found in QML"
    

    def setVideoFrame(frame: QVideoFrame):
        if root.property("playing"):
            videoOutput.setVideoFrame(frame)

    frameThrottle.ready.connect(setVideoFrame)
    thread.start()

    exit_code = app.exec()

    del engine
    exit(exit_code)