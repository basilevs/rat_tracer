from logging import getLogger, basicConfig, DEBUG
from pprint import pprint
from sys import argv, path, exit
from pathlib import Path
import random
import sys
from threading import Lock
from time import time
from typing import Generic, TypeVar

from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtCore import Qt, Signal
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QObject, Signal, Slot
from threading import Lock
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink, QVideoFrame
from ultralytics import YOLO

from rat_tracer.coverage import CoverageHistory
from rat_tracer.paint import apply_red_mask, presence_frames
from rat_tracer.lib import best_model_path


T = TypeVar('T')

import cv2
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat

logger = getLogger(__name__)
logger.setLevel(DEBUG)

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

if __name__ == "__main__":
    basicConfig()
    app = QGuiApplication(argv)
    QQuickStyle.setStyle("Material")
    engine = QQmlApplicationEngine()
    # Add the current directory to the import paths and load the main module.
    engine.addImportPath(path[0])
    engine.loadFromModule(".", "Main")

    if not engine.rootObjects():
        exit(-1)

    root = engine.rootObjects()[0]

    videoOutput = root.findChild(QVideoSink)
    assert videoOutput is not None, "QVideoSink not found in QML"
    

    def set_video_frame(frame: QVideoFrame):
        if root.property("playing"):
            videoOutput.setVideoFrame(frame)


    history = CoverageHistory()
    root = engine.rootObjects()[0]

    videoOutput = root.findChild(QVideoSink)
    assert videoOutput is not None, "QVideoSink not found in QML"

    def set_video_frame(frame: QVideoFrame):
        if root.property("playing"):
            videoOutput.setVideoFrame(frame)


    class BackgroundWorker(QThread):
        def run(self):
            video = Path("input/2026-02-07-2.mp4")
            start = time()
            logger.info("Processing video: %s", video)
            for img, mask in presence_frames(video, model=YOLO(best_model_path)):
                history.append(mask)
                apply_red_mask(img, history.visited)
                frame = bgr_array_to_qvideoframe(img)
                set_video_frame(frame)
                if self.isInterruptionRequested():
                    return
            logger.info("Finished processing video: %s in %.2f seconds", video, time() - start)

    thread = BackgroundWorker()
    app.aboutToQuit.connect(thread.requestInterruption)
    app.aboutToQuit.connect(thread.wait)

    thread.start()

    exit_code = app.exec()

    del engine
    exit(exit_code)