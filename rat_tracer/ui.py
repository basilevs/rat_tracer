from logging import getLogger, basicConfig, DEBUG
from pprint import pprint
from sys import argv, path, exit
from pathlib import Path
import random
import sys
from threading import Lock
from time import time
from typing import Generic, TypeVar

from cv2 import CAP_PROP_POS_FRAMES, VideoCapture

from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtCore import Qt, Signal
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QObject, Signal, Slot
from threading import Lock
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink, QVideoFrame
from PySide6.QtWidgets import QSlider
from PySide6.QtQuick import QQuickItem

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



def print_qobject_children(obj: QObject, indent: str = ""):
    """Recursively prints the QObject tree for debugging purposes."""
    prefix = indent
    print(f"{prefix}{obj.__class__.__name__} (objectName='{obj.objectName()}')")
    for child in obj.children():
        print_qobject_children(child, indent + "  ")


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
    print_qobject_children(root)

    videoOutput = root.findChild(QVideoSink)
    assert videoOutput, "QVideoSink not found in QML"
    def set_video_frame(frame: QVideoFrame):
        if root.property("playing"):
            videoOutput.setVideoFrame(frame)


    history = CoverageHistory()

    slider = root.findChild(QSlider, "slider_here")
    assert slider, "QSlider not found in QML"

    video = Path("input/2026-02-07-2.mp4")

    cap = VideoCapture(str(video))


    def slide_to_frame(frame_idx: int):
        if frame_idx < 0 or frame_idx >= len(history):
            logger.warning("Frame index %d out of range", frame_idx)
            return
        logger.info("Sliding to frame %d", frame_idx)
        cap.set(CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_idx}")
        apply_red_mask(img, history[frame_idx])
        frame = bgr_array_to_qvideoframe(img)
        videoOutput.setVideoFrame(frame)
        if slider.value != frame_idx:
            slider.setValue(frame_idx)

    class BackgroundWorker(QThread):
        def run(self):
            start = time()
            logger.info("Processing video: %s", video)
            for _, mask in presence_frames(video, model=YOLO(best_model_path)):
                history.append(mask)
                if root.property("playing"):
                    slide_to_frame(len(history)-1)
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