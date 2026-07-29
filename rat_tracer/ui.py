import argparse
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from signal import SIGINT, signal
from sys import argv, exit
from time import time
from typing import TypeVar

import cv2
from cv2 import CAP_PROP_POS_FRAMES, VideoCapture
from cv2.typing import MatLike
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
from rat_tracer.paint import apply_red_mask, presence_frames
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
        self._history = CoverageHistory()
        self._position = 0.0
        self._video = None
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
        self._mask_rendered = False
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
        self._cap = VideoCapture(str(self._video))
        self._total_frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        t = CoverageComputer(self._history, self._video)
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
        total = self._total_frame_count
        if total == 0:
            masker_logger.debug("_on_frame_ready: no frames yet (total=0)")
            return
        last_frame = len(self._history) - 1
        processed_position = float(len(self._history) - 1) / total
        masker_logger.debug(
            "Frame ready: %d/%d, playing: %s, mask_rendered: %s",
            last_frame,
            total,
            self._playing,
            self._mask_rendered,
        )
        if self._playing:
            cap = self._cap
            if cap:
                # PySide Property setter
                self.position = processed_position  # type: ignore[assignment]
        else:
            if not self._mask_rendered and self.position < processed_position:
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
        self._history.clear()
        self._position = 0.0
        self._cap = None
        self._total_frame_count = 0.0
        self._emit_frame(QVideoFrame())

    def _get_playing(self) -> bool:
        masker_logger.debug("_get_playing: %s", self._playing)
        return self._playing

    def _set_playing(self, value: bool) -> None:
        masker_logger.debug("_set_playing: %s", value)
        self._playing = value
        self._on_frame_ready()

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
        masker_logger.debug("_get_position: %.3f", self._position)
        return self._position

    def _set_position(self, new_value: float) -> None:
        masker_logger.debug("_set_position: %.3f", new_value)
        # Prevent infinite binding loops by only updating if the value actually changed
        if self._pending_position != new_value:
            # Coalesce: overwrite the pending value; schedule a single render if not already queued
            self._pending_position = new_value
            masker_logger.debug("Position requested %.3f", new_value)
            self._schedule_render()

    position = Property(float, _get_position, _set_position, notify=position_changed)

    def _schedule_render(self):
        masker_logger.debug("_schedule_render: position %.3f", self._pending_position)
        if not self._do_render_pending:
            self._do_render_pending = True
            QTimer.singleShot(1, self._rerender_if_needed)

    @Slot()
    def _rerender_if_needed(self):
        masker_logger.debug(
            "_rerender_if_needed: position=%.3f pending=%s mask_rendered=%s",
            self._position,
            self._pending_position,
            self._mask_rendered,
        )
        try:
            if self._position == self._pending_position and (
                self._mask_rendered
                or not self._history.contains(self._position_to_frame_index(self._position))
            ):
                masker_logger.debug("_rerender_if_needed: nothing to render")
                return
            new_value = self._pending_position
            assert new_value is not None
            self._position = new_value
            frame = self._produce_frame(new_value)
            self._emit_frame(frame)

            self.position_changed.emit(self._position)
        finally:
            self._do_render_pending = False

    def _produce_frame(self, position: float):
        frame = QVideoFrame()
        masker_logger.debug("_produce_frame: rendering frame for position %.3f", position)
        capture = self._cap
        if not capture:
            masker_logger.warning("_produce_frame: no video capture available for rendering")
            self._emit_frame(QVideoFrame())
            return
        frame_idx = self._position_to_frame_index(position)
        masker_logger.debug("_produce_frame: sliding to frame %d", frame_idx)
        capture.set(CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = capture.read()
        if not ok:
            masker_logger.warning("_produce_frame: cannot read frame %d", frame_idx)
            return
        self._mask_rendered = False
        if frame_idx < 0 or frame_idx >= len(self._history):
            masker_logger.debug("_produce_frame: frame index %d is not processed yet", frame_idx)
        else:
            apply_red_mask(img, self._history[frame_idx])
            self._mask_rendered = True
        frame = bgr_array_to_qvideoframe(img)
        self._frame_count += 1
        return frame

    def _position_to_frame_index(self, position: float) -> int:
        idx = int(position * self._total_frame_count)
        masker_logger.debug("_position_to_frame_index: %.3f -> %d", position, idx)
        return idx


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
