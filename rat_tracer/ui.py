import argparse
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from signal import SIGINT, signal
from sys import argv, exit
from threading import Event, Thread
from typing import override

import cv2
from cv2 import VideoCapture
from cv2.typing import MatLike
from numpy import ndarray
from PySide6.QtCore import (
    Property,
    QLocale,
    QObject,
    QSize,
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

from rat_tracer.background import Job
from rat_tracer.bad_frames import BadFrameStore, configure_application_identity
from rat_tracer.frame_detector import YoloFrameDetector
from rat_tracer.translations import resolve_translations
from rat_tracer.video_file import FrameCapture
from rat_tracer.video_review import ReviewListener, VideoReview

logger = getLogger(__name__)
logger.setLevel(DEBUG)

masker_logger = logger.getChild("VideoMasker")
masker_logger.setLevel(DEBUG)

QML_IMPORT_NAME = "MyBackend"
QML_IMPORT_MAJOR_VERSION = 1


class _FutureJob:
    """A :class:`~rat_tracer.background.Job` backed by a ``Future``."""

    def __init__(self, future: Future):
        self._future = future

    def cancel(self) -> bool:
        return self._future.cancel()


class QtBackgroundExecutor(QObject):
    """The application's :class:`~rat_tracer.background.BackgroundExecutor`.

    One worker thread, so the review's jobs stay in submission order: a mark and
    its retraction must not interleave into a state where the files and the
    index disagree, and a stale detection must not overwrite a fresh one.
    Completions are handed back to the GUI thread through a queued signal, which
    is what lets the review stay a single-threaded object.

    A ``ThreadPoolExecutor`` rather than a ``QThread``: Python joins its workers
    at interpreter exit, whereas a ``QThread`` still parked on its queue there is
    destroyed while running, which Qt treats as fatal and turns into an abort.
    """

    #: Carries a completion from the worker thread to this object's thread.
    #: Emitting is thread-safe, and the auto-connection queues it.
    _completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Created on first use, so an application that only ever watches a video
        # never starts a thread at all.
        self._pool: ThreadPoolExecutor | None = None
        self._completed.connect(self._run_completion)

    def _ensure(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review")
        return self._pool

    def submit[T](
        self,
        work: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> Job:
        def run() -> None:
            try:
                result = work()
            except Exception as error:
                # A failed job must never take the thread down with it: the
                # researcher keeps navigating and the next job still runs.
                logger.exception("background job failed")
                if on_error is not None:
                    self._completed.emit(partial(on_error, error))
                return
            if on_done is not None:
                self._completed.emit(partial(on_done, result))

        return _FutureJob(self._ensure().submit(run))

    @Slot(object)
    def _run_completion(self, completion: Callable[[], None]) -> None:
        completion()

    def stop(self) -> None:
        """Finish what is queued, then join the worker.

        Queued work is drained rather than dropped. A detection nobody will look
        at costs a moment; an abandoned write loses the observation, which is the
        only copy there is.
        """
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)


class CoveragePass:
    """Runs the cumulative pass over one video, on a thread of its own.

    Not on the shared executor: this lasts as long as the video does, and a
    serial executor running it would starve every detection and every write
    behind it. A plain ``Thread`` because nothing in the pass needs Qt -- what it
    runs is one blocking call into the review.
    """

    def __init__(self, review: VideoReview):
        self._review = review
        self._interrupted = Event()
        self._thread = Thread(target=self._run, name="coverage-pass", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        self._review.process_video(self._interrupted.is_set)

    def stop(self) -> None:
        """Ask the pass to save where it got to, and wait until it has."""
        self._interrupted.set()
        if self._thread.is_alive():
            self._thread.join()


@QmlElement
class VideoMasker(QObject):
    """Qt's half of the application: properties, signals, timers and threads.

    Every decision belongs to :class:`~rat_tracer.video_review.VideoReview`,
    which this class drives. What is left here is the part that genuinely needs
    Qt -- exposing state to QML, scheduling a render on the event loop, giving
    the review threads to work on, and turning an image into a ``QVideoFrame``.

    The review reports back through one callback, so the fan-out to Qt's several
    notifications happens in :meth:`_apply_review_changed` and nowhere else.
    """

    # 1. Define a signal to notify QML when the property changes
    position_changed = Signal(float)
    video_changed = Signal(str)
    problem_mode_changed = Signal(bool)
    #: The review pauses itself -- stepping a frame, entering problem reporting
    #: mode -- so the control that shows playback has to follow it rather than
    #: keep its own idea of the answer.
    playing_changed = Signal(bool)
    mark_state_changed = Signal()
    #: Emitted with the frame index once a mark is safely on disk.
    mark_saved = Signal(int)
    #: Emitted with the frame index when a mark could *not* be stored.
    mark_failed = Signal(int)
    #: Internal. The review's own notification, hopped onto this object's
    #: thread: it also arrives from the cumulative pass, and everything below
    #: -- QML property notifies, ``QTimer`` -- belongs to the GUI thread.
    _review_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._review_changed.connect(self._apply_review_changed)
        self._executor = QtBackgroundExecutor(self)
        self._review = VideoReview(
            listener=ReviewListener(
                changed=self._review_changed.emit,
                mark_stored=self.mark_saved.emit,
                mark_failed=self.mark_failed.emit,
            ),
            executor=self._executor,
            detector=YoloFrameDetector(),
            store=BadFrameStore(),
        )
        self._video = None
        self._cap = None
        self._pass: CoveragePass | None = None
        self._video_output = None
        self._video_sink = QVideoSink()
        self._strings = resolve_translations(QLocale.system().name())
        masker_logger.debug("__init__: VideoMasker initialized")

    @Slot()
    def _apply_review_changed(self) -> None:
        """Turn the review's one notification into Qt's several."""
        self._schedule_render()
        self.mark_state_changed.emit()
        self.problem_mode_changed.emit(self._review.problem_mode)
        self.playing_changed.emit(self._review.playing)

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
        self._review.open_video(FrameCaptureAdapter(), self._video)
        self._pass = CoveragePass(self._review)
        self._pass.start()

    video = Property(str, _get_video, _set_video, notify=video_changed)

    @Slot(QUrl)
    def openVideo(self, url: QUrl) -> None:
        """Load a video from a QML file URL (from a file dialog or drag-and-drop)."""
        masker_logger.debug("openVideo: %s", url)
        local_path = url.toLocalFile()
        if local_path:
            self.video = local_path  # type: ignore[assignment]  # PySide Property setter

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
        # There is somewhere to draw now, which may be all that was missing.
        self._schedule_render()

    video_output = Property(QObject, _get_video_output, _set_video_output)

    @Slot()
    def reset(self):
        masker_logger.debug("reset: resetting VideoMasker")
        current_pass = self._pass
        if current_pass is not None:
            current_pass.stop()
        self._pass = None
        self._video = None
        self._cap = None
        # Closing first abandons the outstanding detection, whose answer nobody
        # is going to look at; stopping then finishes the writes queued behind
        # it. Every thread this object started is joined here, not just the
        # pass: reset() is what ``aboutToQuit`` runs, so it is the last chance
        # to finish a write, and a dropped one loses the only copy there is.
        self._review.close_video()
        self._executor.stop()
        self._emit_frame(QVideoFrame())

    def _get_playing(self) -> bool:
        return self._review.playing

    def _set_playing(self, value: bool) -> None:
        masker_logger.debug("_set_playing: %s", value)
        self._review.set_playing(value)

    playing = Property(bool, _get_playing, _set_playing, notify=playing_changed)

    @Property(str, notify=position_changed)
    def time_text(self) -> str:
        return self._review.time_text

    @Property(int, notify=position_changed)
    def frame_index(self) -> int:
        """The displayed frame's index, so a researcher and a technician can
        refer to the same frame unambiguously."""
        return self._review.frame_index

    def _get_position(self) -> float:
        return self._review.position

    def _set_position(self, new_value: float) -> None:
        masker_logger.debug("_set_position: %.3f", new_value)
        self._review.seek(new_value)

    position = Property(float, _get_position, _set_position, notify=position_changed)

    def _schedule_render(self):
        masker_logger.debug("_schedule_render")
        QTimer.singleShot(1, self._rerender_if_needed)

    @Slot()
    def _rerender_if_needed(self):
        outcome = self._review.render_frame()
        masker_logger.debug("_rerender_if_needed: should_emit=%s", outcome.should_emit)
        if not outcome.should_emit:
            return
        frame = (
            bgr_array_to_qvideoframe(outcome.image) if outcome.image is not None else QVideoFrame()
        )
        self._emit_frame(frame)
        self.position_changed.emit(self._review.position)
        # A render is what can make a frame judgeable -- the detection it was
        # waiting for is drawn now -- and the review does not report its own
        # renders, so the control is refreshed here.
        self.mark_state_changed.emit()

    def _get_problem_mode(self) -> bool:
        return self._review.problem_mode

    def _set_problem_mode(self, value: bool) -> None:
        masker_logger.debug("_set_problem_mode: %s", value)
        self._review.set_problem_mode(value)

    problem_mode = Property(bool, _get_problem_mode, _set_problem_mode, notify=problem_mode_changed)

    @Property(bool, notify=mark_state_changed)
    def can_mark(self) -> bool:
        return self._review.can_mark

    @Property(bool, notify=mark_state_changed)
    def frame_marked(self) -> bool:
        return self._review.frame_marked

    @Slot()
    def toggleMark(self) -> None:
        """Store the frame on screen, or withdraw it if it is already stored."""
        self._review.toggle_mark()

    @Slot()
    def undoLastMark(self) -> None:
        self._review.undo()

    @Slot(int)
    def stepFrame(self, delta: int) -> None:
        """Move exactly *delta* frames, pausing playback."""
        self._review.step(delta)


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
