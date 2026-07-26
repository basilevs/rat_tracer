from collections.abc import Generator
from logging import getLogger
from pathlib import Path
from platform import system
from queue import Empty, Queue, ShutDown
from threading import Thread

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    VideoCapture,
    VideoWriter,
    VideoWriter_fourcc,
    createBackgroundSubtractorMOG2,
    getStructuringElement,
    imshow,
    morphologyEx,
    waitKey,
)
from numpy import add, bool_, dtype, multiply, ndarray, zeros
from torch.accelerator import current_accelerator
from ultralytics import YOLO

from rat_tracer.lib import model_path

RAT_CLASS = 0
ALPHA = 0.35
MACOS, LINUX, WINDOWS = (system() == x for x in ["Darwin", "Linux", "Windows"])

logger = getLogger(__name__)

type MaskFrame = ndarray[tuple[int, int], dtype[bool_]]


def presence_frames(input_video: Path, model: YOLO) -> Generator[tuple[ndarray, MaskFrame]]:
    cap = VideoCapture(str(input_video))
    try:
        width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    # frame_batches = chunk(video_frames(input_video), 1)
    frame_batches = generate_in_thread(video_frames(input_video), 15)
    mog = createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False,
    )

    open_kernel = getStructuringElement(MORPH_ELLIPSE, (5, 5))
    for batch in frame_batches:
        logger.debug("Processing batch of size %d", len(batch))
        results_batch = model.predict(
            source=batch,
            batch=len(batch),
            conf=0.25,
            stream=True,
            verbose=False,
            show=False,
            device=current_accelerator(),
        )

        for results in results_batch:
            visited: MaskFrame = zeros((height, width), dtype=bool)
            img = results.orig_img
            fg = mog.apply(img)
            visited[:] = False
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls.item()) != RAT_CLASS:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi = fg[y1:y2, x1:x2]

                    roi = morphologyEx(
                        roi,
                        MORPH_OPEN,
                        open_kernel,
                    )
                    visited[y1:y2, x1:x2][roi > 0] = True
            yield (img, visited)


def video_frames(input_video: Path) -> Generator[ndarray]:
    cap = VideoCapture(str(input_video))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def generate_in_thread[T](generator: Generator[T], maxsize: int = 100) -> Generator[list[T]]:
    q: Queue = Queue(maxsize=maxsize)
    error = None

    def worker():
        nonlocal error
        try:
            for item in generator:
                q.put(item)
        except ShutDown:
            pass
        except Exception as e:
            error = e
        finally:
            q.shutdown()

    buffer = []
    try:
        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            buffer.append(q.get(block=True))
            try:
                while True:
                    buffer.append(q.get(block=False))
            except Empty:
                if buffer:
                    yield buffer[:]
                    buffer = []
    except ShutDown:
        if buffer:
            yield buffer
    finally:
        q.shutdown()
        thread.join()
        if error:
            raise error


def apply_red_mask(img: ndarray, mask: MaskFrame):
    img[mask.astype(bool)] = multiply(img[mask.astype(bool)], 1.0 - ALPHA, casting="unsafe")
    img[mask.astype(bool)] = add(img[mask.astype(bool)], [0, 0, int(255 * ALPHA)])


def main(input_video: Path, output_video: Path):
    model = YOLO(model_path())
    cap = VideoCapture(str(input_video))
    try:
        width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(CAP_PROP_FPS)
    finally:
        cap.release()

    if not output_video:
        raise ValueError("Output argument is missing")
    if output_video.is_dir():
        output_video = output_video / input_video.with_suffix("").name
    if output_video.with_suffix("") == input_video.with_suffix(""):
        output_video = input_video.parent / (input_video.with_suffix("").name + "_painted")

    suffix, fourcc = (
        (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
    )

    output_video = output_video.with_suffix(suffix)
    writer = VideoWriter(
        str(output_video.with_suffix(suffix)),
        VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError("Can't write to " + str(output_video))

    visited: MaskFrame = zeros((height, width), dtype=bool)

    for frame_idx, (img, mask) in enumerate(presence_frames(input_video, model)):
        visited |= mask
        mask = visited.astype(bool)
        apply_red_mask(img, mask)

        writer.write(img)

        # ---- streaming debug preview ----
        imshow("MOG foreground (ROI only)", img)
        if waitKey(1) == 27:  # ESC
            break

        if frame_idx % 100 == 0:
            print("\rFrame", frame_idx, end="")

    writer.release()
    print("Saved to " + str(output_video))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: paint_rat_coverage_track.py input.mp4 output.mp4")

    main(Path(sys.argv[1]), Path(sys.argv[2]))
