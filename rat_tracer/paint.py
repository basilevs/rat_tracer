from platform import system
from pathlib import Path

from numpy import zeros, uint8
from numpy import ndarray

from cv2 import (
    VideoCapture,
    VideoWriter,
    VideoWriter_fourcc,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    createBackgroundSubtractorMOG2,
    cvtColor,
    COLOR_BGR2GRAY,
    morphologyEx,
    MORPH_OPEN,
)

from ultralytics import YOLO
from rat_tracer.lib import best_model_path

RAT_CLASS = 0
ALPHA = 0.35
MACOS, LINUX, WINDOWS = (system() == x for x in ["Darwin", "Linux", "Windows"])


def main(
    input_video: Path,
    output_video: Path,
):
    model = YOLO(best_model_path)

    cap = VideoCapture(str(input_video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_video}")
        width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(CAP_PROP_FPS)
    finally:
        cap.release()

    suffix, fourcc = (
        (".mp4", "avc1") if MACOS
        else (".avi", "WMV2") if WINDOWS
        else (".avi", "MJPG")
    )

    writer = VideoWriter(
        str(output_video.with_suffix(suffix)),
        VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )

    visited: ndarray = zeros((height, width), dtype=uint8)

    mog = createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False,
    )

    results_stream = model.track(
        source=str(input_video),
        conf=0.25,
        persist=True,
        stream=True,
        verbose=False,
        show=True,
    )

    red = zeros((height, width, 3), dtype=uint8)
    red[:, :, 2] = 255

    frame_idx = 0

    for results in results_stream:
        img = results.orig_img
        gray = cvtColor(img, COLOR_BGR2GRAY)
        h, w = results.orig_shape

        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls.item()) != RAT_CLASS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = gray[y1:y2, x1:x2]

                fg = mog.apply(roi)

                fg = morphologyEx(
                    fg,
                    MORPH_OPEN,
                    zeros((3, 3), dtype=uint8),
                )

                visited[y1:y2, x1:x2][fg > 0] = 255

        mask = visited.astype(bool)
        img[mask] = (
            img[mask] * (1 - ALPHA) +
            red[mask] * ALPHA
        ).astype(uint8)

        writer.write(img)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print("Frame", frame_idx)

    writer.release()
    print(f"Processed {frame_idx} frames")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: paint_rat_coverage_track.py input.mp4 output.mp4")
        raise SystemExit(1)

    main(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
    )
