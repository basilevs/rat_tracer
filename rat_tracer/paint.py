from platform import system
from pathlib import Path
import numpy as np
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, circle
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

    suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")

    writer = VideoWriter(
        str(output_video.with_suffix(suffix)),
        VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )

    visited = np.zeros((height, width), dtype=np.uint8)
    radius = max(1, int(0.01 * width))

    # Use tracker; persist=True keeps internal state
    results_stream = model.track(
        source=str(input_video),
        conf=0.25,
        persist=True,
        stream=True,
        verbose=False,
        show=True,
    )

    frame_idx = 0
    red = np.zeros((height, width, 3), np.uint8)
    red[:, :, 2] = 255

    for results in results_stream:
        img = results.orig_img
        h, w = results.orig_shape

        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls.item()) != RAT_CLASS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                circle(
                    visited,
                    (cx, cy),
                    radius,
                    255,
                    thickness=-1,
                )

        mask_bool = visited.astype(bool)
        img[mask_bool] = (
            img[mask_bool] * (1 - ALPHA) +
            red[mask_bool] * ALPHA
        ).astype(np.uint8)

        writer.write(img)
        frame_idx += 1
        if not frame_idx % 100:
            print('Frame', frame_idx)

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
