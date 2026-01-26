from platform import system
from pathlib import Path

from numpy import ones, zeros, uint8
from numpy import ndarray

from cv2 import (
    MORPH_ELLIPSE,
    VideoCapture,
    VideoWriter,
    VideoWriter_fourcc,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    createBackgroundSubtractorMOG2,
    cvtColor,
    COLOR_BGR2GRAY,
    getStructuringElement,
    morphologyEx,
    MORPH_OPEN,
    imshow,
    waitKey,
)

from ultralytics import YOLO
from rat_tracer.lib import best_model_path

RAT_CLASS = 0
ALPHA = 0.35
MACOS, LINUX, WINDOWS = (system() == x for x in ["Darwin", "Linux", "Windows"])

def main(input_video: Path, output_video: Path):
    model = YOLO(best_model_path)

    if not output_video:
        raise ValueError('Output argument is missing')
    if output_video.is_dir():
        output_video = output_video / input_video.with_suffix('').name
    if output_video.with_suffix('') == input_video.with_suffix(''):
        output_video = input_video.parent / (input_video.with_suffix('').name + '_painted')

    cap = VideoCapture(str(input_video))
    try:
        width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(CAP_PROP_FPS)
    finally:
        cap.release()

    suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")

    output_video = output_video.with_suffix(suffix)
    writer = VideoWriter(
        str(output_video.with_suffix(suffix)),
        VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError("Can't write to " + str(output_video))

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
        show=False,
    )

    red = zeros((height, width, 3), dtype=uint8)
    red[:, :, 2] = 255

    open_kernel = getStructuringElement(MORPH_ELLIPSE,(5,5))
    for frame_idx, results in enumerate(results_stream):
        img = results.orig_img

        fg = mog.apply(img)
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
                visited[y1:y2, x1:x2][roi > 0] = 255

        mask = visited.astype(bool)
        img[mask] = (
            img[mask] * (1 - ALPHA) +
            red[mask] * ALPHA
        )

        writer.write(img)

        # ---- streaming debug preview ----
        imshow("MOG foreground (ROI only)", img)
        if waitKey(1) == 27:  # ESC
            break


        if frame_idx % 100 == 0:
            print("\rFrame", frame_idx, end='')

    writer.release()
    print("Saved to " + str(output_video))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: paint_rat_coverage_track.py input.mp4 output.mp4")

    main(Path(sys.argv[1]), Path(sys.argv[2]))
