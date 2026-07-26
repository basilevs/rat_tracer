from collections.abc import Iterator
from pathlib import Path
from sys import argv

from cv2 import imshow, waitKey
from ultralytics import YOLO

from rat_tracer.lib import model_path, nms_callback, visualize_gt_vs_pred

label_map = {  # Define the label map with all annotated class labels.
    0: "rat",
    1: "human",
    2: "labyrinth",
    3: "pipe_port",
}
model = YOLO(model_path())
model.add_callback("on_predict_postprocess_end", nms_callback)


def visualize(images: Iterator[Path], cls: int):
    paths = list(images)
    for i in paths:
        assert i.is_file()
    results = model.predict(
        list(paths),
        show=False,
        stream=True,
        save=False,
        verbose=True,
        conf=0.01,
    )

    for r in results:
        img = visualize_gt_vs_pred(r, cls)
        imshow("Ground truth and prediction", img)
        print(r.path)
        while True:
            key = waitKey(100)
            if key == 32:  # Space
                break
            if key == 27:  # Esc
                return


def main():
    visualize(map(Path, argv[1:]), 0)


if __name__ == "__main__":
    main()
