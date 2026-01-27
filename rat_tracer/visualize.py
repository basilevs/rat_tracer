from pathlib import Path
from sys import argv
from typing import Iterator
from cv2 import imshow, waitKey
from ultralytics import YOLO
from ultralytics.data.utils import visualize_image_annotations

from lib import label_path_from_image, best_model_path, visualize_gt_vs_pred

label_map = {  # Define the label map with all annotated class labels.
    0: "rat",
    1: "human",
    2: "labyrinth",
    3: "pipe_port",
}
model = YOLO(best_model_path)

def visualize(images:Iterator[Path], cls: int):
    l = list(images)
    for i in l:
        assert i.is_file()
    results = model.predict(
        list(l),
        show=False,
        stream=True,
        save=False,
        verbose=True,
    )

    for r in results:
        img = visualize_gt_vs_pred(r, cls)
        imshow("Ground truth and prediction", img)
        print(r.path)
        while True:
            key = waitKey(100)
            if key == 32: #Space
                break
            if key == 27: #Esc
                return


def main():
    visualize(map(Path, argv[1:]), 0)

if __name__ == '__main__':
    main()
