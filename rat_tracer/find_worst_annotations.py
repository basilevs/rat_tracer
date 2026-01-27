from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass
from heapq import heappop, nlargest
from itertools import chain
from shutil import copy2
from os import makedirs
from typing import Iterator, TypeVar

from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, imwrite, line, waitKey

from ultralytics import YOLO
from ultralytics.engine.results import Results

from rat_tracer.lib import Point, Box, Prediction, annotation_to_box, box_error, box_iou, nms_callback, pop_minimum, best_model_path, truth_for_results, visualize_gt_vs_pred


def pop_nearest(boxes:list[Prediction], to_find: Box) -> Box | None:
    def distance(box:Prediction):
        return -box_iou(box.box, to_find)
    return pop_minimum(boxes, distance)

def boxes_error(truth: list[Prediction], prediction: list[Prediction]):
    truth = list(truth)
    prediction = list(prediction)
    error = 0.
    while truth or prediction:
        try:
            t = truth.pop()
            closest = pop_nearest(prediction, t.box)
            local_error = box_error(t, closest)
            assert local_error >= 0.
            assert abs(local_error - box_error(closest, t)) < 0.00001
            error += local_error
        except IndexError:
            error += box_error(prediction.pop(), None)
    return error

def result_error(results: Results, cls: int) -> float:
    prediction = []
    for box in results.boxes:
        if int(box.cls.item()) != cls:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        prediction.append(Prediction(cls, Box(Point(x1, y1), Point(x2, y2)), float(box.conf), None))
    annotations = list(x for x in truth_for_results(results) if x.cls == cls)
    height, width = results.orig_shape
    truth = [annotation_to_box(b, width, height) for b in annotations]
    return boxes_error(list(Prediction(cls, x, 1., None) for x in truth), prediction)

def reannotate(files: list[Path]):
    model.predict(list(files), show=False, stream=False, save_txt=True, save=True, verbose=True)

def relabel(files: list[Path]):
    target = Path('/tmp/relabel')
    makedirs(target, exist_ok=True)
    while files:
        d = heappop(files)
        print(d)
        copy2(d.path, target)


@dataclass
class Datum:
    error: float
    path: Path
    def __lt__(self, other:Datum):
        return self.error < other.error
    def __str__(self):
        return f"{self.path}: {self.error}"


def files_to_errors(files: list[Path]) -> Iterator[Datum]:
    for result in model.predict(list(files), show=False, stream=True, save_txt=False, save=False, verbose=True, batch=1):
        path = Path(result.path)
        error: float = result_error(result, 0)
        result = Datum(error, path)
        print(result)
        yield result

def save_gt_vs_pred(
    results: Results,
    cls: int
):
    img = visualize_gt_vs_pred(results, cls)

    name = Path(results.path).with_suffix('').name
    filename = f"{name}.jpg"
    target_dir = Path(results.save_dir) / 'visualization'
    target_dir.mkdir(parents=True, exist_ok=True)
    imwrite(target_dir / filename, img)


def visualize(model, worst, cls: int):
    paths = [d.path for d in worst]
    results = model.predict(
        paths,
        show=True,
        stream=True,
        save=True,
        verbose=False,
    )

    for r in results:
        if not r.save_dir:
            r.save_dir = '/tmp/'
        if waitKey(1) == 27:
            break
        save_gt_vs_pred(
            r,
            cls=cls
        )

root = Path('data/images')
images = chain(*[root.rglob(pattern) for pattern in ['*.jpeg', '*.png', '*.jpg']])
#images = [Path('data/images/Train/2026-01-15-2_000356.png')]
model = YOLO(best_model_path)
model.add_callback("on_predict_postprocess_end", nms_callback)
worst = nlargest(30, files_to_errors(list(images)), lambda x: x.error)
worst.sort(key = lambda x: x.error)
for i in worst:
    print(i)

visualize(model, worst, 0)
#reannotate(map(lambda x: x.path, worst))
