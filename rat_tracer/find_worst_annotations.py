from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from heapq import nlargest
from itertools import chain
from typing import Iterator, TypeVar

from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, imwrite, line, waitKey

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from rat_tracer.lib import Annotation, Point, Box, Prediction, annotation_to_box, box_error, box_iou, nms_callback, pop_minimum, best_model_path, truth_for_results, visualize_gt_vs_pred


def pop_nearest(boxes:list[Prediction], to_find: Box) -> Prediction | None:
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

T = TypeVar("T")
K = TypeVar("K")
def group_by(seq: Iterator[T], key: Callable[[T], K]) -> dict[K, list[T]] :
    result = defaultdict(list)
    for i in seq:
        result[key(i)].append(i)
    return result

def yolo_boxes_to_predictions(boxes: Boxes) -> Iterator[Prediction]:
    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        yield Prediction(int(box.cls.item()), Box(Point(x1, y1), Point(x2, y2)), float(box.conf), None)


def result_error(results: Results, cls: int) -> float:
    predictions_by_cls = group_by(yolo_boxes_to_predictions(results.boxes), lambda x: x.cls)
    annotations_by_cls = group_by(truth_for_results(results), lambda x: x.cls)
    height, width = results.orig_shape
    if cls >= 0:
        clss = set([cls])
    else:
        clss = predictions_by_cls.keys() | annotations_by_cls.keys()
    def annotation_to_prediction(annotation: Annotation) -> Prediction:
        return Prediction(annotation.cls, annotation_to_box(annotation, width, height), 1., None)
    def truth_for_cls(cls:int):
        return list(map(annotation_to_prediction, annotations_by_cls[cls]))
    return sum(boxes_error(truth_for_cls(cls),  predictions_by_cls[cls]) for cls in clss)

@dataclass
class Datum:
    error: float
    path: Path
    def __lt__(self, other:Datum):
        return self.error < other.error
    def __str__(self):
        return f"{self.path}: {self.error}"


def files_to_errors(model: YOLO, files: list[Path], cls: int) -> Iterator[Datum]:
    for result in model.predict(list(files), show=False, stream=True, save_txt=False, save=False, verbose=True):
        path = Path(result.path)
        error: float = result_error(result, cls)
        d = Datum(error, path)
        print(d)
        yield d

def save_gt_vs_pred(
    results: Results,
    cls: int
):
    img = visualize_gt_vs_pred(results, cls)

    name = Path(results.path).with_suffix('').name
    filename = f"{name}.jpg"
    target_dir = Path(results.save_dir) / 'visualization'
    target_dir.mkdir(parents=True, exist_ok=True)
    imwrite(str(target_dir / filename), img)


def visualize(results: Iterator[Results], cls: int):
    for r in results:
        if not r.save_dir:
            r.save_dir = '/tmp/'
        if waitKey(1) == 27:
            break
        save_gt_vs_pred(
            r,
            cls=cls
        )

def main():
    root = Path('data/images')
    images = chain(*[root.rglob(pattern) for pattern in ['*.jpeg', '*.png', '*.jpg']])
    #images = [Path('data/images/Train/2026-01-15-2_000356.png')]
    model = YOLO(best_model_path)
    model.add_callback("on_predict_postprocess_end", nms_callback)
    cls = 3
    predictions = model.predict(
        list(images),
        show=False,
        stream=True,
        save_txt=False,
        save=False,
        verbose=True,
    )
    def result_to_datum(results: Results):
        result = Datum(result_error(results, cls), Path(results.path))
        print(result)
        return result
    worst = nlargest(30, map(result_to_datum, predictions), lambda x: x.error)
    worst.sort(key = lambda x: x.error)
    for i in worst:
        print(i)

    predictions = model.predict(
        list(w.path for w in worst),
        show=True,
        stream=True,
        save=True,
        verbose=False,
    )
    visualize(predictions, cls)

if __name__ == '__main__':
    main()
