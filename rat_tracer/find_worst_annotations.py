from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field, replace
from itertools import chain
from statistics import fmean
from typing import Iterator, TypeVar

from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA, imshow, putText, waitKey, destroyAllWindows

from ultralytics import YOLO
from ultralytics.engine.results import Results

from rat_tracer.lib import Annotation, Point, Box, Prediction, annotation_to_box, box_error, box_iou, nms_callback, pop_minimum, best_model_path, truth_for_results, visualize_gt_vs_pred


def pop_nearest(boxes:list[Prediction], to_find: Box) -> Prediction | None:
    def distance(box:Prediction):
        return -box_iou(box.box, to_find)
    return pop_minimum(boxes, distance)

def boxes_errors(truth: list[Prediction], prediction: list[Prediction]) -> Iterator[Prediction]:
    truth = list(truth)
    prediction = list(prediction)
    while truth or prediction:
        try:
            t = truth.pop()
            closest = pop_nearest(prediction, t.box)
            local_error = box_error(t, closest)
            assert local_error >= 0.
            assert abs(local_error - box_error(closest, t)) < 0.00001
            yield replace(t, confidence=local_error)
        except IndexError:
            p = prediction.pop()
            yield  replace(p, confidence=box_error(p, None))

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


def result_errors(results: Results, cls: int) -> Iterator[Prediction]:
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
    return (b for c in clss for b in boxes_errors(truth_for_cls(c),  predictions_by_cls[c]) )

@dataclass
class Datum:
    error: float
    path: Path
    errors: list[Prediction] = field(repr=False)

def interactive_view(model: YOLO, data: list[Datum], cls: int):
    idx = 0
    n = len(data)

    while True:
        path = data[idx].path
        results = next(model.predict([str(path)], stream=True, verbose=False))

        img = visualize_gt_vs_pred(results, cls)
        for p in result_errors(results, cls):
            putText(
                img,
                f"{p.confidence:.2f}",
                (int(p.box.center.x), int(p.box.center.y)),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                LINE_AA,
            )
        imshow("Worst predictions", img)

        print(f"[{idx+1}/{n}] {path}  error={data[idx].error:.4f}")

        key = waitKey(0)
        # print('Key detected:', key)
        if key == 27:          # ESC
            break
        elif key in (2, 81, 2424832):   # LEFT (Linux / macOS)
            idx = max(0, idx - 1)
        elif key in (3, 83, 2555904):   # RIGHT (Linux / macOS)
            idx = min(n - 1, idx + 1)

    destroyAllWindows()

# ---------- main ----------

def main():
    root = Path("data/images")
    images = list(chain(
        root.rglob("*.jpg"),
        root.rglob("*.png"),
        root.rglob("*.jpeg"),
    ))

    model = YOLO(best_model_path)
    model.add_callback("on_predict_postprocess_end", nms_callback)

    cls = -1  # all classes

    data: list[Datum] = []

    for r in model.predict(images, stream=True, verbose=False):
        errors = list(result_errors(r, cls))
        err = fmean(p.confidence for p in errors) if errors else 0.
        d = Datum(err, Path(r.path), errors)
        print(d)
        data.append(d)

    # worst first
    data.sort(key=lambda d: d.error, reverse=True)

    interactive_view(model, data, cls)

if __name__ == "__main__":
    main()
