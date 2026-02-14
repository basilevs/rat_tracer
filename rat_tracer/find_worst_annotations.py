from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from gc import collect
from pathlib import Path
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Iterator, TypeVar

from cv2 import (
    FONT_HERSHEY_SIMPLEX,
    IMREAD_UNCHANGED,
    LINE_AA,
    imshow,
    putText,
    waitKey,
    destroyAllWindows,
    imread,
    rectangle,
    line
)
from cv2.typing import MatLike

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from rat_tracer.lib import Annotation, Point, Box, Prediction, annotation_to_box, box_error, box_iou, dashed_rectangle, nms_callback, pop_minimum, best_model_path, truth_for_results


def pop_nearest(boxes: list[Prediction], to_find: Box) -> Prediction | None:
    def distance(box: Prediction):
        result = -box_iou(box.box, to_find)
        # result = distance_squared(box.box.center, to_find.center)
        return result
    return pop_minimum(boxes, distance)


def boxes_errors(truth: list[Prediction], prediction: list[Prediction]) -> Iterator[Error]:
    truth = list(truth)
    prediction = list(prediction)
    while truth or prediction:
        try:
            t = truth.pop()
            closest = pop_nearest(prediction, t.box)
            if closest:
                if box_iou(t.box, closest.box) > 0.0001:
                    local_error = box_error(t, closest)
                    assert local_error >= 0.
                    assert abs(local_error - box_error(closest, t)) < 0.00001
                    yield Error(t.cls, local_error, t.box, closest)
                    continue
                prediction.append(closest)
            yield Error(t.cls, box_error(t, None), t.box, None)
        except IndexError:
            p = prediction.pop()
            yield Error(p.cls, box_error(p, None), None, p)


T = TypeVar("T")
K = TypeVar("K")


def group_by(seq: Iterator[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    result = defaultdict(list)
    for i in seq:
        result[key(i)].append(i)
    return result


def yolo_boxes_to_predictions(boxes: Boxes) -> Iterator[Prediction]:
    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        yield Prediction(int(box.cls.item()), Box(Point(x1, y1), Point(x2, y2)), float(box.conf), None)


@dataclass
class Error:
    cls: int
    error: float
    ground_truth: Box | None
    prediction: Prediction | None

    @property
    def center(self):
        if self.ground_truth:
            return self.ground_truth.center
        if self.prediction:
            return self.prediction.box.center
        return None


def result_errors(results: Results, cls: int) -> Iterator[Error]:
    predictions: Iterator[Prediction] = yolo_boxes_to_predictions(
        results.boxes) if results.boxes else iter([])
    # predictions = (replace(x, confidence = 1.) for x in predictions) # ignore confidence experiment
    predictions_by_cls = group_by(predictions, lambda x: x.cls)
    annotations_by_cls = group_by(truth_for_results(results), lambda x: x.cls)
    height, width = results.orig_shape
    if cls >= 0:
        clss = set([cls])
    else:
        clss = predictions_by_cls.keys() | annotations_by_cls.keys()

    def annotation_to_prediction(annotation: Annotation) -> Prediction:
        return Prediction(annotation.cls, annotation_to_box(annotation, width, height), 1., None)

    def truth_for_cls(cls: int):
        return list(map(annotation_to_prediction, annotations_by_cls[cls]))
    return (b for c in clss for b in boxes_errors(truth_for_cls(c),  predictions_by_cls[c]))


@dataclass
class Datum:
    error: float
    path: Path
    errors: list[Error] = field(repr=False)


def visualize_errors(img: MatLike, errors: list[Error]) -> None:
    for e in errors:
        # ---- ground truth (green) ----
        if e.ground_truth is not None:
            gt = e.ground_truth
            rectangle(
                img,
                (int(gt.tl.x), int(gt.tl.y)),
                (int(gt.br.x), int(gt.br.y)),
                (0, 200, 0),
                2,
            )

        # ---- prediction (red) ----
        if e.prediction is not None:
            pb = e.prediction.box
            dashed_rectangle(
                img,
                (int(pb.tl.x), int(pb.tl.y)),
                (int(pb.br.x), int(pb.br.y)),
                (0, 0, 200),
                2,
                gap_len=5
            )

        # ---- connect centers if both exist ----
        if e.ground_truth and e.prediction:
            c1 = e.ground_truth.center
            c2 = e.prediction.box.center
            line(
                img,
                (int(c1.x), int(c1.y)),
                (int(c2.x), int(c2.y)),
                (255, 0, 0),
                1,
            )

        # ---- annotation text ----
        c = e.center
        if c is None:
            continue

        x, y = int(c.x), int(c.y)

        putText(
            img,
            f"err={e.error:.2f}",
            (x + 4, y - 4),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            LINE_AA,
        )

        if e.prediction is not None:
            putText(
                img,
                f"conf={e.prediction.confidence:.2f}",
                (x + 4, y + 12),
                FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 255),
                1,
                LINE_AA,
            )


def interactive_view(data: list[Datum]):
    idx = 0
    n = len(data)

    while True:
        datum = data[idx]
        img = imread(str(datum.path), IMREAD_UNCHANGED)
        if img is None:
            continue
        visualize_errors(img, datum.errors)
        imshow("Worst predictions", img)

        print(f"[{idx+1}/{n}] {datum.path}  error={data[idx].error:.4f}")

        key = waitKey(0)
        # print('Key detected:', key)
        if key == 27:          # ESC
            break
        if key in (2, 81, 2424832):   # LEFT (Linux / macOS)
            idx = max(0, idx - 1)
        elif key in (3, 83, 2555904):   # RIGHT (Linux / macOS)
            idx = min(n - 1, idx + 1)

    destroyAllWindows()

# ---------- main ----------


def main() -> None:
    root = Path("data/images")
    images = chain(
        root.rglob("*.jpg"),
        root.rglob("*.png"),
        root.rglob("*.jpeg"),
    )
    # images = [Path('data/images/Val/2025-10-10_001027.png')]
    # images = [Path('data/images/Val/2026-01-15-2_002532.png')]
    # images = [Path('data/images/Train/2026-01-27_000006.jpeg')]

    model = YOLO(best_model_path)
    model.add_callback("on_predict_postprocess_end", nms_callback)

    cls = -1  # all classes
    #cls = 0  # rat

    data: list[Datum] = []

    for r in model.predict(list(images), stream=True, verbose=False, workers=0, conf=0.01, deterministic=True):
        errors: list[Error] = list(result_errors(r, cls))
        if not errors:
            continue
        err = max(errors, key=lambda e: e.error)
        d = Datum(err.error, Path(r.path), [err])
        print(d)
        data.append(d)
    del model
    del r
    del d
    collect()

    # worst first
    data.sort(key=lambda d: d.error, reverse=True)
    interactive_view(data)


if __name__ == "__main__":
    main()
