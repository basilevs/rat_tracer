from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass
from heapq import heappop, nlargest
from itertools import chain
from shutil import copy2
from os import makedirs
from typing import Iterator, TypeVar

from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, imwrite, line

from ultralytics import YOLO
from ultralytics.engine.results import Results

from rat_tracer.lib import Annotation, Point, Box, read_annotations


T = TypeVar("T")

@dataclass
class Prediction:
    box: Box
    confidence: float

EMPTY_PREDICTION = Prediction(Box(Point(0, 0), Point(0, 0)), 0.)
def box_error(truth: Prediction, prediction: Prediction) -> float:
    prediction = prediction or EMPTY_PREDICTION
    truth = truth or EMPTY_PREDICTION
    intersection_area = truth.box.intersection_area(prediction.box)
    assert intersection_area >= 0
    difference_area = prediction.box.area + truth.box.area - intersection_area
    assert difference_area >= 0
    result = (prediction.box.area - intersection_area) * prediction.confidence + (truth.box.area - intersection_area) * truth.confidence + intersection_area * abs(truth.confidence - prediction.confidence)
    assert result >= 0.
    return result

def pop_minimum(items: list[T], key: Callable[[T], float]) -> T | None:
    """Find, remove and return the element with minimal key(input[i])."""
    if not items:
        return None

    min_idx = 0
    min_val = key(items[0])

    for i in range(1, len(items)):
        v = key(items[i])
        if v < min_val:
            min_val = v
            min_idx = i

    return items.pop(min_idx)

def box_iou(a: Box, b: Box) -> float:
    inter = a.intersection_area(b)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    assert union > 0
    return inter / union

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

def label_path_from_image(image: Path) -> Path:
    root:Path = image
    while root.name != 'images':
        root = root.parent
    relative = image.relative_to(root).with_suffix('.txt')
    return root.parent / 'labels' / relative

def annotation_to_box(a: Annotation, width: float, height: float) -> Box:
    cx, cy, w, h = a.coords
    half_w = w * width / 2
    half_h = h * height / 2
    return Box(
        Point(cx * width - half_w, cy * height - half_h),
        Point(cx * width + half_w, cy * height + half_h)
    )

def truth_for_results(results: Results) -> Iterator[Annotation]:
    return read_annotations(label_path_from_image(Path(results.path)))
                     
def result_error(results: Results, cls: int) -> float:
    prediction = []
    for box in results.boxes:
        if int(box.cls.item()) != cls:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        prediction.append(Prediction(Box(Point(x1, y1), Point(x2, y2)), float(box.conf)))
    annotations = list(x for x in truth_for_results(results) if x.cls == cls)
    height, width = results.orig_shape
    truth = [annotation_to_box(b, width, height) for b in annotations]
    return boxes_error(list(Prediction(x, 1.) for x in truth), prediction)

def reannotate(files: list[Path]):
    model.predict(list(files), show=True, stream=False, save_txt=True, save=True, verbose=True)

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
    for result in model.predict(list(files), show=False, stream=True, save_txt=False, save=False, verbose=True):
        path = Path(result.path)
        error: float = result_error(result, 0)
        result = Datum(error, path)
        print(result)
        yield result

def dashed_line(img, p1, p2, color, thickness=1, dash_len=6, gap_len=4):
    x1, y1 = p1
    x2, y2 = p2
    length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    if length == 0:
        return

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    dist = 0.0
    draw = True
    while dist < length:
        seg_len = dash_len if draw else gap_len
        nx = x1 + dx * min(dist + seg_len, length)
        ny = y1 + dy * min(dist + seg_len, length)
        if draw:
            line(
                img,
                (int(x1 + dx * dist), int(y1 + dy * dist)),
                (int(nx), int(ny)),
                color,
                thickness,
            )
        dist += seg_len
        draw = not draw

def dashed_rectangle(img, tl, br, color, thickness=1, gap_len=4):
    x1, y1 = tl
    x2, y2 = br
    dashed_line(img, (x1, y1), (x2, y1), color, thickness, gap_len=gap_len)
    dashed_line(img, (x2, y1), (x2, y2), color, thickness, gap_len=gap_len)
    dashed_line(img, (x2, y2), (x1, y2), color, thickness, gap_len=gap_len)
    dashed_line(img, (x1, y2), (x1, y1), color, thickness, gap_len=gap_len)


def visualize_gt_vs_pred(
    results: Results,
    cls: int
):
    img = results.orig_img.copy()
    h, w = results.orig_shape

    # ---- Draw GT (green) ----
    for ann in truth_for_results(results):
        if ann.cls != cls:
            continue
        box: Box = annotation_to_box(ann, w, h)
        rectangle(img, (int(box.tl.x), int(box.tl.y)), (int(box.br.x), int(box.br.y)), (0, 200, 0), 2)

    # ---- Draw predictions (red, dashed-ish) ----
    offset = 0
    for box in results.boxes:
        if int(box.cls.item()) != cls:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)

        dashed_rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 2, gap_len=6)
        putText(
            img,
            f"{conf:.2f}",
            (x1 + offset, y1 - 4),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1,
            LINE_AA,
        )
        offset += 40

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
        save=False,
        verbose=False,
    )

    for r in results:
        if not r.save_dir:
            r.save_dir = '/tmp/'
        visualize_gt_vs_pred(
            r,
            cls=cls
        )

root = Path('data/images')
images = chain((root / 'Train').glob('*.png'), (root / 'Val').glob('*.png') )
#images = [Path('data/images/Train/2026-01-15-2_000356.png')]
model = YOLO("runs/detect/train24/weights/best.pt")
worst = nlargest(30, files_to_errors(list(images)), lambda x: x.error)
worst.sort(key = lambda x: x.error)
for i in worst:
    print(i)

visualize(model, worst, 0)
#reannotate(map(lambda x: x.path, worst))
