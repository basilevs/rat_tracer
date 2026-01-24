from __future__ import annotations

from collections.abc import Callable
from os import mkdir
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from heapq import heappushpop, heappush, heappop, nlargest
from itertools import chain, starmap, zip_longest
from shutil import copy2
from os import makedirs
from typing import Callable, Iterator, TypeVar

from ultralytics import YOLO
from ultralytics.engine.results import Results

from rat_tracer.lib import Annotation, read_annotations


T = TypeVar("T")

def intersection_length(min1: float, max1: float, min2: float, max2: float):
    """ length of intersection between intervals 1 and 2"""
    assert(min1 <= max1)
    assert(min2 <= max2)
    if min1 >= max2 or min2 >= max1:
        return 0
    result = min(max1, max2) - max(min1, min2)
    assert(result > 0)
    return result

@dataclass
class Point:
    x: float
    y: float

def middle(a: Point, b: Point) -> Point:
    return Point((a.x + b.x)/2, (a.y+b.y)/2)

def distance_squared(a: Point, b: Point) -> float:
    return (a.x - b.x)**2 + (a.y - b.y)**2

@dataclass
class Box:
    tl: Point
    br: Point
    confidence: float = 1
    def __post_init__(self):
        assert self.tl.x <= self.br.x
        assert self.tl.y <= self.br.y
        assert self.confidence >= 0.
        assert self.confidence <= 1.
        self.area = (self.br.x - self.tl.x) * (self.br.y - self.tl.y)
        assert self.area >= 0.
        self.center = middle(self.tl, self.br)
    def intersection_area(self, other) -> float:
        return intersection_length(self.tl.x, self.br.x, other.tl.x, other.br.x) * intersection_length(self.tl.y, self.br.y, other.tl.y, other.br.y)

EMPTY_BOX = Box(Point(0, 0), Point(0, 0), 0.)
def box_error(truth: Box, prediction: Box) -> float:
    prediction = prediction or EMPTY_BOX
    truth = truth or EMPTY_BOX
    intersection_area = truth.intersection_area(prediction)
    assert intersection_area >= 0
    difference_area = prediction.area + truth.area - intersection_area
    assert difference_area >= 0
    result = (prediction.area - intersection_area) * prediction.confidence + (truth.area - intersection_area) * truth.confidence + intersection_area * abs(truth.confidence - prediction.confidence)
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

def pop_nearest(boxes:list[Box], to_find: Point) -> Box | None:
    def distance(box):
        return distance_squared(box.center, to_find)
    return pop_minimum(boxes, distance)

def boxes_error(truth: list[Box], prediction: list[Box]):
    truth = list(truth)
    prediction = list(prediction)
    error = 0.
    while truth or prediction:
        try:
            t = truth.pop()
            closest = pop_nearest(prediction, t.center)
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
    return Box(Point(a.coords[0] * width, a.coords[1] * height), Point((a.coords[0] + a.coords[2] * 2) * width, (a.coords[1] + a.coords[3] * 2)*height), 1.)

def result_error(results: Results, cls: int) -> float:
    prediction = []
    for box in results.boxes:
        if box.cls != cls:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        prediction.append(Box(Point(x1, y1), Point(x2, y2), float(box.conf)))
    annotations = list(x for x in read_annotations(label_path_from_image(Path(results.path))) if x.cls == cls)
    height, width = results.orig_shape
    truth = [annotation_to_box(b, width, height) for b in annotations]
    return boxes_error(truth, prediction)

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
        if not result.boxes.conf.numel():
            continue
        path = Path(result.path)
        error: float = result_error(result, 0)
        result = Datum(error, path)
        print(result)
        yield result

root = Path('data/images')
images = chain((root / 'Train').glob('*.png'), (root / 'Val').glob('*.png') )
model = YOLO("runs/detect/train20/weights/best.pt")
worst = nlargest(20, files_to_errors(list(images)), lambda x: x.error)
worst.sort(key = lambda x: x.error)
for i in worst:
    print(i)

reannotate(map(lambda x: x.path, worst))
