from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Self, TypeVar

from numpy import ndarray

from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, line

from torch import Tensor, float32, tensor
from ultralytics.engine.results import Results, Boxes
from ultralytics.engine.predictor import BasePredictor

best_model_path=Path('runs/detect/train33/weights/last.pt')
#best_model_path=Path('input/yolo26n.pt')

@dataclass
class Point:
    x: float
    y: float
    def moved(self, x:float, y:float) -> Self:
        return Point(self.x + x, self.y + y)

def middle(a: Point, b: Point) -> Point:
    return Point((a.x + b.x)/2, (a.y+b.y)/2)

def distance_squared(a: Point, b: Point) -> float:
    return (a.x - b.x)**2 + (a.y - b.y)**2

def intersection_length(min1: float, max1: float, min2: float, max2: float):
    """ length of intersection between intervals 1 and 2"""
    assert min1 <= max1
    assert min2 <= max2
    if min1 >= max2 or min2 >= max1:
        return 0
    result = min(max1, max2) - max(min1, min2)
    assert result >= 0
    return result

def _near_range(begin: float, end: float, target:float, margin: float) -> bool:
    return begin - margin < target < end + margin

@dataclass
class Box:
    tl: Point
    br: Point
    def __post_init__(self):
        assert self.tl.x <= self.br.x
        assert self.tl.y <= self.br.y
        self.area = (self.br.x - self.tl.x) * (self.br.y - self.tl.y)
        assert self.area >= 0.
        self.center = middle(self.tl, self.br)
    def intersection_area(self, other) -> float:
        return intersection_length(self.tl.x, self.br.x, other.tl.x, other.br.x) * intersection_length(self.tl.y, self.br.y, other.tl.y, other.br.y)
    def expanded(self, margin:float) -> Self:
        return Box(self.tl.moved(-margin, -margin), self.br.moved(margin, margin))
    def near(self, other:Self, margin:float) -> bool:
        assert margin >= 0
        return self.expanded(margin).intersection_area(other) > 0

def box_iou(a: Box, b: Box) -> float:
    inter = a.intersection_area(b)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    assert union > 0
    return inter / union

@dataclass
class Annotation:
    cls: int
    coords: list[float]

def read_annotations(path: Path) -> Iterator[Annotation]:
    for line in path.read_text().split('\n'):
        if not line:
            continue
        fields = line.split(' ')
        yield Annotation(int(fields[0]), list(map(float, fields[1:])))

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

def label_path_from_image(image: Path) -> Path:
    annotations = image.parent / 'annotations'
    if annotations.is_dir():
        return annotations / image.with_suffix('.txt').name
    else:
        try:
            root:Path = next(x for x in image.absolute().parents if x.name == 'images')
        except StopIteration as exc:
            raise ValueError(image) from exc
        relative = image.absolute().relative_to(root).with_suffix('.txt')
        return root.parent / 'labels' / relative


@dataclass
class Prediction:
    cls: int
    box: Box
    confidence: float
    track: int | None


class Predictions:
    def __init__(self, results: Results):
        self._predictions: list[Prediction] = []
        boxes = results.boxes
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            self._predictions.append(
                Prediction(
                    cls=cls,
                    box=Box(
                        Point(float(boxes.xyxy[i, 0]), float(boxes.xyxy[i, 1])),
                        Point(float(boxes.xyxy[i, 2]), float(boxes.xyxy[i, 3])),
                    ),
                    confidence=float(boxes.conf[i]),
                    track=int(boxes.id[i]) if boxes.id is not None else None
                )
            )

    def by_track(self, track_id: int) -> Prediction:
        return next(p for p in self._predictions if p.track == track_id)
    
    def by_class(self, cls: int) -> list[Prediction]:
        return [p for p in self._predictions if p.cls == cls]
    
EMPTY_PREDICTION = Prediction(0, Box(Point(0, 0), Point(0, 0)), 0., None)
def box_error(truth: Prediction | None, prediction: Prediction | None) -> float:
    prediction = prediction or EMPTY_PREDICTION
    truth = truth or EMPTY_PREDICTION
    intersection_area = truth.box.intersection_area(prediction.box)
    assert intersection_area >= 0
    total_area = prediction.box.area + truth.box.area - intersection_area
    assert total_area >= 0
    result = (prediction.box.area - intersection_area) * prediction.confidence + (truth.box.area - intersection_area) * truth.confidence + intersection_area * abs(truth.confidence - prediction.confidence)
    result /= total_area
    assert result >= 0.
    return result

T = TypeVar("T")

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

def nms(
    predictions: list[Prediction],
    iou_threshold: float = 0.5,
) -> list[Prediction]:
    if not predictions:
        return []

    # sort by confidence descending
    preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    kept: list[Prediction] = []

    while preds:
        best = preds.pop(0)
        kept.append(best)

        preds = [
            p for p in preds
            if box_iou(best.box, p.box) < iou_threshold
        ]

    return kept


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

def visualize_gt_vs_pred(results: Results, cls:int) -> ndarray:
    img = results.orig_img.copy()
    h, w = results.orig_shape

    # ---- Draw GT (green) ----
    for ann in truth_for_results(results):
        if cls >= 0 and ann.cls != cls:
            continue
        box: Box = annotation_to_box(ann, w, h)
        rectangle(img, (int(box.tl.x), int(box.tl.y)), (int(box.br.x), int(box.br.y)), (0, 200, 0), 2)

    putText(
        img,
        results.path,
        (0, h-4),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 0, 0),
        1,
        LINE_AA,
    )

    # ---- Draw predictions (red, dashed-ish) ----
    annotated_boxes: list[Box] = []
    for box in results.boxes:
        if cls >= 0 and int(box.cls.item()) != cls:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)

        dashed_rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 2, gap_len=6)
        box = Box(Point(x1, y1), Point(x2, y2))
        offset = 40 * [box_iou(box, b) > 0.2 for b in annotated_boxes].count(True)
        annotated_boxes.append(box)
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
    return img

def nms_callback(predictor: BasePredictor):

    for r in predictor.results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes:Boxes = r.boxes
        device = boxes.xyxy.device

        # ---- collect predictions per class ----
        by_class: dict[int, list[Prediction]] = defaultdict(list)

        predictions = Predictions(r)

        for i in predictions._predictions:
            by_class[i.cls].append(i)

        # ---- apply NMS per class ----
        kept: list[Prediction] = []
        for _, preds in by_class.items():
            kept.extend(nms(preds, iou_threshold=0.25))

        if not kept:
            r.boxes = None
            continue

        r.boxes = Boxes(build_boxes_tensor(kept, device), r.orig_shape)

def build_boxes_tensor(
    preds: list[Prediction],
    device,
) -> Tensor:
    has_id = preds and preds[0].track is not None
    dtype = float32
    if has_id:
        return tensor(
            [
                [
                    p.box.tl.x,
                    p.box.tl.y,
                    p.box.br.x,
                    p.box.br.y,
                    p.track,
                    p.confidence,
                    p.cls,
                ]
                for p in preds
            ],
            device=device,
            dtype=dtype,
        )
    else:
        return tensor(
            [
                [
                    p.box.tl.x,
                    p.box.tl.y,
                    p.box.br.x,
                    p.box.br.y,
                    p.confidence,
                    p.cls,
                ]
                for p in preds
            ],
            device=device,
            dtype=dtype,
        )
