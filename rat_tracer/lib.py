from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Self

best_model_path=Path('runs/detect/train25/weights/best.pt')

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
