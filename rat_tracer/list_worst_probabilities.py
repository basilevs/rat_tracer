from __future__ import annotations

from os import mkdir
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from heapq import heappushpop, heappush, heappop
from itertools import chain
from shutil import copy2
from os import makedirs

from ultralytics import YOLO

model = YOLO("/Users/vasiligulevich/git/rat_tracer/runs/detect/train4/weights/best.pt")

@dataclass
class Datum:
    confidence: float
    path: Path
    def __lt__(self, other:Datum):
        # inverted to use for maximum detection in Python 3.12 minheap
        return self.confidence > other.confidence
    def __str__(self):
        return f"{self.path}: {self.confidence}"

worst: list[Datum] = []

root = Path('data/images')
files = chain((root / 'Train').glob('*.png'), (root / 'Val').glob('*.png') )
for result in model.predict(list(files), show=False, stream=True, save_txt=True, save=True, verbose=True, batch=100):
    if not result.boxes.conf.numel():
        continue
    conf: float = min(result.boxes.conf)
    datum = Datum(conf, Path(result.path))
    if len(worst) < 20:
        heappush(worst, datum)
    else:
        heappushpop(worst, datum)

target = Path('/tmp/relabel')
makedirs(target, exist_ok=True)
while worst:
    d = heappop(worst)
    print(d)
    copy2(d.path, target)
