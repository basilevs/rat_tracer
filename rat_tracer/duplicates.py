from __future__ import annotations

from os import mkdir, system
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from heapq import heappushpop, heappush, heappop
from itertools import chain
from shutil import copy2
from os import makedirs
from typing import Dict, List

from torch import Tensor, float32
from torch import tensor
from ultralytics.utils.metrics import bbox_iou

from ultralytics import YOLO

root = Path('data')
def main():
    data: List[(Path, Dict[int, Tensor])] = []
    for file in (root / 'labels' / 'Train').glob('*.txt'):
        parsed: Dict[int, Tensor] = dict()
        for line in file.read_text().split('\n'):
            if not line:
                continue
            fields = line.split(' ')
            parsed[int(fields[0])] = tensor(tuple(map(float, fields[1:])), dtype = float32)
        data.append((file, parsed))
    
    while data:
        file, annotation = data.pop()
        for i in data:
            product = 1.
            class_ids = frozenset(annotation.keys()).union(i[1].keys())
            for k in class_ids :
                try:
                    a = annotation[k]
                    b = i[1][k]
                    kproduct = bbox_iou(a, b, xywh = True)
                except KeyError:
                    product = tensor(0.)
                    break
                product *= kproduct
            if product.item() > .85:
                print(label_to_image_path(file), label_to_image_path(i[0]), product.item())
                system(f'open {label_to_image_path(file)} {label_to_image_path(i[0])}')

def label_to_image_path(path: Path):
    return root / 'images' / path.relative_to(root / 'labels').with_suffix('.png')
    
if __name__ == "__main__":
    main()