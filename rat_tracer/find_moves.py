"""
Finds groups of annotations at the same position and prints a single representative for each.
Multiple instances are not supported (always printed)
"""

from dataclasses import dataclass
from pathlib import Path
from sys import argv
from typing import Iterator, Tuple

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

def labyrinth_coordinates(annotations:list[Annotation]) -> list[float]:
    labs = [a for a in annotations if a.cls == 2]
    if len(labs) != 1:
        raise ValueError("No labyrinth")
    return labs[0].coords


def equal(a: list[float], b:list[float]):
    if len(a) != len(b):
        return False
    result = all(abs(ae - be) < 0.003 for ae, be in zip(a, b))
    return result

def find_moves(cls: int, label_files: Iterator[Path]) -> Iterator[Path]:
    found: list[Tuple[Path, list[float]]] = []
    for path in  label_files:
        class_annotations = list(a for a in read_annotations(path) if a.cls == cls)
        if len(class_annotations) != 1:
            yield path
            continue
        coords = class_annotations[0].coords
        if any(equal(a, coords) for p, a in found):
            continue
        found.append((path, coords))
        yield path

def main():
    l = Path(argv[2]).glob('*.txt')
    l = [p for p in l if not any(3 == a.cls for a in read_annotations(p))]
    l = list(l)
    l.sort()
    l = list(find_moves(int(argv[1]), l))
    for p in l:
        print(p)


if __name__ == "__main__":
    main()
