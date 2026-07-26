"""
Finds groups of annotations at the same position and prints a single representative for each.
Multiple instances are not supported (always printed)
"""

from collections.abc import Iterator
from pathlib import Path
from sys import argv

from rat_tracer.lib import Annotation, read_annotations


def labyrinth_coordinates(annotations: list[Annotation]) -> list[float]:
    labs = [a for a in annotations if a.cls == 2]
    if len(labs) != 1:
        raise ValueError("No labyrinth")
    return labs[0].coords


def equal(a: list[float], b: list[float]):
    if len(a) != len(b):
        return False
    result = all(abs(ae - be) < 0.003 for ae, be in zip(a, b, strict=True))
    return result


def find_moves(cls: int, label_files: Iterator[Path]) -> Iterator[Path]:
    found: list[tuple[Path, list[float]]] = []
    for path in label_files:
        class_annotations = [a for a in read_annotations(path) if a.cls == cls]
        if len(class_annotations) != 1:
            yield path
            continue
        coords = class_annotations[0].coords
        if any(equal(a, coords) for p, a in found):
            continue
        found.append((path, coords))
        yield path


def main():
    paths = Path(argv[2]).glob("*.txt")
    paths = [p for p in paths if not any(a.cls == 3 for a in read_annotations(p))]
    paths = list(paths)
    paths.sort()
    paths = list(find_moves(int(argv[1]), paths))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
