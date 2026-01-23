from sys import argv, stdout
from os import SEEK_END
from dataclasses import dataclass
from pathlib import Path
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


def spread_annotations(annotations: list[Annotation], target_cls: int, files: Iterator[Path]):
    annotations = list(annotations)
    original_lab_coords = labyrinth_coordinates(annotations)
    annotations_to_add = [a for a in list(annotations) if a.cls == target_cls]
    if not annotations_to_add:
        raise ValueError("No input annotations from class " + str(target_cls))

    for path in files:
        existing_annotations = list(read_annotations(path))
        if any(a.cls == target_cls for a in existing_annotations):
            continue
        if not equal(labyrinth_coordinates(existing_annotations), original_lab_coords):
            continue
        print("  ", path)
        ends_with_eol = False
        with open(path, 'rb') as f:
            f.seek(-1, SEEK_END)
            b = f.read()
            ends_with_eol = b == b'\n'

        with open(path, 'a', encoding='utf-8') as f:
            o = f
            if not ends_with_eol:
                print(file = o)
            for i in annotations_to_add:
                print(i.cls, *i.coords, file = o)


def main():
    target_dir = Path(argv[1])
    assert target_dir.is_dir()
    target_cls = 3

    origin = []
    to = []
    for path in  target_dir.glob('*.txt'):
        existing_annotations = list(read_annotations(path))
        if any(a.cls == target_cls for a in existing_annotations):
            origin.append(existing_annotations)
            continue
        to.append(path)
    
    for o in origin:
        spread_annotations(o, target_cls, to)

if __name__ == "__main__":
    main()