from sys import argv
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
    result = all(abs(ae - be) < 0.0001 for ae, be in zip(a, b))
    return result

def main():
    input = Path(argv[2])
    target_dir = Path(argv[3])
    assert input.exists()
    assert target_dir.is_dir()
    annotations = list(read_annotations(input))
    original_lab_coords = labyrinth_coordinates(annotations)
    target_cls = 3
    annotations_to_add = [a for a in annotations if a.cls == target_cls]

    for path in  target_dir.glob('*.txt'):
        existing_annotations = list(read_annotations(path))
        if any(a.cls == target_cls for a in existing_annotations):
            continue
        if not equal(labyrinth_coordinates(existing_annotations), original_lab_coords):
            continue
        print(path)

if __name__ == "__main__":
    main()