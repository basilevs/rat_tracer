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
        fields = line.split(' ')
        yield Annotation(int(fields[0]), map(float, fields[1:]))

def labyrinth_coordinates(annotations:list[Annotation]) -> list[float]:
    labs = [a for a in annotations if a.cls == 2]
    if len(labs != 1):
        raise ValueError("No labyrinth")
    return labs[0].coords

def main():
    input = Path(argv[1])
    assert input.exists()
    annotations = list(read_annotations(input))
    original_lab_coords = labyrinth_coordinates(annotations)
    for path in Path('.').glob('*.txt'):
        if 
    