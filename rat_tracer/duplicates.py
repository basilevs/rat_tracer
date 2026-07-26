from __future__ import annotations

from os import system
from pathlib import Path

from torch import Tensor, float32, tensor
from ultralytics.utils.metrics import bbox_iou

root = Path("data")


def main():
    data: list[(Path, dict[int, Tensor])] = []
    for file in (root / "labels" / "Train").glob("*.txt"):
        parsed: dict[int, Tensor] = {}
        for line in file.read_text().split("\n"):
            if not line:
                continue
            fields = line.split(" ")
            parsed[int(fields[0])] = tensor(tuple(map(float, fields[1:])), dtype=float32)
        data.append((file, parsed))

    while data:
        file, annotation = data.pop()
        for i in data:
            product = 1.0
            class_ids = frozenset(annotation.keys()).union(i[1].keys())
            for k in class_ids:
                try:
                    a = annotation[k]
                    b = i[1][k]
                    kproduct = bbox_iou(a, b, xywh=True)
                except KeyError:
                    product = tensor(0.0)
                    break
                product *= kproduct
            if product.item() > 0.85:
                print(label_to_image_path(file), label_to_image_path(i[0]), product.item())
                system(f"open {label_to_image_path(file)} {label_to_image_path(i[0])}")


def label_to_image_path(path: Path):
    return root / "images" / path.relative_to(root / "labels").with_suffix(".png")


if __name__ == "__main__":
    main()
