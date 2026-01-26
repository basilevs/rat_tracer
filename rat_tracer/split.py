import argparse
import random
import shutil
from pathlib import Path


def split_dataset(
    gt_dir: Path,
    out_dir: Path,
    val_ratio: float,
    seed: int = 42,
):
    images_src = gt_dir / "images"
    labels_src = gt_dir / "labels"

    if not labels_src.is_dir():
        labels_src = images_src / "annotations"

    assert labels_src.is_dir()

    images = sorted(images_src.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not images:
        raise RuntimeError("No images found")

    random.seed(seed)
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = set(images[:val_count])
    train_images = images[val_count:]

    train_images_dir = out_dir / "images" / "Train"
    val_images_dir = out_dir / "images" / "Val"
    train_labels_dir = out_dir / "labels" / "Train"
    val_labels_dir = out_dir / "labels" / "Val"

    for p in (
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    def copy_pair(img_path: Path, img_dst: Path, lbl_dst: Path):
        label = labels_src / img_path.with_suffix(".txt").name
        if not label.exists():
            raise RuntimeError(f"Missing label for image: {img_path.name}")

        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(label, lbl_dst / label.name)

    for img in train_images:
        copy_pair(img, train_images_dir, train_labels_dir)

    for img in val_images:
        copy_pair(img, val_images_dir, val_labels_dir)

    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        gt_dir=args.ground_truth,
        out_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
