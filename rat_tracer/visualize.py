from pathlib import Path
from ultralytics.data.utils import visualize_image_annotations

label_map = {  # Define the label map with all annotated class labels.
    0: "rat",
    1: "human",
    2: "labyrinth"
}

root = Path('/Users/vasiligulevich/git/rat_tracer/data/')
labels_dir = root / 'labels'
images_dir = root / 'images'

def show(i:Path):
    l = labels_dir / i.relative_to(images_dir).with_suffix(".txt")
    print(i, l)
    print(l.read_text(), end='')
    # Visualize
    visualize_image_annotations(
        i,  # Input image path.
        l,  # Annotation file path for the image.
        label_map,
    )


for i in images_dir.rglob('*.png'):
    show(i)

#show(images_dir / 'Train/frame_000050.png')
