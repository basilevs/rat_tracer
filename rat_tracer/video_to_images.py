from typing import Iterator
from cv2 import imwrite, VideoCapture, CAP_PROP_POS_FRAMES
from pathlib import Path
from sys import argv

def _main():
    VIDEO = Path(argv[1])
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    FRAMES = map(int, argv[2:])
    OUT_DIR = Path("images")
    extract_frames(VIDEO, FRAMES, OUT_DIR)

def extract_frames(video: Path, frames: Iterator[int], output_directory: Path):
    output_directory.mkdir(exist_ok=True)

    filename_prefix = video.with_suffix("").name;
    cap = VideoCapture(video)

    for frame in frames:
        cap.set(CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame}")
        imwrite(str(output_directory / f"{filename_prefix}_{frame:0>6}.png"), img)

    cap.release()

__all__ = ['extract_frames']
if __name__ == "__main__":
    _main()