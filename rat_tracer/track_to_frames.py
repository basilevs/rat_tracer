"""
Given a directory of images named xxx_000001.jpg, create a nearby directory of video frames extracted from video.
"""

from pathlib import Path
from sys import argv
from typing import Iterator
from video_to_images import extract_frames


def _path_to_frame(p: Path) -> int:
    return int(p.with_suffix('').name.split('_')[:-1])

def _main():
    video = Path(argv[1])
    input_dir = Path(argv[2])
    frames: Iterator[int] = [_path_to_frame(p) for p in input_dir.glob("*.jpg")]
    extract_frames(video, frames, input_dir.parent / 'images')

if __name__ == "__main__":
    _main()
