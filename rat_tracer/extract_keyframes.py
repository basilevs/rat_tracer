"""Extracts annotation keyframes as rectangles"""

from pathlib import Path
from xml.etree import ElementTree
from typing import Dict, Generator, Tuple
from imageio import imwrite
from imageio_ffmpeg import read_frames
from numpy import frombuffer, uint8

OUT_DIR = Path("rat_images")
OUT_DIR.mkdir(exist_ok=True)


def iter_frames(video_path, *, pix_fmt="rgb24"):
    """
    Yield (frame_index, frame_array) for all decoded frames.

    frame_array shape: (height, width, 3), dtype=uint8
    """
    reader = read_frames(video_path, pix_fmt=pix_fmt)
    meta = next(reader)
    width, height = meta["size"]

    for idx, frame_bytes in enumerate(reader):
        frame = frombuffer(frame_bytes, dtype=uint8)
        frame = frame.reshape((height, width, 3))
        yield idx, frame

def extract_box_coordinates(xml_path: str) -> Generator[Tuple[int, float, float, float, float], None, None]:
    """
    Parse an XML file and yield rectangle coordinates from <box> elements under <track>.

    Args:
        xml_path: Path to the XML file.

    Yields:
        Tuple of floats: (xtl, ytl, xbr, ybr)
    """
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()

    for box in root.findall('./track[@label="rat"]/box[@keyframe="1"]'):
        try:
            if box.get('occluded') == '1':
                continue
            if box.get('outside') == '1':
                continue
            coords = (
                int(box.get('frame')),
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr")),
            )
            yield coords
        except (TypeError, ValueError):
            # Skip boxes with missing or non-float attributes
            continue
by_frame: Dict[int, Tuple[float, float, float, float]] = {}

def extract_region(frame, x1:float, y1:float, x2:float, y2:float):
    """
    Extract a rectangular region from a frame.

    (x1, y1) – top-left corner (inclusive)
    (x2, y2) – bottom-right corner (exclusive)

    Returns a view (no copy) when possible.
    """
    return frame[int(y1):int(y2), int(x1):int(x2)]

for i in extract_box_coordinates('input/annotations.xml'):
    by_frame[i[0]] = i[1:]


for idx, frame in iter_frames("input/input.mp4"):
    if idx not in by_frame:
        continue
    region = by_frame.pop(idx)
    frame = extract_region(frame, *region)
    imwrite(OUT_DIR / f"frame_{idx}.png", frame)
