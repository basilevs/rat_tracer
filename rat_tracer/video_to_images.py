from pathlib import Path
import imageio_ffmpeg
import imageio.v3 as iio
from numpy import uint8, frombuffer

VIDEO = "/private/tmp/input.mp4"
FRAMES = {2000, 4000, 6000}   # use set for O(1) lookup
OUT_DIR = Path("frames")
OUT_DIR.mkdir(exist_ok=True)

reader = imageio_ffmpeg.read_frames(
    VIDEO,
    pix_fmt="rgb24",   # raw RGB frames
)

meta = next(reader)          # first item is metadata
width = meta["size"][0]
height = meta["size"][1]

for idx, frame_bytes in enumerate(reader):
    if idx not in FRAMES:
        continue

    frame = frombuffer(frame_bytes, dtype=uint8)
    frame = frame.reshape((height, width, 3))

    iio.imwrite(OUT_DIR / f"frame_{idx}.png", frame)

    if len(FRAMES) == 1:
        break
    FRAMES.remove(idx)

