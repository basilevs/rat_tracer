"""Extract some images from video"""
from pathlib import Path
from imageio_ffmpeg import read_frames
from imageio.v3 import imwrite
from numpy import uint8, frombuffer

VIDEO = "/private/tmp/input.mp4"
FRAMES = {2000, 4000, 6000}
OUT_DIR = Path("frames")
OUT_DIR.mkdir(exist_ok=True)

reader = read_frames(
    VIDEO,
    pix_fmt="rgb24",
)

meta = next(reader)
width, height = meta["size"]

for idx, frame_bytes in enumerate(reader):
    if not FRAMES:
        break
    if idx not in FRAMES:
        continue
    FRAMES.remove(idx)

    frame = frombuffer(frame_bytes, dtype=uint8)
    frame = frame.reshape((height, width, 3))

    imwrite(OUT_DIR / f"frame_{idx}.png", frame)
