from pathlib import Path
from imageio import imwrite
from imageio_ffmpeg import read_frames
from numpy import frombuffer, uint8
from sys import argv

video = Path(argv[1])
frame = argv[2]

reader = read_frames(video, output_params=["-vf", f"select=eq(n\\,{frame})"])

meta = next(reader, None)
width, height = meta["size"]

for data in reader:
    data = frombuffer(data, dtype=uint8)
    data = data.reshape((height, width, 3))
    imwrite(f"frame_{frame:0>6}.png", data)
    break