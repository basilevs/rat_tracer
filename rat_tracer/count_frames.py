"""Count frames in video"""
from pprint import pprint
from imageio_ffmpeg import count_frames_and_secs

VIDEO = "/private/tmp/input.mp4"

data = count_frames_and_secs(VIDEO)

pprint(data)
