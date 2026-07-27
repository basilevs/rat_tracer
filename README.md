# Rat Tracer
Trace a rat in a video recording of a labyrinth traversal and highlight visited areas.

Helps researcher to find key moments in labyrinth exploration experiment:
- goal reached
- whole labyrinth is explored

# Disclaimer
This is a research project not meant for general consumption. It has no covenient interface.

# Installation
pip install git+https://github.com/basilevs/rat_tracer

# Tracking workflow

## Prerequisites
- A video recording of an experiment from a static camera

## Steps
- Run:

    rat_tracer[.exe]

- Click "Open..."
- Select a video recording of an experiment
- Track subject's progress visually by watching the hightlighted visited area, or
- "Pause" monitoring to save CPU cycles
- When paused, seek around the already processed part of the video to locate key moments:
  - first visit of an area can be found by boxing it between moments where it has not yet been visited/highlighted and already has
  - complete exploration of a labirinth

The painted track is never erased, so once a goal it painted over, you know to rewind back to find the exact moment it happens.


# Training workflow
The project comes with a set of scripts to train custom object detection models for specific environments and subjects. This part is not user-friendly yet.

- Film your rat in a top-down view (referenced below as `video.mp4`)
- The published model is downloaded automatically from Hugging Face on first use;
  no manual model configuration is needed. Set the `RAT_TRACER_MODEL` environment
  variable to a local `.pt` path to override it (e.g. when training or publishing).
- Run `track.py video.mp4` (supports other popular video formats, but not everything)
- Inspect images from runs/detect/track*/track_loss
- Delete images with correct annotations
- Convert images with incorrect annotations to raw frames with track_to_frames.py
  - track_to_frames.py video.mp4 runs/detect/track*/track_loss
- Annotate raw frames runs/detect/track*/images/*.png in YOLO format using your prefered labeler
- `split.py uns/detect/track*/images data` (`data` is a path to `data` directory in rat_tracer repository)
- `RAT_TRACER_MODEL=path/to/last.pt train.py --new --pre`
- publish the resulting over-trained model by uploading it as `rat_tracer.pt`:
  `hf upload basilevs83/rat-tracer runs/detect/track*/weights/last.pt rat_tracer.pt`
- Repeat the cycle with other videos, remove `--pre` argument once dataset is larger that 500 images, you may have to edit `train.py` to use a freshly downloaded YOLO pretrained model
