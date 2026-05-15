Scripts to YOLO track a rat in a labyrinth.

Helps researcher to find key moments in labyrinth exploration experiment:
- goal reached
- whole labyrinth is explored

Top-down view
Custom dataset
Experiment - does not have an inteface

# Disclaimer
This is a research project not meant for general consumption. It has no covenient interface.

# Installation
git clone https://github.com/basilevs/rat_tracer.git  
pip install -e rat_tracer

# Training workflow
- Film your rat in a top-down view (referenced below as `video.mp4`)
- Download a pre-trained model and configure it in `lib.py/best_model_path`
- Run `track.py video.mp4` (supports other popular video formats, but not everything)
- Inspect images from runs/detect/track*/track_loss
- Delete images with correct annotations
- Convert images with incorrect annotations to raw frames with track_to_frames.py
  - track_to_frames.py video.mp4 runs/detect/track*/track_loss
- Annotate raw frames runs/detect/track*/images/*.png in YOLO format using your prefered labeler
- `split.py uns/detect/track*/images data` (`data` is a path to `data` directory in rat_tracer repository)
- `train.py --new --pre`
- update `lib.py/best_model_path` with the resulting over-trained model
- Repeat the cycle with other videos, remove `--pre` argument once dataset is larger that 500 images, you may have to edit `train.py` to use a freshly downloaded YOLO pretrained model

# Tracking workflow
Run:

    paint.py video.mp4 output.mp4

To highlight the ground covered by a rat in an experiment. Open output.mp4 in a video viewer, use quick rewind function to evaluate the experiment.
The painted track is never erased, so once it paints the labyrinth goal, you know to rewind back to find the exact moment it happens.
