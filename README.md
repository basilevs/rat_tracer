# Rat Tracer
Trace a rat in a video recording of a labyrinth traversal and highlight visited areas.

Helps researcher to find key moments in labyrinth exploration experiment:
- goal reached
- whole labyrinth is explored

# Demo
[![Rat Tracer demo](https://img.youtube.com/vi/zyiqXmX0mmo/0.jpg)](https://www.youtube.com/watch?v=zyiqXmX0mmo)

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

# Reporting detection failures

The painted track is cumulative, so a red area is the union of everything detected so far — you cannot tell from it whether *this* frame was detected correctly. "Check detection" answers that, and lets you save frames the detector gets wrong so they can be used to improve it later.

- Tick "Check detection". Playback pauses, the track disappears, and the frame you stopped on is shown with only its own detection outlined.
- Seek, or step frame by frame with ◀ / ▶ (or the arrow keys), until you see a failure: a box where there is no animal, no box where there is one, or a box that is offset or far too large.
- Press "Mark bad frame" (or F2). The frame is saved; a message names it and offers Undo for five seconds.
- The control shows as already ticked for a frame you have saved before, so you never store the same one twice. Untick it (click, or F2 again) to remove a frame you saved by mistake — the five-second Undo only covers the frame you just saved, this works whenever the frame is on screen.
- Untick "Check detection" — or just press Play — to get the cumulative track back. Nothing is lost from it.

Saved frames accumulate across videos, experiments and restarts. When you want to hand them over, run:

    rat_tracer-collect

It takes no arguments and prints the path of the single archive it writes to your Desktop (or Documents). Nothing is deleted or moved, so you can run it as often as you like. Send that file to whoever maintains the model.

Set `RAT_TRACER_BAD_FRAMES` to keep saved frames somewhere else — a removable drive, for instance.

## Ingesting a collection round (for the model maintainer)

Extract the archive and annotate `images/*.png` in YOLO format, placing the labels in `labels/` beside it. The image names are the same ones `video_to_images.py` produces, so `split.py` accepts the directory unchanged:

    split.py <extracted-dir> data

Each frame has a sidecar in `meta/` recording what the model produced for it (an empty `boxes` list means it detected nothing — a missed detection), which weights produced it, and when it was marked. `index.jsonl` logs every mark and retraction, so `video_key` + `frame_index` identifies frames already ingested in an earlier round, and `marked_at` tells you which are new.

[![Rat Tracer traversal evaluation workflow](https://img.youtube.com/vi/Ybt3lNtIi9M/0.jpg)](https://www.youtube.com/watch?v=Ybt3lNtIi9M)


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
