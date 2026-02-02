from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from os import mkdir
from sys import argv
from numpy import frombuffer, uint8
from ultralytics import YOLO
from ultralytics.engine.results import Results

from lib import Box, Point, Predictions, best_model_path, nms_callback

LABYRINTH_CLASS = 2

# normalized margins (fractions of image size)
EDGE_MARGIN = 0.02
LABYRINTH_MARGIN = 0.1


def save_result(idx: int, results: Results):
    height, width = results.orig_shape
    data = frombuffer(results.orig_img, dtype=uint8)
    name = Path(results.path).with_suffix('').name
    data = data.reshape((height, width, 3))
    filename = f"{name}_{idx:0>6}.jpg"
    target_dir = Path(results.save_dir) / 'track_loss'
    try:
        mkdir(target_dir)
    except FileExistsError:
        pass
    results.save(target_dir / filename)


def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    return (
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height
    )

def near_image_edge_norm(box_norm, margin):
    x1, y1, x2, y2 = box_norm
    return near_border(box_norm, (0., 0., 1., 1.), margin)

def near_border(box, border, margin):
    if border is None:
        return False

    x1, y1, x2, y2 = box
    lx1, ly1, lx2, ly2 = border

    return (
        abs(x1 - lx1) < margin or
        abs(y1 - ly1) < margin or
        abs(x2 - lx2) < margin or
        abs(y2 - ly2) < margin
    )


def extract_labyrinth_box(results: Results):
    """Return normalized labyrinth box (0-1) or None"""
    if results.boxes.cls is None:
        return None

    height, width = results.orig_shape

    for box, cls in zip(results.boxes.xyxy.tolist(),
                        results.boxes.cls.tolist()):
        if int(cls) == LABYRINTH_CLASS:
            return normalize_box(box, width, height)

    return None


def find_box_for_track(results: Results, track_id):
    """Return normalized box (0-1) for the given track_id using results.orig_shape"""
    if results is None or results.boxes.id is None:
        return None

    height, width = results.orig_shape

    for tid, box in zip(results.boxes.id.tolist(),
                        results.boxes.xyxy.tolist()):
        if tid == track_id:
            return normalize_box(box, width, height)

    return None

def track_set(results: Results) -> set[float]:
    if results.boxes.id is None or not results.boxes.id.numel():
        found = set()
    else:
        found = set(results.boxes.id.tolist())
    return found

def main(input_video: Path):
    model = YOLO(best_model_path)
    model.add_callback("on_predict_postprocess_end", nms_callback)

    stream = model.track(
        input_video,
        show=True,
        conf=0.1,
        stream=True,
        save_txt=False,
        save=True,
        nms=True,
        verbose=False,
        tracker="botsort.yaml"
    )

    previous_result: Results = None
    frames_with_rat = 0
    frames_without_rat = 0
    track_loss = 0
    for idx, results in enumerate(stream):
        if not previous_result:
            previous_result = results
            continue

        if (results.boxes.cls == 0).any():
            frames_with_rat += 1
        else:
            frames_without_rat += 1

        current_tracks = track_set(previous_result)
        found = track_set(results)

        lost = current_tracks.difference(found)

        if lost and previous_result is not None:
            previous_predictions = Predictions.from_results(previous_result)
            for tid in lost:
                lost_prediction = previous_predictions.by_track(tid)
                if lost_prediction.cls != 0: # consider only rats
                    continue

                all_predictions = previous_predictions + Predictions.from_results(results)
                # rat is expected to exit through ports and to be occluded by humans
                ports = all_predictions.by_class(3)
                humans = all_predictions.by_class(1)
                if any(b.box.near(lost_prediction.box, 5) for b in [*ports, *humans]):
                    continue

                save_result(idx - 1, previous_result)
                save_result(idx, results)
                print(f"Frame {idx} has mined track loss: {tid}")
                track_loss += 1
                break  # save each frame only once

        previous_result = results
    
    with open(Path(previous_result.save_dir) / 'statistics.txt', 'w', encoding='utf-8') as f:
        def p(*args):
            print(*args, file=f)
            print(*args)
        p('Input file:', input_video)
        p('Model:', model.model_name)
        p('Frames with rats:', frames_with_rat)
        p('Frames without rats:', frames_without_rat)
        p('Rat track loss:', track_loss)

if __name__ == "__main__":
    main(argv[1])
