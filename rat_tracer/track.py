from pathlib import Path
from os import mkdir
from numpy import frombuffer, uint8
from ultralytics import YOLO
from ultralytics.engine.results import Results

model = YOLO("/Users/vasiligulevich/git/rat_tracer/runs/detect/train11/weights/best.pt")

def save_result(idx:int, results:Results):
    height, width = results.orig_shape
    data = frombuffer(results.orig_img, dtype=uint8)
    name = Path(results.path).with_suffix('').name
    data = data.reshape((height, width, 3))
    filename = f"{name}_{idx:0>6}.jpg"
    #imwrite(raw_dir / filename, data)
    target_dir = Path(results.save_dir) / 'track_loss'
    try:
        mkdir(target_dir)
    except FileExistsError:
        pass
    results.save(target_dir / filename)


def main():
    current_tracks = set()
    previous_result: Results = None
    stream = model.track('input/2025-10-16.mp4', show=True, stream=True, save_txt=False, save=True, verbose=False)
    for idx, results in enumerate(stream):
        if results.boxes.id is None or not results.boxes.id.numel():
            found = set()
        else:
            found = set(results.boxes.id.tolist())
        lost = current_tracks.difference(found)
        current_tracks = found
        print("Frame:", idx, "tracks:", found)
        if lost:
            save_result(idx - 1, previous_result)
            save_result(idx, results)
            print(f"Frame {idx} has lost tracks: " + str(lost))
        previous_result = results

if __name__ == "__main__":
    main()