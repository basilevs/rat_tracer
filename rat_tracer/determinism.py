"""
Demonstrates detection non-determinism depending on input length
"""

from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import init_seeds

model = YOLO('runs/detect/train35/weights/last.pt')
model.checks()

root = Path('data/images')
file_of_interest = root / 'Val' / '2025-10-10_001027.png'
side_effect_file = root / 'Train' / '2026-01-27_000003.jpeg'

def is_expected(files: list[Path]):
    init_seeds(0, True)
    assert file_of_interest in files
    resultsIterator = model.predict(files, stream=True, imgsz=640, verbose=False, rect=False, workers=0, deterministic=True)
    result = next(i for i in resultsIterator if Path(i.path) == file_of_interest)
    return 0 not in result.boxes.cls

assert is_expected([file_of_interest]) # pass
assert is_expected([file_of_interest, side_effect_file]) # fail
