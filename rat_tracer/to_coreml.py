from ultralytics import YOLO
from rat_tracer.lib import best_model_path

model = YOLO(best_model_path)
model.export(format='coreml')