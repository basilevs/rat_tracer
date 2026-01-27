from ultralytics import YOLO
from lib import best_model_path

model = YOLO(best_model_path)
print(model.task)
print(model.model.args)
