from ultralytics import YOLO

from rat_tracer.lib import model_path

model = YOLO(model_path())
model.export(format="coreml")
