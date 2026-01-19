from pprint import pprint
from sys import argv

from ultralytics import YOLO

model = YOLO("/Users/vasiligulevich/git/rat_tracer/runs/detect/train3/weights/best.pt")

results = model.predict(argv[1], stream=True)

for i in results:
    i.show()
