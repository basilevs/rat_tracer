from pprint import pprint
from psutil import Process

from ultralytics import YOLO

# Callback: print resident set size (RSS) memory
def print_rss_after_epoch(trainer):
    m = Process().memory_info()
    rss = m.rss / 1024. / 1024.
    vms = m.vms / 1024. / 1024.
    print(f"Epoch {trainer.epoch + 1}: RSS memory = {rss:.2f} MB, VMS memory = {vms:.2f} MB")

# Load a model
model = YOLO("runs/detect/train24/weights/last.pt")

model.add_callback("on_train_epoch_end", print_rss_after_epoch)

model.train(data="data/data.yaml", epochs=70, imgsz=640, device="mps", workers=2, dfl=10., resume=True)
