from pprint import pprint
from psutil import Process

from ultralytics import YOLO

from lib import best_model_path

# Callback: print resident set size (RSS) memory
def print_rss_after_epoch(trainer):
    m = Process().memory_info()
    rss = m.rss / 1024. / 1024.
    vms = m.vms / 1024. / 1024.
    print(f"Epoch {trainer.epoch + 1}: RSS memory = {rss:.2f} MB, VMS memory = {vms:.2f} MB")

# Load a model
model = YOLO(best_model_path)

model.add_callback("on_train_epoch_end", print_rss_after_epoch)

model.train(data="data/data.yaml", epochs=100, workers=2, patience=15, resume=True,
    device="mps",
)
