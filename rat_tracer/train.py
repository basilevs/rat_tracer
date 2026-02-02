from pathlib import Path
from pprint import pprint
from sys import argv
from gc import collect
from psutil import Process

from ultralytics import YOLO
from wakepy import keep

from lib import best_model_path


# Callback: print resident set size (RSS) memory
def print_rss_after_epoch(trainer):
    collect()
    m = Process().memory_info()
    rss = m.rss / 1024. / 1024.
    vms = m.vms / 1024. / 1024.
    print(f"Epoch {trainer.epoch + 1}: RSS memory = {rss:.2f} MB, VMS memory = {vms:.2f} MB")

def latest_train():
    root = Path('runs') / 'detect'
    index = max([int(p.name[5:] or 0)  for p in root.glob('train*')])
    suffix = str (index) if index else "0"
    return root / (f"train{suffix}")

def main():
    resume = False

    if "--new" in argv:
        resume = False
        model = YOLO('input/yolo26n.pt')
    else:
        train = latest_train()
        resume = True
        if (train / 'results.png').exists():
            raise ValueError(f'{train} is fully trained. Use --new')
        model = YOLO(train / 'weights' / 'best.pt')

    model.add_callback("on_train_epoch_end", print_rss_after_epoch)


    with keep.running():
        model.train(data="data/data.yaml", epochs=100, workers=2, resume=resume,
            device="mps",
            mosaic=0.5,
        )

if __name__ == '__main__':
    main()