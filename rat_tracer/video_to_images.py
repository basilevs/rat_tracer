from cv2 import imwrite, VideoCapture, CAP_PROP_POS_FRAMES
from pathlib import Path
from sys import argv

def main():
    VIDEO = Path(argv[1])
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    FRAMES = map(int, argv[2:])
    OUT_DIR = Path("images")
    OUT_DIR.mkdir(exist_ok=True)

    filename_prefix = VIDEO.with_suffix("").name;
    cap = VideoCapture(VIDEO)

    for frame in FRAMES:
        cap.set(CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame}")
        imwrite(str(OUT_DIR / f"{filename_prefix}_{frame:0>6}.png"), img)

    cap.release()

if __name__ == "__main__":
    main()