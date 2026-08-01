"""On-demand detection for a single frame the researcher is looking at.

The cumulative pass in :mod:`rat_tracer.paint` cannot answer this. It runs
strictly forward from the start of the video and refines each detection with a
MOG2 background subtractor whose state depends on every preceding frame, so a
frame reached by seeking has no reproducible answer there -- and its own
momentary mask cannot be recovered from ``CoverageHistory`` either, which
stores the cumulative union rather than per-frame masks.

So problem reporting mode runs its own inference, on its own model instance.
Sharing the cumulative pass's model would make every request wait for the
in-flight batch; a second instance costs about 100 MB and answers in roughly
150-270 ms even while the cumulative pass saturates the CPU.
"""

import zlib
from logging import getLogger
from os import environ
from pathlib import Path
from typing import Any, Protocol

from numpy import ndarray

from rat_tracer.bad_frames import Detection
from rat_tracer.lib import MODEL_ENV_VAR, MODEL_FILENAME, MODEL_REPO_ID, model_path

logger = getLogger(__name__)

#: Matches ``rat_tracer.paint.RAT_CLASS``: other classes are not the subject.
RAT_CLASS = 0
#: Matches the confidence threshold the cumulative pass predicts with, so the
#: researcher judges the same detections the coverage mask was built from.
CONFIDENCE_THRESHOLD = 0.25

_CHUNK = 1 << 20


class FrameDetector(Protocol):
    """What problem reporting mode needs from a detector.

    Narrow on purpose: the UI is tested against a fake implementing this, with
    no model weights present -- which matters because the published model is
    downloaded from Hugging Face and is unavailable offline and in CI.
    """

    @property
    def model_id(self) -> str:
        """Identifies the weights that produced a detection, for the sidecar."""
        ...

    def prewarm(self) -> None:
        """Load and exercise the model so the first real request is not slow."""
        ...

    def detect(self, image: ndarray) -> Detection:
        """Run detection on one BGR frame."""
        ...


def _file_fingerprint(path: Path) -> str:
    """Hex crc32 of a file, matching ``progress_cache.video_key``'s scheme."""
    crc = 0
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    crc = zlib.crc32(size.to_bytes(8, "little"), crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def resolve_model_id(weights: Path) -> str:
    """Describe *weights* well enough for the technician to identify them.

    Without an override the published model is named by its Hugging Face
    coordinates. With ``RAT_TRACER_MODEL`` set, the path alone would be
    useless -- a training run overwrites ``last.pt`` in place -- so the file's
    content fingerprint is appended.
    """
    if not environ.get(MODEL_ENV_VAR):
        return f"{MODEL_REPO_ID}:{MODEL_FILENAME}"
    try:
        return f"file:{weights}#crc32={_file_fingerprint(weights)}"
    except OSError:
        logger.warning("Cannot fingerprint %s; recording its path alone", weights)
        return f"file:{weights}"


class YoloFrameDetector(FrameDetector):
    """Runs YOLO on single frames, loading the model on first use.

    Construction is cheap and must stay that way: it happens on the UI thread,
    while ``prewarm`` and ``detect`` are expected to run on a worker.
    """

    def __init__(self, weights: Path | None = None):
        self._weights = weights
        # Untyped because ultralytics is imported lazily: importing it costs
        # seconds and must not happen just because the UI module was loaded.
        self._model: Any = None
        self._model_id: str | None = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            weights = self._weights if self._weights is not None else model_path()
            logger.info("Loading on-demand detection model from %s", weights)
            self._model = YOLO(weights)
            self._model_id = resolve_model_id(weights)
        return self._model

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            weights = self._weights if self._weights is not None else model_path()
            self._model_id = resolve_model_id(weights)
        return self._model_id

    def prewarm(self) -> None:
        """Pay the model load and the first-inference cost up front.

        The first inference in a process costs seconds (framework warmup) while
        later ones cost a fraction of a second. Doing it when the mode is
        entered keeps that cost off the first frame the researcher judges.
        """
        from numpy import zeros

        try:
            self.detect(zeros((64, 64, 3), dtype="uint8"))
        except Exception:
            logger.exception("Prewarming the on-demand detector failed")

    def detect(self, image: ndarray) -> Detection:
        from torch.accelerator import current_accelerator

        model = self._ensure_model()
        results = model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            show=False,
            device=current_accelerator(),
        )
        boxes: list[list[float]] = []
        confidences: list[float] = []
        for result in results:
            detected = result.boxes
            if detected is None:
                continue
            for i in range(len(detected)):
                if int(detected.cls[i].item()) != RAT_CLASS:
                    continue
                # Normalized [cx, cy, w, h] -- the convention the technician's
                # YOLO label files use, so the sidecar needs no conversion.
                boxes.append([float(v) for v in detected.xywhn[i].tolist()])
                confidences.append(float(detected.conf[i].item()))
        logger.debug("detect: %d box(es)", len(boxes))
        return Detection(boxes=boxes, conf=confidences)
