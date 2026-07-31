"""Tests for on-demand single-frame detection.

The published model is downloaded from Hugging Face and is unavailable offline,
so nothing here loads real weights: the box-extraction logic is driven through
a fake ultralytics result, and the rest is pure path handling.
"""

from pathlib import Path

import numpy as np
import pytest
from rat_tracer.frame_detector import RAT_CLASS, YoloFrameDetector, resolve_model_id
from rat_tracer.lib import MODEL_ENV_VAR, MODEL_FILENAME, MODEL_REPO_ID


class _FakeTensor:
    """Stands in for the torch tensors ultralytics returns per box."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return self._value


class _FakeBoxes:
    def __init__(self, boxes, confidences, classes):
        self.xywhn = [_FakeTensor(box) for box in boxes]
        self.conf = [_FakeTensor(c) for c in confidences]
        self.cls = [_FakeTensor(c) for c in classes]

    def __len__(self):
        return len(self.xywhn)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self._results


def _detector_with(results) -> YoloFrameDetector:
    detector = YoloFrameDetector(weights=Path("unused.pt"))
    detector._model = _FakeModel(results)
    return detector


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_boxes_are_reported_as_normalized_cxcywh_with_confidence():
    detector = _detector_with(
        [_FakeResult(_FakeBoxes([[0.5, 0.25, 0.1, 0.2]], [0.91], [RAT_CLASS]))]
    )

    detection = detector.detect(_frame())

    assert detection.boxes == [[0.5, 0.25, 0.1, 0.2]]
    assert detection.conf == [pytest.approx(0.91)]


def test_a_frame_with_nothing_found_yields_empty_boxes():
    """Not an error and not a missing answer: this is the record of a false
    negative, the most important defect a researcher can report."""
    detector = _detector_with([_FakeResult(_FakeBoxes([], [], []))])

    detection = detector.detect(_frame())

    assert detection.boxes == []
    assert detection.conf == []


def test_results_without_boxes_are_tolerated():
    detector = _detector_with([_FakeResult(None)])

    assert detector.detect(_frame()).boxes == []


def test_other_classes_are_ignored():
    """Matches the cumulative pass, which paints only the subject class."""
    detector = _detector_with(
        [
            _FakeResult(
                _FakeBoxes(
                    [[0.1, 0.1, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]],
                    [0.8, 0.9],
                    [RAT_CLASS + 1, RAT_CLASS],
                )
            )
        ]
    )

    detection = detector.detect(_frame())

    assert detection.boxes == [[0.7, 0.7, 0.2, 0.2]]
    assert detection.conf == [pytest.approx(0.9)]


def test_published_weights_are_identified_by_their_hugging_face_coordinates(monkeypatch):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    assert resolve_model_id(Path("/cache/rat_tracer.pt")) == f"{MODEL_REPO_ID}:{MODEL_FILENAME}"


def test_overridden_weights_are_identified_by_path_and_content(tmp_path: Path, monkeypatch):
    """A path alone is not enough: training overwrites last.pt in place, so two
    archives could name the same file and mean different weights."""
    weights = tmp_path / "last.pt"
    weights.write_bytes(b"weights-v1")
    monkeypatch.setenv(MODEL_ENV_VAR, str(weights))

    first = resolve_model_id(weights)
    assert first.startswith(f"file:{weights}#crc32=")

    weights.write_bytes(b"weights-v2")
    assert resolve_model_id(weights) != first


def test_model_id_of_unreadable_overridden_weights_falls_back_to_the_path(
    tmp_path: Path, monkeypatch
):
    missing = tmp_path / "gone.pt"
    monkeypatch.setenv(MODEL_ENV_VAR, str(missing))

    assert resolve_model_id(missing) == f"file:{missing}"


def test_prewarm_survives_an_unloadable_model():
    """Prewarming is an optimization; failing it must not break the mode -- the
    researcher should still get an answer (or a disabled control), not a crash
    on a machine where the weights are missing or corrupt."""

    class _BrokenModel:
        def predict(self, **kwargs):
            raise RuntimeError("weights are not readable")

    detector = YoloFrameDetector(weights=Path("unused.pt"))
    detector._model = _BrokenModel()

    detector.prewarm()  # must not raise
