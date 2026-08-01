"""Tests for the marked-frame storage tree.

None of this needs Qt or a detection model: ``BadFrameStore`` is handed a root
directory explicitly, so the OS data-path lookup (the only Qt-dependent part of
:mod:`rat_tracer.bad_frames`) is exercised separately and only for the
environment override.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from rat_tracer.bad_frames import (
    STORAGE_ENV_VAR,
    BadFrameStore,
    Detection,
    MarkRequest,
    frame_name,
    storage_root,
)

_H, _W = 6, 8


def _image(shade: int = 40) -> np.ndarray:
    return np.full((_H, _W, 3), shade, dtype=np.uint8)


def _request(
    tmp_path: Path,
    *,
    frame_index: int = 42,
    video_key: str = "1a2b3c4d",
    stem: str = "run3",
    shade: int = 40,
    detection: Detection | None = None,
) -> MarkRequest:
    return MarkRequest(
        image=_image(shade),
        video_path=tmp_path / f"{stem}.mp4",
        video_key=video_key,
        frame_index=frame_index,
        timestamp_ms=frame_index * 40,
        detection=detection
        if detection is not None
        else Detection([[0.5, 0.3, 0.08, 0.11]], [0.9]),
        model_id="basilevs83/rat-tracer:rat_tracer.pt",
    )


def test_mark_writes_raw_image_sidecar_and_index(tmp_path: Path):
    store = BadFrameStore(tmp_path / "bad_frames")
    request = _request(tmp_path)

    name = store.mark(request)

    assert name == frame_name("run3", 42) == "run3_000042"
    image_path = store.images_dir / "run3_000042.png"
    assert image_path.is_file()
    # The stored image must be the raw frame: masked or box-annotated pixels
    # are unusable as training data (FR-12).
    import cv2

    stored_image = cv2.imread(str(image_path))
    assert stored_image is not None
    assert np.array_equal(stored_image, request.image)

    meta = json.loads((store.meta_dir / "run3_000042.json").read_text(encoding="utf-8"))
    assert meta["video_key"] == "1a2b3c4d"
    assert meta["frame_index"] == 42
    assert meta["video_stem"] == "run3"
    assert meta["detection"] == {"boxes": [[0.5, 0.3, 0.08, 0.11]], "conf": [0.9]}
    assert meta["marked_at"].endswith("Z")

    rows = [json.loads(line) for line in store.index_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["mark"]
    assert rows[0]["video_key"] == "1a2b3c4d"
    assert rows[0]["frame_index"] == 42


def test_missed_detection_is_recorded_as_empty_boxes(tmp_path: Path):
    """A false negative is the most valuable defect; it must not look like a
    frame whose detection was never attempted."""
    store = BadFrameStore(tmp_path / "bad_frames")
    store.mark(_request(tmp_path, detection=Detection()))

    meta = json.loads((store.meta_dir / "run3_000042.json").read_text(encoding="utf-8"))
    assert "detection" in meta
    assert meta["detection"] == {"boxes": [], "conf": []}


def test_marked_state_survives_a_restart(tmp_path: Path):
    root = tmp_path / "bad_frames"
    BadFrameStore(root).mark(_request(tmp_path, frame_index=7))

    reopened = BadFrameStore(root)
    assert reopened.is_marked("1a2b3c4d", 7)
    assert not reopened.is_marked("1a2b3c4d", 8)
    assert not reopened.is_marked("deadbeef", 7)


def test_retract_deletes_files_and_appends_a_retraction(tmp_path: Path):
    store = BadFrameStore(tmp_path / "bad_frames")
    store.mark(_request(tmp_path, frame_index=7))

    store.retract("1a2b3c4d", 7, "run3")

    assert not (store.images_dir / "run3_000007.png").exists()
    assert not (store.meta_dir / "run3_000007.json").exists()
    assert not store.is_marked("1a2b3c4d", 7)
    events = [json.loads(line)["event"] for line in store.index_path.read_text().splitlines()]
    # The original mark row is kept: erasing it would make the retraction rate
    # unmeasurable (FR-17).
    assert events == ["mark", "retract"]


def test_any_sequence_of_mark_and_undo_leaves_at_most_one_stored_frame(tmp_path: Path):
    store = BadFrameStore(tmp_path / "bad_frames")
    request = _request(tmp_path, frame_index=7)

    for _ in range(3):
        store.mark(request)
        store.retract("1a2b3c4d", 7, "run3")
    store.mark(request)

    stored = list(store.images_dir.glob("*.png"))
    assert len(stored) == 1
    assert store.marked_frames() == {("1a2b3c4d", 7)}


def test_same_frame_index_of_different_videos_deduplicates_independently(tmp_path: Path):
    """Duplicate detection is keyed on (video_key, frame_index), so the same
    physical video marked from different paths deduplicates while two genuinely
    different videos do not."""
    store = BadFrameStore(tmp_path / "bad_frames")
    store.mark(_request(tmp_path, frame_index=7, video_key="aaaa1111", stem="a"))

    assert store.is_marked("aaaa1111", 7)
    assert not store.is_marked("bbbb2222", 7)


def test_colliding_stems_keep_data_keyed_and_warn(tmp_path: Path, caplog):
    """Two videos sharing a stem overwrite one image file -- an accepted cost of
    filenames matching extract_frames. The stored data stays correctly keyed."""
    store = BadFrameStore(tmp_path / "bad_frames")
    store.mark(_request(tmp_path, frame_index=7, video_key="aaaa1111", stem="run3", shade=10))

    with caplog.at_level("WARNING", logger="rat_tracer.bad_frames"):
        store.mark(_request(tmp_path, frame_index=7, video_key="bbbb2222", stem="run3", shade=200))

    assert any("ambiguous" in record.message for record in caplog.records)
    assert store.marked_frames() == {("aaaa1111", 7), ("bbbb2222", 7)}
    meta = json.loads((store.meta_dir / "run3_000007.json").read_text(encoding="utf-8"))
    assert meta["video_key"] == "bbbb2222"


def test_a_torn_final_row_does_not_hide_earlier_marks(tmp_path: Path):
    """An interrupted append should cost one mark, not the whole log."""
    store = BadFrameStore(tmp_path / "bad_frames")
    store.mark(_request(tmp_path, frame_index=1))
    store.mark(_request(tmp_path, frame_index=2))
    with open(store.index_path, "a", encoding="utf-8") as fh:
        fh.write('{"event": "mark", "video_key": "1a2b3c4d", "frame_ind')

    reopened = BadFrameStore(store.root)
    assert reopened.marked_frames() == {("1a2b3c4d", 1), ("1a2b3c4d", 2)}


def test_an_interrupted_write_leaves_no_partial_image(tmp_path: Path, monkeypatch):
    store = BadFrameStore(tmp_path / "bad_frames")

    def explode(self, data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", explode)
    with pytest.raises(OSError):
        store.mark(_request(tmp_path))

    assert list(store.images_dir.glob("*")) == []
    assert not store.index_path.exists()
    assert not store.is_marked("1a2b3c4d", 42)


def test_storage_root_honours_the_environment_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(STORAGE_ENV_VAR, str(tmp_path / "elsewhere"))
    assert storage_root() == tmp_path / "elsewhere"


@pytest.mark.skipif(os.environ.get(STORAGE_ENV_VAR) is not None, reason="override is set")
def test_storage_root_defaults_to_a_persistent_user_directory(monkeypatch):
    """Never the temp dir: progress_cache lives there correctly (regenerable
    cache), but a marked frame is the only copy of the observation."""
    import tempfile

    monkeypatch.delenv(STORAGE_ENV_VAR, raising=False)
    root = storage_root()
    assert root.name == "bad_frames"
    assert Path(tempfile.gettempdir()) not in root.parents
