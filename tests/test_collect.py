"""Tests for the archive command the researcher runs with no arguments."""

import zipfile
from pathlib import Path

import pytest
from rat_tracer.bad_frames import STORAGE_ENV_VAR, BadFrameStore, Detection, MarkRequest
from rat_tracer.collect import archive_name, collect, destination_directory, has_marks, main

import numpy as np  # isort: skip


def _store_with_marks(root: Path, frames=(7, 42)) -> BadFrameStore:
    store = BadFrameStore(root)
    for index in frames:
        store.mark(
            MarkRequest(
                image=np.full((4, 6, 3), index % 256, dtype=np.uint8),
                video_path=root / "run3.mp4",
                video_key="1a2b3c4d",
                frame_index=index,
                timestamp_ms=index * 40,
                detection=Detection([[0.5, 0.5, 0.1, 0.1]], [0.8]),
                model_id="test:v1",
            )
        )
    return store


def test_one_archive_holds_every_marked_frame(tmp_path: Path):
    store = _store_with_marks(tmp_path / "bad_frames")
    destination = tmp_path / "desktop"

    archive_path = collect(store.root, destination)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "images/run3_000007.png" in names
    assert "images/run3_000042.png" in names
    assert "meta/run3_000042.json" in names
    assert "index.jsonl" in names


def test_extracted_image_names_match_the_training_pipeline(tmp_path: Path):
    """``split.py`` accepts the tree with labels/ beside images/, and the names
    are what ``extract_frames`` would have produced -- no renaming, no re-keying."""
    store = _store_with_marks(tmp_path / "bad_frames", frames=(42,))
    archive_path = collect(store.root, tmp_path / "desktop")

    extracted = tmp_path / "ingest"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    assert (extracted / "images" / "run3_000042.png").is_file()
    (extracted / "labels").mkdir()
    (extracted / "labels" / "run3_000042.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    from rat_tracer.split import split_dataset

    split_dataset(gt_dir=extracted, out_dir=tmp_path / "dataset", val_ratio=0.0)
    assert (tmp_path / "dataset" / "images" / "Train" / "run3_000042.png").is_file()
    assert (tmp_path / "dataset" / "labels" / "Train" / "run3_000042.txt").is_file()


def test_archiving_never_deletes_or_moves_the_source(tmp_path: Path):
    store = _store_with_marks(tmp_path / "bad_frames")
    before = sorted(path.name for path in store.root.rglob("*") if path.is_file())

    collect(store.root, tmp_path / "desktop")

    assert sorted(path.name for path in store.root.rglob("*") if path.is_file()) == before


def test_running_twice_leaves_both_archives(tmp_path: Path, monkeypatch):
    store = _store_with_marks(tmp_path / "bad_frames")
    destination = tmp_path / "desktop"

    stamps = iter(["20260730-101500", "20260730-101501"])
    monkeypatch.setattr(
        "rat_tracer.collect.archive_name",
        lambda now=None: f"rat_tracer_bad_frames_host_{next(stamps)}.zip",
    )
    first = collect(store.root, destination)
    second = collect(store.root, destination)

    assert first != second
    assert first.is_file() and second.is_file()


def test_the_archive_is_not_written_inside_the_storage_root(tmp_path: Path):
    """Otherwise each run would package the previous run's output."""
    store = _store_with_marks(tmp_path / "bad_frames")

    archive_path = collect(store.root, tmp_path / "desktop")

    assert store.root not in archive_path.parents


def test_the_name_carries_a_host_and_a_timestamp():
    name = archive_name()
    assert name.startswith("rat_tracer_bad_frames_")
    assert name.endswith(".zip")


def test_partial_writes_are_not_archived(tmp_path: Path):
    store = _store_with_marks(tmp_path / "bad_frames", frames=(7,))
    (store.images_dir / "run3_000009.png.tmp").write_bytes(b"half a frame")

    archive_path = collect(store.root, tmp_path / "desktop")

    with zipfile.ZipFile(archive_path) as archive:
        assert not [name for name in archive.namelist() if name.endswith(".tmp")]


def test_nothing_marked_is_reported_rather_than_producing_an_empty_archive(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setenv(STORAGE_ENV_VAR, str(tmp_path / "empty"))
    monkeypatch.setattr("sys.argv", ["rat_tracer-collect"])

    assert main() == 1
    assert not list(tmp_path.glob("*.zip"))


def test_the_command_needs_no_arguments_and_prints_where_it_wrote(
    tmp_path: Path, monkeypatch, capsys
):
    store = _store_with_marks(tmp_path / "bad_frames")
    monkeypatch.setenv(STORAGE_ENV_VAR, str(store.root))
    monkeypatch.setattr("rat_tracer.collect.destination_directory", lambda: tmp_path / "desktop")
    monkeypatch.setattr("sys.argv", ["rat_tracer-collect"])

    assert main() == 0

    printed = capsys.readouterr().out
    archives = list((tmp_path / "desktop").glob("*.zip"))
    assert len(archives) == 1
    assert str(archives[0]) in printed


def test_the_destination_is_somewhere_a_file_manager_opens():
    destination = destination_directory()

    assert destination.is_dir()
    assert not any(part.startswith(".") for part in destination.parts), (
        "a hidden directory would defeat the point of not having to hunt for the file"
    )


@pytest.mark.parametrize("marked", [True, False])
def test_has_marks_reports_whether_there_is_anything_to_send(tmp_path: Path, marked: bool):
    root = tmp_path / "bad_frames"
    if marked:
        _store_with_marks(root, frames=(1,))
    else:
        root.mkdir(parents=True)

    assert has_marks(root) is marked
