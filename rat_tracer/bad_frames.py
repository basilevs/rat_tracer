"""Persistent storage for frames a researcher marked as detection failures.

The storage root is a user-level, per-OS data directory -- deliberately *not*
``tempfile.gettempdir()`` where :mod:`rat_tracer.progress_cache` lives. That is
correct for a regenerable cache and fatal here: a marked frame is the only copy
of the observation, and a reboot would wipe it.

Layout::

    bad_frames/
      images/   <video-stem>_<frame:06d>.png   # raw frame -- annotate these
      meta/     <video-stem>_<frame:06d>.json  # sidecar, see MarkRecord
      index.jsonl                              # append-only log of all marks

Image names are byte-identical to what
:func:`rat_tracer.video_to_images.extract_frames` produces, so the technician's
existing ``split.py`` / ``track_to_frames.py`` tooling accepts the extracted
tree with no renaming. The cost is that two videos sharing a stem collide on
the same frame index; the stored data stays correctly keyed by
``video_key + frame_index`` and only the filename is ambiguous, so a collision
is logged rather than worked around.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from threading import Lock

import cv2
from numpy import ndarray

logger = getLogger(__name__)

#: Overrides the OS data directory. Mainly for tests and field machines that
#: keep everything on removable media.
STORAGE_ENV_VAR = "RAT_TRACER_BAD_FRAMES"

APPLICATION_NAME = "rat_tracer"

_EVENT_MARK = "mark"
_EVENT_RETRACT = "retract"


def configure_application_identity() -> None:
    """Give Qt the application name its data paths are derived from.

    ``QStandardPaths.AppDataLocation`` appends ``applicationName`` to the OS
    data directory, and Qt defaults that to ``argv[0]`` -- which would scatter
    marked frames across a different directory per launch method (``rat_tracer``
    console script, ``python -m``, a test runner). Only ``applicationName`` is
    set: adding ``organizationName`` would nest the path as
    ``rat_tracer/rat_tracer`` on Linux.
    """
    from PySide6.QtCore import QCoreApplication

    if not QCoreApplication.applicationName() or QCoreApplication.applicationName() != (
        APPLICATION_NAME
    ):
        QCoreApplication.setApplicationName(APPLICATION_NAME)


def storage_root() -> Path:
    """Resolve the directory holding marked frames, creating nothing yet."""
    override = os.environ.get(STORAGE_ENV_VAR)
    if override:
        return Path(override)

    from PySide6.QtCore import QStandardPaths

    configure_application_identity()
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        # Qt could not determine a data directory (no HOME, sandboxed shell).
        base = str(Path.home() / ".local" / "share" / APPLICATION_NAME)
        logger.warning("Qt reported no writable data location; falling back to %s", base)
    return Path(base) / "bad_frames"


@dataclass(frozen=True)
class Detection:
    """What the detector produced for one frame.

    ``boxes`` are normalized ``[cx, cy, w, h]`` -- the same convention as the
    YOLO label files the technician will produce, so no conversion is needed at
    annotation time. An empty list is the record of a *missed* detection and is
    meaningfully different from no detection having been attempted, which is
    represented by the absence of a ``Detection`` altogether.
    """

    boxes: list[list[float]] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"boxes": self.boxes, "conf": self.conf}


@dataclass(frozen=True)
class MarkRequest:
    """Everything needed to store one marked frame."""

    image: ndarray
    video_path: Path
    video_key: str
    frame_index: int
    timestamp_ms: int
    detection: Detection
    model_id: str

    @property
    def video_stem(self) -> str:
        return self.video_path.stem


def frame_name(video_stem: str, frame_index: int) -> str:
    """The shared stem of a marked frame's image and sidecar files."""
    return f"{video_stem}_{frame_index:06d}"


def app_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("rat_tracer")
    except PackageNotFoundError:  # running from a source tree without an install
        return "unknown"


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write *data* to *dest* so an interrupted save leaves no partial file.

    Mirrors :func:`rat_tracer.progress_cache.save_progress`: write a sibling
    temp file, then rename it over the destination, which is atomic within a
    filesystem.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class BadFrameStore:
    """Reads and writes the bad-frame storage tree.

    Instances are safe to share across threads: every mutation is serialized on
    a single lock, so a mark and a retraction of the same frame cannot
    interleave into a state where files and index disagree.
    """

    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else storage_root()
        self._lock = Lock()
        self._marked: set[tuple[str, int]] = set()
        self._loaded = False

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    def _ensure_loaded(self) -> None:
        """Rebuild the marked set from the log, at most once per instance."""
        if self._loaded:
            return
        self._marked = self._replay_index()
        self._loaded = True

    def _replay_index(self) -> set[tuple[str, int]]:
        """Derive the currently-marked frames by replaying the append-only log.

        The log -- not the image directory -- is the source of truth for what is
        marked. Two videos sharing a stem write to the same image filename, so
        file existence cannot answer "is this frame marked" for either of them.
        """
        marked: set[tuple[str, int]] = set()
        if not self.index_path.exists():
            return marked
        try:
            lines = self.index_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.exception("Cannot read %s; treating all frames as unmarked", self.index_path)
            return marked
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row["video_key"], int(row["frame_index"]))
                event = row["event"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A torn final row is the expected shape of an interrupted
                # append; skipping it loses one mark, not the whole log.
                logger.warning("Skipping malformed row %d in %s", number, self.index_path)
                continue
            if event == _EVENT_MARK:
                marked.add(key)
            elif event == _EVENT_RETRACT:
                marked.discard(key)
        logger.debug("Replayed %s: %d frames currently marked", self.index_path, len(marked))
        return marked

    def is_marked(self, video_key: str, frame_index: int) -> bool:
        with self._lock:
            self._ensure_loaded()
            return (video_key, frame_index) in self._marked

    def marked_frames(self) -> set[tuple[str, int]]:
        with self._lock:
            self._ensure_loaded()
            return set(self._marked)

    def _append_index(self, event: str, video_key: str, frame_index: int, marked_at: str) -> None:
        row = {
            "event": event,
            "marked_at": marked_at,
            "video_key": video_key,
            "frame_index": frame_index,
        }
        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def mark(self, request: MarkRequest) -> str:
        """Store one marked frame; returns its ``<stem>_<index>`` name.

        Raises on any I/O failure so the caller can tell the researcher the
        frame was *not* saved.
        """
        with self._lock:
            self._ensure_loaded()
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.meta_dir.mkdir(parents=True, exist_ok=True)

            name = frame_name(request.video_stem, request.frame_index)
            image_path = self.images_dir / f"{name}.png"
            meta_path = self.meta_dir / f"{name}.json"

            if image_path.exists() and not self._names_same_video(meta_path, request.video_key):
                logger.warning(
                    "%s already holds a frame from a different video; overwriting it. "
                    "The stored data stays keyed by video_key, but this filename is ambiguous.",
                    image_path,
                )

            ok, encoded = cv2.imencode(".png", request.image)
            if not ok:
                raise OSError(f"Cannot PNG-encode frame {request.frame_index}")

            marked_at = _utc_now()
            meta = {
                "video_path": str(request.video_path),
                "video_stem": request.video_stem,
                "video_key": request.video_key,
                "frame_index": request.frame_index,
                "timestamp_ms": request.timestamp_ms,
                "marked_at": marked_at,
                "model_id": request.model_id,
                "app_version": app_version(),
                "detection": request.detection.as_json(),
            }

            _atomic_write(image_path, encoded.tobytes())
            _atomic_write(meta_path, json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
            self._append_index(_EVENT_MARK, request.video_key, request.frame_index, marked_at)
            self._marked.add((request.video_key, request.frame_index))

        logger.info("Marked frame %d of %s -> %s", request.frame_index, request.video_path, name)
        return name

    def _names_same_video(self, meta_path: Path, video_key: str) -> bool:
        """True when an existing sidecar belongs to the video we are storing."""
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True  # unreadable sidecar: nothing useful to warn about
        return existing.get("video_key") == video_key

    def retract(self, video_key: str, frame_index: int, video_stem: str) -> None:
        """Undo a mark: delete its files, but record the retraction.

        The log stays append-only. Erasing the original ``mark`` row would make
        the retraction rate -- the signal that the control is too easy to hit by
        accident -- unmeasurable.
        """
        with self._lock:
            self._ensure_loaded()
            name = frame_name(video_stem, frame_index)
            for path in (self.images_dir / f"{name}.png", self.meta_dir / f"{name}.json"):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Cannot delete %s while retracting a mark", path)
            self.root.mkdir(parents=True, exist_ok=True)
            self._append_index(_EVENT_RETRACT, video_key, frame_index, _utc_now())
            self._marked.discard((video_key, frame_index))
        logger.info("Retracted mark for frame %d (%s)", frame_index, video_key)
