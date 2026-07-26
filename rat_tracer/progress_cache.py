import pickle
import tempfile
import zlib
from logging import DEBUG, getLogger
from pathlib import Path
from time import perf_counter

from rat_tracer.coverage import CoverageHistory

_CHUNK = 1 << 20  # 1 MiB read chunks for hashing

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def video_key(path: Path) -> str:
    """Return a hex crc32 fingerprint of the full video file."""
    start = perf_counter()
    crc = 0
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    # Mix file size into the key to distinguish truncated vs. full files
    crc = zlib.crc32(size.to_bytes(8, "little"), crc)
    key = f"{crc & 0xFFFFFFFF:08x}"
    elapsed = perf_counter() - start
    logger.debug(
        "Hashed %s (%.1f MiB) in %.3fs -> key %s (%s)",
        path,
        size / (1 << 20),
        elapsed,
        key,
        cache_path(key),
    )
    return key


def cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "rat_tracer_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.pkl"


def save_progress(history: CoverageHistory, key: str) -> None:
    dest = cache_path(key)
    tmp = dest.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(history, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dest)
        logger.info("Progress saved to %s", dest)
    except Exception:
        logger.exception("Failed to save progress to %s", dest)
        tmp.unlink(missing_ok=True)


def load_progress(key: str) -> CoverageHistory | None:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, CoverageHistory):
            logger.warning("Cache file %s contains unexpected type; ignoring", p)
            return None
        logger.info("Resumed progress from %s (%d frames)", p, len(obj))
        return obj
    except Exception:
        logger.warning("Corrupt or unreadable cache at %s; ignoring", p)
        return None
