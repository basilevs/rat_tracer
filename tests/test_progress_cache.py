import pickle
from threading import Lock

import numpy as np
from rat_tracer.coverage import CoverageHistory, MaskFrame
from rat_tracer.progress_cache import load_progress, save_progress, video_key

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_history(n: int = 3, h: int = 4, w: int = 6) -> CoverageHistory:
    hist = CoverageHistory()
    rng = np.random.default_rng(42)
    for _ in range(n):
        hist.append(rng.integers(0, 2, size=(h, w), dtype=bool).reshape(h, w))
    return hist


# ── video_key ─────────────────────────────────────────────────────────────────


def test_video_key_stable(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"abcdefgh" * 1024)
    assert video_key(f) == video_key(f)


def test_video_key_differs_on_content(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"aaa" * 512)
    b.write_bytes(b"bbb" * 512)
    assert video_key(a) != video_key(b)


def test_video_key_differs_on_size(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 101)
    assert video_key(a) != video_key(b)


# ── save / load round-trip ────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("rat_tracer.progress_cache.cache_dir", lambda: tmp_path)

    hist = _make_history(5)
    key = "testkey"
    save_progress(hist, key)

    restored = load_progress(key)
    assert restored is not None
    assert len(restored) == len(hist)
    for i in range(len(hist)):
        np.testing.assert_array_equal(restored[i], hist[i])


def test_load_returns_none_for_missing_key(tmp_path, monkeypatch):
    monkeypatch.setattr("rat_tracer.progress_cache.cache_dir", lambda: tmp_path)
    assert load_progress("no_such_key") is None


def test_load_returns_none_for_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr("rat_tracer.progress_cache.cache_dir", lambda: tmp_path)
    (tmp_path / "bad.pkl").write_bytes(b"\x00" * 64)
    assert load_progress("bad") is None


# ── CoverageHistory pickle ────────────────────────────────────────────────────


def test_coverage_history_pickle_round_trip():
    hist = _make_history(4)
    data = pickle.dumps(hist)
    restored = pickle.loads(data)
    assert isinstance(restored._lock, Lock)
    assert len(restored) == len(hist)
    for i in range(len(hist)):
        np.testing.assert_array_equal(restored[i], hist[i])


# ── replace_with ──────────────────────────────────────────────────────────────


def test_replace_with_copies_state():
    src = _make_history(3)
    dst = CoverageHistory()
    dst.replace_with(src)
    assert len(dst) == 3
    for i in range(3):
        np.testing.assert_array_equal(dst[i], src[i])


def test_replace_with_allows_continued_appends():
    src = _make_history(3, h=4, w=6)
    dst = CoverageHistory()
    dst.replace_with(src)
    new_mask: MaskFrame = np.zeros((4, 6), dtype=bool)
    new_mask[0, 0] = True
    dst.append(new_mask)
    assert len(dst) == 4
