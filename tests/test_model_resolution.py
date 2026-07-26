from pathlib import Path

import pytest

from rat_tracer.lib import MODEL_ENV_VAR, model_path


def test_override_returns_literal_path(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "/tmp/some/local/model.pt")
    assert model_path() == Path("/tmp/some/local/model.pt")


def test_override_takes_precedence_without_network(monkeypatch, tmp_path):
    local = tmp_path / "weights.pt"
    local.write_bytes(b"")
    monkeypatch.setenv(MODEL_ENV_VAR, str(local))
    # Would raise if the Hugging Face download were attempted.
    monkeypatch.setattr(
        "rat_tracer.lib.hf_hub_download",
        lambda *args, **kwargs: pytest.fail("network fetch should not happen with override set"),
    )
    assert model_path() == local


@pytest.mark.network
def test_hf_resolve_returns_existing_file(monkeypatch):
    # A warm cache counts as success; do not wipe HF_HOME or force a redownload.
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    resolved = model_path()
    assert resolved.is_file()


def test_offline_fallback_uses_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    cached = tmp_path / "rat_tracer.pt"
    cached.write_bytes(b"")

    def fake_download(*args, local_files_only=False, **kwargs):
        if not local_files_only:
            raise ConnectionError("simulated offline")
        return str(cached)

    monkeypatch.setattr("rat_tracer.lib.hf_hub_download", fake_download)
    assert model_path() == cached
