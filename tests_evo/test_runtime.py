import json

import pytest

from evo.runtime import register_webgpu, resolve_webgpu_library


def test_missing_custom_runtime_fails_before_registration(tmp_path):
    with pytest.raises(ValueError, match="WebGPU library not found"):
        register_webgpu(tmp_path / "missing.dll")


def test_modified_custom_runtime_is_rejected_before_loading(tmp_path):
    library = tmp_path / "provider.dll"
    library.write_bytes(b"modified library")
    (tmp_path / "evo-runtime.json").write_text(
        json.dumps({"dlls": {"provider.dll": {"sha256": "wrong-hash"}}})
    )
    with pytest.raises(ValueError, match="differs from its build manifest"):
        register_webgpu(library)


def test_project_runtime_selection_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evo.runtime.asset_root", lambda: tmp_path / "assets")
    monkeypatch.delenv("EVO_WEBGPU_LIBRARY", raising=False)
    installed = tmp_path / "installed.dll"
    assert resolve_webgpu_library(None, installed) == (installed, "installed")
    (tmp_path / ".evo").mkdir()
    (tmp_path / ".evo/runtime.json").write_text(json.dumps({"webgpu_library": "../built.dll"}))
    assert resolve_webgpu_library(None, installed) == (tmp_path / "built.dll", "project")
    monkeypatch.setenv("EVO_WEBGPU_LIBRARY", str(tmp_path / "env.dll"))
    assert resolve_webgpu_library(None, installed) == (tmp_path / "env.dll", "environment")
    assert resolve_webgpu_library(tmp_path / "explicit.dll", installed) == (
        tmp_path / "explicit.dll",
        "explicit",
    )
