from types import SimpleNamespace

import psutil
import pytest

from evo.measurement_guard import competing_workloads, require_idle_host


def test_guard_rejects_other_gpu_checks_but_ignores_own_parent_and_ui(monkeypatch):
    monkeypatch.setattr(
        psutil, "Process", lambda: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=11)])
    )
    items = [
        (11, "python.exe", ["python", "-m", "evo", "benchmark"]),
        (12, "python.exe", ["python", "-m", "evo", "ui"]),
        (13, "onnxruntime_provider_test.exe", ["test"]),
        (14, "python.exe", ["python", "check_qwen_cached.py"]),
        (15, "git.exe", ["git", "index-pack", "--stdin"]),
        (16, "git.exe", ["git", "status"]),
    ]
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda _: [
            SimpleNamespace(info={"pid": pid, "name": name, "cmdline": cmd, "create_time": 1})
            for pid, name, cmd in items
        ],
    )
    assert {p["pid"] for p in competing_workloads()} == {13, 14, 15}
    with pytest.raises(RuntimeError, match="competing workloads"):
        require_idle_host()
