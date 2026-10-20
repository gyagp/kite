"""Detect competing build/test work before collecting native GPU timing evidence."""

import os


def competing_workloads():
    import psutil

    current = psutil.Process()
    own_tree = {os.getpid(), *(p.pid for p in current.parents())}
    executables = {
        "cl.exe",
        "link.exe",
        "ninja.exe",
        "ctest.exe",
        "onnxruntime_test_all.exe",
        "onnxruntime_provider_test.exe",
        "dawn_end2end_tests.exe",
        "model_benchmark.exe",
    }
    scripts = (
        "check_qwen_cached.py",
        "check_qwen_quality.py",
        "check_qwen_generation.py",
        "check_qwen_pair.py",
        "benchmark_interleaved.py",
        "check_gdn_kernel.py",
        "embedding_validate",
        "embedding_benchmark",
        "-m evo benchmark",
        "-m evo optimize",
        "-m evo convert",
        "-m evo quantize",
        "onnxruntime_genai.models.builder",
        "embedding_export",
        "evo.image_generate",
        "evo.image_prepare",
        "evo.image_qwen21",
        "evo image-generate",
        "evo image-prepare",
        "evo image-export",
    )
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = process.info
            if info["pid"] in own_tree:
                continue
            name = (info["name"] or "").lower()
            command = " ".join(info["cmdline"] or [])
            active_msbuild = name == "msbuild.exe" and "/nodemode" not in command.lower()
            active_git_pack = name == "git.exe" and any(
                arg in {"index-pack", "pack-objects"} for arg in (info["cmdline"] or [])
            )
            if (
                name in executables
                or active_msbuild
                or active_git_pack
                or name.startswith("python")
                and any(s in command for s in scripts)
            ):
                found.append(
                    {
                        "pid": info["pid"],
                        "name": info["name"],
                        "started": info["create_time"],
                        "command": command[:800],
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def require_idle_host():
    workloads = competing_workloads()
    if workloads:
        names = ", ".join(f"{p['name']} (PID {p['pid']})" for p in workloads)
        raise RuntimeError("Timing requires an idle host; competing workloads: " + names)
