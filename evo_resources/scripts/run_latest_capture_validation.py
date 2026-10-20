"""Package the requested upstream build and qualify Qwen capture/FP16 experiments."""

import json
import time
from pathlib import Path

from run_gdn_clean_comparison import command

from evo.measurement_guard import competing_workloads
from evo.pipeline import write_json


def main():
    source = Path("E:/workspace/project/ws-ort/onnxruntime")
    output = Path("artifacts/comparison-latest-capture")
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    build_exit = source / "build/evo-webgpu-latest-build.exit"
    started = time.monotonic()
    while not build_exit.exists():
        write_json(status, {"state": "waiting_for_build", "updated": time.time()})
        if time.monotonic() - started > 7200:
            raise RuntimeError("Build did not complete within two hours")
        time.sleep(5)
    if build_exit.read_text().strip() != "0":
        raise RuntimeError("Upstream build failed; see source build log")
    quiet_since = None
    while True:
        activity = competing_workloads()
        write_json(
            status, {"state": "waiting_for_idle", "activity": activity, "updated": time.time()}
        )
        if activity:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= 20:
            break
        if time.monotonic() - started > 10800:
            raise RuntimeError("Host remained busy after build")
        time.sleep(5)
    package = Path("artifacts/runtimes/ort-webgpu-main-86d60730c1")
    if not package.exists():
        command(
            [
                "scripts/package_webgpu.py",
                "--source",
                str(source),
                "--revision",
                "86d60730c14f546ba1a4ba06b4887b53285117a5",
                "--build",
                str(source / "build/evo-webgpu-latest/Release"),
                "--build-log",
                str(source / "build/evo-webgpu-latest-build.log"),
                "--build-exit",
                str(build_exit),
                "--dxc-package",
                "artifacts/runtimes/ort-webgpu-gdn-subgroup",
                "--output",
                str(package),
            ],
            output / "package.log",
        )
    library = str(package / "onnxruntime_providers_webgpu.dll")
    write_json(status, {"state": "kernel_validation", "updated": time.time()})
    kernel_report = Path("artifacts/gdn-kernel-validation-main-86d60730c1/summary.json")
    if kernel_report.exists():
        from evo.pipeline import hash_file

        kernel = json.loads(kernel_report.read_text())
        if not kernel["passed"] or kernel["runtime_library"]["sha256"] != hash_file(Path(library)):
            raise RuntimeError("Existing kernel validation does not match this runtime")
    else:
        command(
            [
                "-u",
                "scripts/check_gdn_kernel.py",
                "--ep-library",
                library,
                "--output",
                str(kernel_report.parent),
            ],
            output / "kernel.log",
        )
    Path(".evo").mkdir(exist_ok=True)
    write_json(
        Path(".evo/runtime.json"),
        {
            "webgpu_library": "../" + library,
            "source_revision": "86d60730c14f546ba1a4ba06b4887b53285117a5",
            "validation": "../artifacts/gdn-kernel-validation-main-86d60730c1/summary.json",
        },
    )
    write_json(status, {"state": "capture_proof", "updated": time.time()})
    gdn = "artifacts/qwen35-4b-gdn-mlp"
    fp16 = "artifacts/qwen35-4b-gdn-mlp-fp16-cache"
    linear = "artifacts/qwen35-4b-linear-fp16-mlp"
    command(
        [
            "-u",
            "scripts/check_qwen_pair.py",
            gdn,
            gdn,
            "--ep-library",
            library,
            "--candidate-capture",
            "--cases",
            "benchmarks/qwen35-state-stress.json",
            "--steps",
            "32",
            "--output",
            str(output / "capture-vs-uncaptured.json"),
            "--profile",
            str(output / "capture-trace"),
        ],
        output / "capture-proof.log",
    )
    for name, model in [("gdn-fp32", gdn), ("gdn-fp16-cache", fp16), ("linear-fp16-mlp", linear)]:
        write_json(status, {"state": "cached_quality", "model": name, "updated": time.time()})
        command(
            [
                "-u",
                "scripts/check_qwen_cached.py",
                model,
                "--ep-library",
                library,
                "--fixtures",
                "artifacts/qwen35-cached-reference",
                "--graph-capture",
                "--output",
                str(output / (name + "-cached.json")),
            ],
            output / (name + "-cached.log"),
        )
    write_json(status, {"state": "fp16_sustained_quality", "updated": time.time()})
    command(
        [
            "-u",
            "scripts/check_qwen_pair.py",
            gdn,
            fp16,
            "--ep-library",
            library,
            "--reference-capture",
            "--candidate-capture",
            "--cases",
            "benchmarks/qwen35-state-stress.json",
            "--steps",
            "512",
            "--output",
            str(output / "fp16-vs-fp32-captured.json"),
        ],
        output / "fp16-sustained.log",
        timeout=600,
    )
    for name, model in [("fp16-cache", fp16), ("linear-fp16", linear)]:
        write_json(status, {"state": "captured_timing", "candidate": name, "updated": time.time()})
        command(
            [
                "-u",
                "scripts/benchmark_interleaved.py",
                gdn,
                model,
                "--ep-library",
                library,
                "--graph-capture",
                "--repeats",
                "6",
                "--new-tokens",
                "128",
                "--output",
                str(Path("artifacts") / ("comparison-latest-" + name + "-capture")),
            ],
            output / (name + "-timing.log"),
            timeout=600,
        )
    write_json(status, {"state": "completed", "updated": time.time()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(
            Path("artifacts/comparison-latest-capture/status.json"),
            {"state": "failed", "error": str(error), "updated": time.time()},
        )
        raise
