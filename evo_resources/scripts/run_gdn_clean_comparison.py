"""Wait for a quiet device, verify validator teardown, then collect paired Qwen timing."""

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from evo.measurement_guard import competing_workloads, require_idle_host


def command(args, log_path, timeout=300, check_idle=True):
    with log_path.open("wb") as log:
        process = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            process.wait(timeout=20)
            raise RuntimeError(f"Own test process exceeded its limit: {log_path}") from None
    if code:
        raise RuntimeError(f"Test exited {code}: {log_path}")
    if check_idle:
        require_idle_host()


def complete_report(path):
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    return (
        data.get("measurement_status") != "contaminated"
        and data.get("require_idle_host") is True
        and data.get("new_tokens") == 128
        and data.get("max_length") == 8192
        and len(data.get("results", [])) == 3
        and all(len(row.get("samples", [])) == 5 for row in data["results"])
    )


def main(resume=False):
    output = Path("artifacts/comparison-gdn-clean")
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    quiet_since = None
    previous = None
    started = time.monotonic()
    while True:
        activity = competing_workloads()
        current = [(p["pid"], p["name"]) for p in activity]
        if current != previous:
            print(
                "Waiting for competing work:" if current else "Checking quiet interval:",
                current,
                flush=True,
            )
            previous = current
        status.write_text(
            json.dumps(
                {"state": "waiting_for_idle", "activity": activity, "updated": time.time()},
                indent=2,
            )
        )
        if activity:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= 30:
            break
        if time.monotonic() - started > 7200:
            raise RuntimeError("Device did not become idle within two hours")
        time.sleep(5)
    runtime = "artifacts/runtimes/ort-webgpu-gdn-subgroup/onnxruntime_providers_webgpu.dll"
    model = "artifacts/qwen35-4b-previous-gdn"
    status.write_text(json.dumps({"state": "checking_cached_decode", "updated": time.time()}))
    quality_complete = (output / "quality.log").is_file() and "Native model released" in (
        output / "quality.log"
    ).read_text()
    if not (resume and quality_complete and complete_report(output / "previous-a.json")):
        command(
            [
                "-u",
                "scripts/check_qwen_cached.py",
                model,
                "--fixtures",
                "artifacts/qwen35-cached-reference",
                "--output",
                str(output / "gdn-cached.json"),
                "--ep-library",
                runtime,
            ],
            output / "quality.log",
            timeout=180,
        )
    status.write_text(json.dumps({"state": "timing", "updated": time.time()}))
    previous_model = r"E:/workspace/project/webgfx-china/ai-models/Qwen3.5-4B/onnx-webgpu"
    for name, part, package, library in [
        ("previous", "a", previous_model, None),
        ("gdn", "a", model, runtime),
        ("gdn", "b", model, runtime),
        ("previous", "b", previous_model, None),
    ]:
        report_path = output / f"{name}-{part}.json"
        if resume and complete_report(report_path):
            print(name, part, "already completed; preserving report", flush=True)
            continue
        if report_path.exists():
            report_path.replace(output / f"{name}-{part}.interrupted.json")
            log_path = output / f"{name}-{part}.log"
            if log_path.exists():
                log_path.replace(output / f"{name}-{part}.interrupted.log")
        args = [
            "-m",
            "evo",
            "benchmark",
            package,
            "--cases",
            "benchmarks/qwen35-webgfx.json",
            "--output",
            str(output / f"{name}-{part}.json"),
            "--repeats",
            "5",
            "--warmup",
            "3",
            "--new-tokens",
            "128",
            "--max-length",
            "8192",
            "--reuse-generator",
            "--synchronization",
            "is_done",
            "--require-idle",
            "--provider-options",
            '{"dawnBackendType":"D3D12","validationMode":"basic","enableGraphCapture":"0"}',
        ]
        if library:
            args += ["--ep-library", library]
        command(args, output / f"{name}-{part}.log")
        print(name, part, "completed and exited", flush=True)
    status.write_text(json.dumps({"state": "completed", "updated": time.time()}))


if __name__ == "__main__":
    try:
        main(resume="--resume" in sys.argv[1:])
    except Exception as error:
        Path("artifacts/comparison-gdn-clean/status.json").write_text(
            json.dumps({"state": "failed", "error": str(error), "updated": time.time()})
        )
        raise
