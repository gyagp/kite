"""Balanced, isolated longer-decode comparison; reports retain every measured sample."""

import json
import sys
import time
from pathlib import Path

from run_gdn_clean_comparison import command

from evo.measurement_guard import require_idle_host


def main():
    output = Path("artifacts/comparison-gdn-extended")
    output.mkdir(parents=True, exist_ok=False)
    runtime = "artifacts/runtimes/ort-webgpu-gdn-subgroup/onnxruntime_providers_webgpu.dll"
    previous = "E:/workspace/project/webgfx-china/ai-models/Qwen3.5-4B/onnx-webgpu"
    candidate = "artifacts/qwen35-4b-previous-gdn"
    require_idle_host()
    for name, package, library in [
        ("previous-a", previous, None),
        ("gdn-a", candidate, runtime),
        ("gdn-b", candidate, runtime),
        ("previous-b", previous, None),
    ]:
        (output / "status.json").write_text(
            json.dumps(
                {
                    "state": "timing",
                    "block": name,
                    "updated": time.time(),
                }
            )
        )
        args = [
            "-u",
            "-m",
            "evo",
            "benchmark",
            package,
            "--cases",
            "benchmarks/qwen35-webgfx.json",
            "--output",
            str(output / f"{name}.json"),
            "--repeats",
            "3",
            "--warmup",
            "2",
            "--new-tokens",
            "512",
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
        command(args, output / f"{name}.log", timeout=300)
        print(name, "completed and exited", flush=True)
    (output / "status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "updated": time.time(),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        raise
