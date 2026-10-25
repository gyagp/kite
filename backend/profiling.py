"""Collect per-request native chat/embedding profiles into local downloadable reports."""

import json
from pathlib import Path

from .image_profile import summarize_trace


def finish_profile(directory, job_id, kind):
    directory = Path(directory)
    shaders, operators, traces, warnings = [], [], [], []
    gpu_events = 0
    for index, source in enumerate(sorted(directory.glob("*.json")), 1):
        name = f"trace-{index}.json"
        target = directory / name
        source.replace(target)
        stage = f"Session {index}"
        traces.append({"stage": stage, "url": f"/api/profiles/{job_id}/{name}"})
        try:
            summary = summarize_trace(target)
            shaders.extend({"stage": stage, **row} for row in summary["shaders"])
            operators.extend({"stage": stage, **row} for row in summary["cpu_operators"])
            gpu_events += summary["gpu_events"]
        except (ValueError, TypeError, AttributeError, OSError) as error:
            warnings.append(f"{stage}: {error}")
    if not traces:
        warnings.append("The selected runtime did not return a profiling trace.")
    if not gpu_events:
        warnings.append(
            "GPU shader timestamps are unavailable in the selected runtime. CPU operator durations include dispatch and GPU waits."
        )
    total = sum(row["total_ms"] for row in shaders)
    cpu_total = sum(row["total_ms"] for row in operators)

    def ranked(rows, elapsed):
        return [
            {**row, "percent": row["total_ms"] / elapsed * 100 if elapsed else 0}
            for row in sorted(rows, key=lambda row: -row["total_ms"])[:20]
        ]

    report = {
        "gpu_events": gpu_events,
        "gpu_total_ms": total,
        "top_shaders": ranked(shaders, total),
        "cpu_total_ms": cpu_total,
        "top_cpu_ops": ranked(operators, cpu_total),
        "traces": traces,
        "warnings": warnings,
        "summary_url": f"/api/profiles/{job_id}/profile.json",
        "scope": (
            "SAM 3 video profile from a fresh model load. Session traces cover vision encoding, mask decoding, memory encoding, and memory attention. "
            if kind == "track"
            else f"{kind.capitalize()} profile from a fresh model load. Includes preprocessing, prefill/decode or embedding execution in the run metrics. "
        )
        + "CPU operator times include dispatch and waits; do not add them to GPU times. Profiling adds overhead. The profiled worker is released after this run.",
    }
    (directory / "profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
