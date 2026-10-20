"""Summarize ORT WebGPU shader timestamps separately from CPU dispatch events."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def summarize_trace(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(events, dict):
        events = events.get("traceEvents", [])
    shaders = defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
    operators = defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
    providers = Counter()
    for event in events:
        args = event.get("args", {})
        if args.get("provider"):
            providers[args["provider"]] += 1
        duration = event.get("dur")
        if (
            event.get("cat") == "Node"
            and args.get("provider")
            and event.get("name", "").endswith("_kernel_time")
            and isinstance(duration, (int, float))
            and math.isfinite(duration)
            and duration >= 0
        ):
            row = operators[args.get("op_name", event["name"])]
            row["calls"] += 1
            row["total_ms"] += duration / 1000
        if (
            event.get("cat") != "Api"
            or "cache_key" not in args
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            continue
        name = event.get("name", "Unknown shader").split("&")[-1]
        row = shaders[name]
        row["calls"] += 1
        row["total_ms"] += duration / 1000
    rows = sorted(
        ({"name": name, **values} for name, values in shaders.items()),
        key=lambda row: -row["total_ms"],
    )
    total = sum(row["total_ms"] for row in rows)
    return {
        "gpu_total_ms": total,
        "gpu_events": sum(row["calls"] for row in rows),
        "shaders": rows,
        "cpu_nodes_by_provider": dict(providers),
        "cpu_operators": [{"name": name, **row} for name, row in operators.items()],
    }


def save_trace(filename, folder, stage):
    if not filename:
        return {"error": "The runtime did not return a profiling trace.", "gpu_events": 0}
    target = Path(folder) / f"{stage}-trace.json"
    try:
        Path(filename).replace(target)
        return {"trace": target.name, **summarize_trace(target)}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {
            "error": str(error),
            "gpu_events": 0,
            **({"trace": target.name} if target.is_file() else {}),
        }


def build_profile(phases, job_id):
    rows, traces, warnings = [], [], []
    gpu_events = 0
    for stage in ("prefix", "denoise"):
        profile = phases.get(stage, {}).get("profile", {})
        gpu_events += profile.get("gpu_events", 0)
        rows.extend({"stage": stage, **row} for row in profile.get("shaders", []))
        if profile.get("trace"):
            traces.append({"stage": stage, "url": f"/api/images/{job_id}/{profile['trace']}"})
        if profile.get("error"):
            warnings.append(f"{stage}: {profile['error']}")
        elif not profile.get("gpu_events"):
            warnings.append(f"{stage}: this runtime did not provide GPU shader timestamps.")
    total = sum(row["total_ms"] for row in rows)
    return {
        "gpu_events": gpu_events,
        "gpu_total_ms": total,
        "top_shaders": [
            {**row, "percent": row["total_ms"] / total * 100 if total else 0}
            for row in sorted(rows, key=lambda row: -row["total_ms"])[:20]
        ],
        "traces": traces,
        "warnings": warnings,
        "summary_url": f"/api/images/{job_id}/profile.json",
        "scope": "Prefix and denoising GPU shader timings. Text/reference encoding and VAE use the phase wall times. Includes first-use work; profiling adds overhead. GPU and CPU trace durations must not be added together.",
    }
