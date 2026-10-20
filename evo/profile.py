"""Profile an existing ONNX model with explicit NPZ inputs on native WebGPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from evo.pipeline import hash_file, write_json


def profile_model(model: Path, inputs: Path, output: Path, runs=3, ep_library=None):
    import onnxruntime as ort

    from evo.runtime import register_webgpu

    if runs < 1:
        raise ValueError("runs must be positive")
    output.mkdir(parents=True, exist_ok=False)
    devices, runtime_library = register_webgpu(ep_library)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str((output / "ort").resolve())
    options.add_provider_for_devices(devices[:1], {"powerPreference": "high-performance"})
    session = ort.InferenceSession(str(model.resolve()), sess_options=options)
    with np.load(inputs, allow_pickle=False) as data:
        feeds = {name: data[name] for name in data.files}
    for _ in range(runs):
        session.run(None, feeds)
    profile_path = session.end_profiling()
    result = {
        "model_sha256": hash_file(model),
        "inputs_sha256": hash_file(inputs),
        "profile": profile_path,
        "runs": runs,
        "providers": session.get_providers(),
        "device": devices[0].device.metadata,
        "runtime_library": runtime_library,
        "timing": "ORT trace event durations; not assumed to equal GPU shader time",
    }
    write_json(output / "profile-info.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ep-library", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            profile_model(args.model, args.inputs, args.output, args.runs, args.ep_library),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
