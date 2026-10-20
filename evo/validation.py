"""Optional CPU reference comparison; this does not establish WebGPU support."""

import math
from pathlib import Path

import numpy as np


def compare_outputs(actual: dict, expected: dict, rtol: float, atol: float) -> dict:
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("Tolerances must be finite and non-negative")
    if set(actual) != set(expected):
        raise ValueError("Expected output names must exactly match model output names")
    reports = {}
    for name, output in actual.items():
        reference = expected[name]
        if output.shape != reference.shape:
            reports[name] = {"passed": False, "reason": "shape mismatch"}
        elif output.dtype.kind in "biufc" and reference.dtype.kind in "biufc":
            finite = bool(np.isfinite(output).all() and np.isfinite(reference).all())
            # Integers and booleans need exact equality; floating tolerances could
            # otherwise hide incorrect token IDs or precision loss above 2**53.
            exact = output.dtype.kind in "biu" and reference.dtype.kind in "biu"
            passed = (
                np.array_equal(output, reference)
                if exact
                else np.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=False)
            )
            reports[name] = {"passed": bool(finite and passed), "finite": finite}
            if finite and not exact:
                difference = np.abs(output.astype(np.complex128) - reference.astype(np.complex128))
                maximum = float(difference.max(initial=0))
                reports[name]["max_absolute_error"] = maximum if math.isfinite(maximum) else None
        else:
            reports[name] = {"passed": bool(np.array_equal(output, reference))}
    return {
        "accuracy": "passed" if all(r["passed"] for r in reports.values()) else "failed",
        "rtol": rtol,
        "atol": atol,
        "outputs": reports,
    }


def validate_cpu(model: Path, inputs: Path, expected: Path, rtol: float, atol: float) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError("Install the validation extra: uv sync --extra validation") from exc
    session = ort.InferenceSession(str(model.resolve()), providers=["CPUExecutionProvider"])
    with np.load(inputs, allow_pickle=False) as data:
        feeds = {name: data[name] for name in data.files}
    required = {value.name for value in session.get_inputs()}
    optional = {value.name for value in session.get_overridable_initializers()}
    if not required <= feeds.keys() or feeds.keys() - required - optional:
        raise ValueError(f"Input NPZ names must match required inputs {sorted(required)}")
    names = [output.name for output in session.get_outputs()]
    actual = dict(zip(names, session.run(names, feeds), strict=True))
    with np.load(expected, allow_pickle=False) as data:
        reference = {name: data[name] for name in data.files}
    return {
        "model": str(model.resolve()),
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "webgpu": "not_run",
        "performance": "not_run",
        **compare_outputs(actual, reference, rtol, atol),
    }
