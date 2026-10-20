"""Check native GatedDeltaNet against independent recurrent equations and state reuse."""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from evo.pipeline import write_json
from evo.runtime import register_webgpu


def reference(feeds, normalize=True):
    q, k, v = [feeds[name].astype(np.float64) for name in ("q", "k", "v")]
    state = feeds["state"].astype(np.float64).copy()
    batch, hv, dv, dk = state.shape
    hq = q.shape[-2]
    q, k, v = [x.reshape(batch, -1, x.shape[-2], x.shape[-1]) for x in (q, k, v)]
    if normalize:
        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-12)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-12)
    q, k = [np.repeat(x, hv // hq, axis=2) for x in (q, k)]
    a = feeds["a"].astype(np.float64).reshape(batch, -1, hv)
    b = feeds["b"].astype(np.float64).reshape(batch, -1, hv)
    decay = np.exp(
        -np.exp(feeds["a_log"].astype(np.float64)) * np.logaddexp(0, a + feeds["dt_bias"])
    )
    beta = 1 / (1 + np.exp(-b))
    output = np.empty_like(v)
    for token in range(q.shape[1]):
        state *= decay[:, token, :, None, None]
        retrieved = np.einsum("bhvk,bhk->bhv", state, k[:, token])
        delta = beta[:, token, :, None] * (v[:, token] - retrieved)
        state += delta[..., None] * k[:, token, :, None, :]
        output[:, token] = np.einsum("bhvk,bhk->bhv", state, q[:, token]) / np.sqrt(dk)
    return output.reshape(feeds["v"].shape), state


def graph(path, dtype, rank4, hq, hv, dk, dv, normalize=True):
    token_axes = ["batch", "sequence"] if rank4 else ["tokens"]
    type_id = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    specs = [
        ("q", type_id, token_axes + [hq, dk]),
        ("k", type_id, token_axes + [hq, dk]),
        ("v", type_id, token_axes + [hv, dv]),
        ("a", TensorProto.FLOAT, token_axes + [hv]),
        ("b", TensorProto.FLOAT, token_axes + [hv]),
        ("state", TensorProto.FLOAT, ["batch", hv, dv, dk]),
        ("a_log", TensorProto.FLOAT, [hv]),
        ("dt_bias", TensorProto.FLOAT, [hv]),
    ]
    g = helper.make_graph(
        [
            helper.make_node(
                "GatedDeltaNet",
                ["q", "k", "v", "", "a", "b", "state", "a_log", "dt_bias"],
                ["output", "next_state"],
                domain="com.microsoft",
                name="gdn",
                gate_activation="qwen",
                beta_activation="sigmoid",
                qk_l2_norm=int(normalize),
                update_rule="gated_delta",
                scale=0.0,
                chunk_size=64,
            )
        ],
        "gdn-check",
        [helper.make_tensor_value_info(*s) for s in specs],
        [
            helper.make_tensor_value_info("output", type_id, token_axes + [hv, dv]),
            helper.make_tensor_value_info("next_state", TensorProto.FLOAT, ["batch", hv, dv, dk]),
        ],
    )
    onnx.save(
        helper.make_model(
            g,
            ir_version=10,
            opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)],
        ),
        path,
    )


def run(library, output):
    output.mkdir(parents=True, exist_ok=False)
    devices, runtime = register_webgpu(library)
    report = {
        "runtime_library": runtime,
        "reference": "NumPy float64 recurrence, FP32 gate parameters, V-major state",
        "device": devices[0].device.metadata,
        "results": [],
    }
    rng = np.random.default_rng(20261011)
    for label, dtype, rank4, batch, hq, hv, dk, dv, normalize in [
        ("qwen-f16-rank4", np.float16, True, 1, 16, 32, 128, 128, True),
        ("qwen-f16-external-norm", np.float16, True, 1, 16, 32, 128, 128, False),
        ("f32-rank4-batch2", np.float32, True, 2, 2, 4, 32, 48, True),
        ("f16-rank3", np.float16, False, 1, 2, 4, 64, 64, True),
    ]:
        path = output / (label + ".onnx")
        graph(path, dtype, rank4, hq, hv, dk, dv, normalize)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(output / label)
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(
            devices[:1],
            {
                "dawnBackendType": "D3D12",
                "validationMode": "basic",
                "enableGraphCapture": "0",
                "powerPreference": "high-performance",
            },
        )
        session = ort.InferenceSession(str(path.resolve()), sess_options=options)
        state = rng.normal(0, 0.02, (batch, hv, dv, dk)).astype(np.float32)
        expected_state = state.copy()
        for step, length in enumerate((17, 1, 1, 33, 1)):
            lead = (batch, length) if rank4 else (batch * length,)
            feeds = {
                name: rng.normal(0, 0.4, lead + tail).astype(dtype)
                for name, tail in [("q", (hq, dk)), ("k", (hq, dk)), ("v", (hv, dv))]
            }
            feeds.update(
                a=rng.normal(-1, 0.2, lead + (hv,)).astype(np.float32),
                b=rng.normal(0, 0.5, lead + (hv,)).astype(np.float32),
                state=state,
                a_log=np.full(hv, -1, np.float32),
                dt_bias=np.full(hv, 0.1, np.float32),
            )
            if not normalize:
                for name in ("q", "k"):
                    value = feeds[name].astype(np.float32)
                    feeds[name] = (value / np.linalg.norm(value, axis=-1, keepdims=True)).astype(
                        dtype
                    )
            expected, expected_state = reference({**feeds, "state": expected_state}, normalize)
            actual, state = session.run(None, feeds)
            max_error = float(np.max(np.abs(actual - expected)))
            state_error = float(np.max(np.abs(state - expected_state)))
            passed = bool(
                np.isfinite(actual).all()
                and np.isfinite(state).all()
                and max_error < 5e-4
                and state_error < 5e-4
            )
            report["results"].append(
                {
                    "case": label,
                    "step": step,
                    "tokens": length,
                    "max_output_error": max_error,
                    "max_state_error": state_error,
                    "passed": passed,
                }
            )
            print(label, step, max_error, state_error, passed, flush=True)
        trace = json.loads(Path(session.end_profiling()).read_text())
        counts = Counter(
            e["args"].get("provider")
            for e in trace
            if e.get("args", {}).get("op_name") == "GatedDeltaNet"
        )
        report.setdefault("execution", {})[label] = dict(counts)
        if counts.get("WebGpuExecutionProvider") != 5:
            raise RuntimeError("Did not observe all five GDN executions on native WebGPU")
        del session
        gc.collect()
    report["passed"] = all(r["passed"] for r in report["results"])
    write_json(output / "summary.json", report)
    if not report["passed"]:
        raise RuntimeError("GDN numerical validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.ep_library, args.output)
