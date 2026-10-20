"""ORT schema knowledge and versioned WebGPU optimization guidance.

Schema availability is intentionally separate from runtime kernel availability.
"""

import argparse
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

ORT_COMMIT = "8fc9fd597453baae1ad73379c313f0586d812867"
ORT_SOURCE = f"https://github.com/microsoft/onnxruntime/blob/{ORT_COMMIT}/"
MOBIUS_SOURCE = (
    "https://github.com/onnxruntime/mobius/blob/88f7d045388ccdcd98e8a23f4abcbcdfd39f65e6/"
)
GENAI_SOURCE = (
    "https://github.com/microsoft/onnxruntime-genai/blob/5c19002edce50beaa3a5c7eeefd8fe5269ee8d19/"
)


def note(family, summary, constraints, path, strategy="Runtime or exporter selected"):
    return {
        "family": family,
        "summary": summary,
        "constraints": constraints,
        "strategy": strategy,
        "webgpu": "Source implementation reviewed; verify target build",
        "source": ORT_SOURCE + path,
    }


KNOWLEDGE = {
    "MatMulNBits": note(
        "Quantization",
        "Blockwise low-bit matrix multiplication.",
        "Weight packing, bits, block size, activation dtype and accuracy level select kernels.",
        "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc",
    ),
    "MatMulNBitsMlp": note(
        "Fusion",
        "Fuses quantized gate/up projections and SiLU multiplication.",
        "Reviewed WebGPU fast path: symmetric INT4, block 32, no explicit zero points. "
        "Preserve input/output, activation and optional normalization contracts. Subgroup, "
        "DP4A or wide-tile kernel selection can bypass the fused decode fast path.",
        "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc",
        "Kite can emit this op from a compatible packed gate/up + Split + SiLU pattern; "
        "ORT can also form it from separate projections.",
    ),
    "MatMulNBitsQkv": note(
        "Fusion",
        "Fuses normalization and quantized Q/K/V projections.",
        "Shared normalized input, compatible projection packing, INT4/block-32 fast path.",
        "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_qkv.cc",
        "Preserve patterns recognized by ORT; Mobius/GenAI provide compatible projections.",
    ),
    "GroupQueryAttention": note(
        "Attention",
        "Grouped-query attention with KV-cache handling.",
        "Layout, head counts, cache dtype, rotary settings and masking must match the kernel.",
        "onnxruntime/contrib_ops/webgpu/bert/group_query_attention.cc",
        "Mobius GQA fusion and GenAI architecture-aware export.",
    ),
    "GatedDeltaNet": note(
        "Recurrent attention",
        "Recurrent attention with Qwen gates and Q/K normalization.",
        "FP32 recurrent state and its layout are part of the ABI. GenAI 0.17.1 restricts "
        "its export path to CUDA; do not merely rename LinearAttention.",
        "onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.cc",
        "Existing WebGPU kernel; exporter integration requires separate validation.",
    ),
    "LinearAttention": note(
        "Recurrent attention",
        "Stateful linear attention used by hybrid decoders.",
        "Update rule, head geometry and recurrent-state layout must be preserved.",
        "onnxruntime/contrib_ops/webgpu/bert/linear_attention.cc",
    ),
    "LinearAttentionGate": note(
        "Recurrent attention",
        "Qwen recurrent gate computation.",
        "Gate projection precision affects state accumulation and output quality.",
        "onnxruntime/contrib_ops/webgpu/bert/linear_attention_gates.cc",
    ),
    "CausalConvWithState": note(
        "State",
        "Causal convolution with a persistent history buffer.",
        "Convolution state must reset between requests and carry forward between tokens.",
        "onnxruntime/contrib_ops/webgpu/bert/causal_conv_with_state.cc",
    ),
    "GatherBlockQuantized": note(
        "Quantization",
        "Embedding lookup directly from block-quantized weights.",
        "Preserve index types, block layout and tied embedding/head policy.",
        "onnxruntime/contrib_ops/webgpu/quantization/gather_block_quantized.cc",
    ),
    "MRotaryEmbedding": note(
        "Position encoding",
        "Multi-dimensional rotary embeddings.",
        "Qwen position IDs have a distinct three-axis layout; generic RoPE replacement is unsafe.",
        "onnxruntime/contrib_ops/webgpu/bert/mrotary_embedding.cc",
    ),
    "RotaryEmbedding": note(
        "Position encoding",
        "Rotary position encoding.",
        "Rotary dimensions, interleaving and position/cache conventions must match.",
        "onnxruntime/contrib_ops/webgpu/bert/rotary_embedding.cc",
    ),
    "SkipSimplifiedLayerNormalization": note(
        "Normalization",
        "Residual addition fused with RMS normalization.",
        "Preserve epsilon, stash precision and optional residual-sum outputs.",
        "onnxruntime/contrib_ops/webgpu/bert/skip_layer_norm.cc",
        "Mobius SkipNorm / GenAI fused export.",
    ),
    "SkipLayerNormalization": note(
        "Normalization",
        "Residual addition fused with layer normalization.",
        "Normalization axes, scale/bias and optional outputs must remain compatible.",
        "onnxruntime/contrib_ops/webgpu/bert/skip_layer_norm.cc",
        "Mobius SkipLayerNorm fusion.",
    ),
    "GatedRMSNorm": note(
        "Normalization",
        "RMS normalization with a gate.",
        "Gate activation and normalization order are model-specific.",
        "onnxruntime/contrib_ops/webgpu/bert/layer_norm.cc",
    ),
    "QuickGelu": note(
        "Activation",
        "Sigmoid-based activation, including SiLU when alpha is one.",
        "Do not substitute exact GELU without checking the intended activation formula.",
        "onnxruntime/contrib_ops/webgpu/bert/quick_gelu.cc",
    ),
    "BiasGelu": note(
        "Activation",
        "Bias addition followed by GELU.",
        "Preserve approximation and broadcasting semantics.",
        "onnxruntime/contrib_ops/webgpu/bert/bias_gelu.cc",
        "Mobius activation fusion.",
    ),
}


def _schema_dict(schema, source):
    def formal(v):
        return {
            "name": v.name,
            "type": getattr(v, "type_str", getattr(v, "typeStr", "")),
            "option": str(v.option).split(".")[-1],
            "description": v.description,
        }

    return {
        "id": f"{schema.domain or 'ai.onnx'}::{schema.name}",
        "name": schema.name,
        "domain": schema.domain or "ai.onnx",
        "version": schema.since_version,
        "schema_source": source,
        "description": (schema.doc or "")[:12000],
        "inputs": [formal(v) for v in schema.inputs],
        "outputs": [formal(v) for v in schema.outputs],
        "attributes": [
            {
                "name": name,
                "type": str(v.type).split(".")[-1],
                "required": v.required,
                "description": v.description,
            }
            for name, v in sorted(schema.attributes.items())
        ],
        "type_constraints": [
            {"parameter": v.type_param_str, "allowed": list(v.allowed_type_strs)}
            for v in schema.type_constraints
        ],
    }


@lru_cache(maxsize=1)
def catalog():
    import onnx

    entries = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        key = (schema.domain, schema.name)
        if key not in entries or entries[key][0].since_version < schema.since_version:
            entries[key] = (schema, f"ONNX {onnx.__version__}")
    ort_version = None
    try:
        from onnxruntime.capi import _pybind_state

        ort_version = importlib.metadata.version("onnxruntime")
        for schema in _pybind_state.get_all_operator_schema():
            key = (schema.domain, schema.name)
            if key not in entries or schema.domain not in ("", "ai.onnx"):
                if key not in entries or entries[key][0].since_version <= schema.since_version:
                    entries[key] = (schema, f"ORT {ort_version}")
    except ImportError:
        pass
    result = []
    for (_, name), (schema, source) in entries.items():
        item = _schema_dict(schema, source)
        if schema.domain == "com.microsoft" and name in KNOWLEDGE:
            item["guidance"] = KNOWLEDGE[name]
        else:
            item["guidance"] = {
                "family": "Standard" if not schema.domain else "Contrib / custom",
                "webgpu": "Not assessed; a schema does not prove kernel support",
                "summary": "",
                "constraints": "",
                "strategy": "",
            }
        result.append(item)
    return {
        "onnx_version": onnx.__version__,
        "ort_version": ort_version,
        "source_commit": ORT_COMMIT,
        "references": [
            MOBIUS_SOURCE + "src/mobius/_optimizations.py",
            GENAI_SOURCE + "src/python/py/models/builders/base.py",
        ],
        "operators": sorted(result, key=lambda v: v["id"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    data = json.dumps(catalog(), indent=2)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)


if __name__ == "__main__":
    main()
