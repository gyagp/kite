"""Qwen-Image 2.1 graph authoring with reusable prefix K/V and native ORT attention.

The source pipeline owns text/image preprocessing, RoPE metadata and scheduling.
These graphs own every learned denoiser operation. Export is shape-only; weights
are streamed from safetensors so a 7B model never becomes a giant protobuf.
"""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np

from evo.pipeline import hash_file, write_json


def wrappers(model):
    import torch
    import torch.nn.functional as functional

    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            return functional.linear(x, weight)

        @staticmethod
        def symbolic(g, x, weight):
            sizes = x.type().varyingSizes()
            if sizes:
                sizes[-1] = weight.type().sizes()[0]
            return g.op("evo::Linear", x, weight).setType(x.type().with_sizes(sizes))

    def linear(self, x):
        value = Linear.apply(x, self.weight)
        return value if self.bias is None else value + self.bias

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(linear, module)

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, heads, bias=None):
            def split(x):
                return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)

            value = functional.scaled_dot_product_attention(split(q), split(k), split(v), bias)
            return value.transpose(1, 2).flatten(2)

        @staticmethod
        def symbolic(g, q, k, v, heads, bias=None):
            inputs = [q, k, v]
            if bias is not None:
                empty = g.op("prim::Constant").setType(torch._C.OptionalType.ofTensor())
                inputs.extend([empty, empty, bias])
            return g.op("com.microsoft::MultiHeadAttention", *inputs, num_heads_i=heads).setType(
                q.type()
            )

    class Graph(torch.nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.model = model
            self.prefix = prefix

        def forward(self, hidden, timestep, cos, sin, *extra):
            m = self.model
            if self.prefix:
                # Padded text and image positions are assembled by the host.
                image_latents, image_mask, bias = extra
                hidden = torch.where(image_mask, m.img_in(image_latents), m.txt_in(hidden))
            else:
                hidden = m.img_in(hidden)
            temb = m.time_text_embed(timestep, hidden)
            modulation = m.modulation(temb).unsqueeze(1)
            scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
            outputs = []
            for index, block in enumerate(m.transformer_blocks):
                x = block.img_norm1(hidden) * (1 + scale1)
                attn = block.attn

                def rotate(value, heads=attn.heads):
                    value = value.reshape(value.shape[0], value.shape[1], heads, -1)
                    even, odd = value[..., ::2].float(), value[..., 1::2].float()
                    real = even * cos - odd * sin
                    imag = even * sin + odd * cos
                    return torch.stack([real, imag], -1).flatten(-2).to(value.dtype).flatten(2)

                def normalized(value, norm, heads=attn.heads, dtype=hidden.dtype):
                    value = value.reshape(value.shape[0], value.shape[1], heads, -1)
                    return norm(value).to(dtype).flatten(2)

                q = rotate(normalized(attn.to_q(x), attn.norm_q))
                k = rotate(normalized(attn.to_k(x), attn.norm_k))
                v = attn.to_v(x)
                if self.prefix:
                    outputs.extend([k, v])
                    result = Attention.apply(q, k, v, attn.heads, bias)
                else:
                    k = torch.cat([extra[index * 2], k], dim=1)
                    v = torch.cat([extra[index * 2 + 1], v], dim=1)
                    result = Attention.apply(q, k, v, attn.heads)
                hidden = hidden + gate1.tanh() * attn.to_out[0](result)
                x = block.img_norm2(hidden) * (1 + scale2)
                hidden = hidden + gate2.tanh() * block.img_mlp(x)
                if hidden.dtype == torch.float16:
                    hidden = hidden.clip(-65504, 65504)
            if self.prefix:
                return tuple(outputs)
            return m.proj_out(m.norm_out(hidden, temb))

    return Graph(True), Graph(False)


def trace_graphs(config, output):
    import torch
    from diffusers import QwenImage21Transformer2DModel

    with torch.device("meta"):
        model = QwenImage21Transformer2DModel.from_config(config).half().eval()
    # A non-persistent meta buffer would be folded to an unreadable constant.
    # Expose it as a parameter in the shape-only trace and bind its known values.
    time = model.time_text_embed.time_proj
    frequency = torch.empty(128, device="meta", dtype=torch.float32)
    del time.freqs
    time.register_parameter("freqs", torch.nn.Parameter(frequency))
    prefix, denoise = wrappers(model)
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    layers = config["num_layers"]

    def empty(*shape, dtype=torch.float16):
        return torch.empty(shape, device="meta", dtype=dtype)

    cache_names = [f"prefix.{i}.{kind}" for i in range(layers) for kind in ("key", "value")]
    for name, module, inputs, names, outputs in [
        (
            "prefix",
            prefix,
            (
                empty(1, 4, config["context_in_dim"]),
                empty(1),
                empty(1, 4, 1, config["attention_head_dim"] // 2, dtype=torch.float32),
                empty(1, 4, 1, config["attention_head_dim"] // 2, dtype=torch.float32),
                empty(1, 4, config["in_channels"]),
                empty(1, 4, 1, dtype=torch.bool),
                empty(1, 1, 4, 4),
            ),
            ["hidden", "timestep", "cos", "sin", "image_latents", "image_mask", "attention_bias"],
            cache_names,
        ),
        (
            "denoise",
            denoise,
            (
                empty(1, 16, config["in_channels"]),
                empty(1),
                empty(1, 16, 1, config["attention_head_dim"] // 2, dtype=torch.float32),
                empty(1, 16, 1, config["attention_head_dim"] // 2, dtype=torch.float32),
                *(empty(1, 4, hidden) for _ in cache_names),
            ),
            ["hidden", "timestep", "cos", "sin", *cache_names],
            ["noise_pred"],
        ),
    ]:
        axes = {
            n: {
                0: "batch",
                1: "prefix_tokens"
                if n.startswith("prefix.") or name == "prefix"
                else "image_tokens",
            }
            for n in names + outputs
            if n != "timestep" and n != "attention_bias"
        }
        if name == "prefix":
            axes["attention_bias"] = {0: "batch", 2: "prefix_tokens", 3: "prefix_tokens"}
        torch.onnx.export(
            module,
            inputs,
            str(output / f"{name}.template.onnx"),
            dynamo=False,
            export_params=False,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            opset_version=22,
            input_names=names,
            output_names=outputs,
            dynamic_axes=axes,
        )


class WeightStore:
    def __init__(self, source):
        from safetensors import safe_open

        self.handles = [
            safe_open(p, framework="pt", device="cpu") for p in sorted(source.glob("*.safetensors"))
        ]
        self.mapping = {key: h for h in self.handles for key in h.keys()}

    def get(self, name):
        import torch

        name = name.removeprefix("model.")
        if name == "time_text_embed.time_proj.freqs":
            return torch.exp(-np.log(10000) * torch.arange(128, dtype=torch.float32) / 128).numpy()
        return self.mapping[name].get_tensor(name).float().numpy()


def bind_weights(output, store, bits=8, block_size=32):
    import onnx
    from onnx import helper
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        DefaultWeightOnlyQuantizer,
    )

    quantizer = DefaultWeightOnlyQuantizer(
        DefaultWeightOnlyQuantConfig(
            bits=bits if bits else 4, block_size=block_size, is_symmetric=True
        )
    )
    weights = {}
    quant_info = {}
    gguf_mode = hasattr(store, "packed")
    with (output / "weights.data").open("wb") as stream:

        def add(name, array):
            if name in weights:
                return weights[name]
            array = np.ascontiguousarray(array)
            if array.dtype == np.int64 and array.size <= 64:
                tensor = onnx.numpy_helper.from_array(array, name)
                weights[name] = tensor
                return tensor
            # 64-byte alignment is compatible with WebGPU uploads and ORT mapping.
            stream.write(b"\0" * (-stream.tell() % 64))
            offset = stream.tell()
            stream.write(array.tobytes())
            tensor = onnx.TensorProto(
                name=name, data_type=helper.np_dtype_to_tensor_dtype(array.dtype), dims=array.shape
            )
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for key, value in {
                "location": "weights.data",
                "offset": str(offset),
                "length": str(array.nbytes),
            }.items():
                entry = tensor.external_data.add()
                entry.key, entry.value = key, value
            weights[name] = tensor
            return tensor

        for part in ("prefix", "denoise"):
            graph = onnx.load(output / f"{part}.template.onnx")
            nodes, initializers, used = [], [], set()
            sums = {}
            for node in graph.graph.node:
                if (node.domain, node.op_type) == ("evo", "Linear"):
                    name = node.input[1]
                    # Preserve modulation/time/projection precision; quantize bulk block weights.
                    quantized = (
                        store.is_quantized(name)
                        if gguf_mode
                        else bool(bits and ".transformer_blocks." in name)
                    )
                    packed_name = name + (".quant" if quantized else ".T")
                    if packed_name not in weights:
                        if quantized and gguf_mode:
                            packed, scale, correction, qbits, qblock = store.packed(name)
                            add(packed_name, packed)
                            add(name + ".scale", scale)
                            if correction is not None:
                                add(name + ".correction", correction)
                            has_float_zero = name in store.zero_points
                            if has_float_zero:
                                add(name + ".zero_points", store.zero_points.pop(name))
                            quant_info[name] = (
                                qbits,
                                qblock,
                                correction is not None,
                                has_float_zero,
                            )
                        elif quantized:
                            value = store.get(name).T.copy().astype(np.float16)
                            packed, scale, _ = quantizer.qbits_block_quant(value)
                            add(packed_name, packed)
                            add(name + ".scale", scale.reshape(-1))
                            quant_info[name] = (bits, block_size, False, False)
                        else:
                            value = store.get(name).T.copy().astype(np.float16)
                            add(packed_name, value)
                    if quantized:
                        shape = next(
                            v for v in graph.graph.input if v.name == name
                        ).type.tensor_type.shape
                        n, k = [d.dim_value for d in shape.dim]
                        qbits, qblock, has_correction, has_float_zero = quant_info[name]
                        base_output = node.output[0] + ".base" if has_correction else node.output[0]
                        quant_inputs = [node.input[0], packed_name, name + ".scale"]
                        if has_float_zero:
                            quant_inputs.append(name + ".zero_points")
                            used.add(name + ".zero_points")
                        nodes.append(
                            helper.make_node(
                                "MatMulNBits",
                                quant_inputs,
                                [base_output],
                                name=node.name,
                                domain="com.microsoft",
                                K=k,
                                N=n,
                                bits=qbits,
                                block_size=qblock,
                                accuracy_level=4,
                            )
                        )
                        used.add(name + ".scale")
                        if has_correction:
                            key = (node.input[0], k, qblock)
                            if key not in sums:
                                stem = f"evo.block_sum.{len(sums)}"
                                shape_name = f"evo.block_shape.{k}.{qblock}"
                                axis_name = "evo.last_axis"
                                add(shape_name, np.array([-1, k // qblock, qblock], np.int64))
                                add(axis_name, np.array([-1], np.int64))
                                used.update([shape_name, axis_name])
                                nodes.extend(
                                    [
                                        helper.make_node(
                                            "Cast",
                                            [node.input[0]],
                                            [stem + ".fp32"],
                                            to=onnx.TensorProto.FLOAT,
                                        ),
                                        helper.make_node(
                                            "Reshape",
                                            [stem + ".fp32", shape_name],
                                            [stem + ".blocks"],
                                        ),
                                        helper.make_node(
                                            "ReduceSum",
                                            [stem + ".blocks", axis_name],
                                            [stem + ".sum"],
                                            keepdims=0,
                                        ),
                                        helper.make_node(
                                            "Cast",
                                            [stem + ".sum"],
                                            [stem],
                                            to=onnx.TensorProto.FLOAT16,
                                        ),
                                    ]
                                )
                                sums[key] = stem
                            correction_name = name + ".correction"
                            used.add(correction_name)
                            nodes.extend(
                                [
                                    helper.make_node(
                                        "MatMul",
                                        [sums[key], correction_name],
                                        [node.output[0] + ".correction.flat"],
                                    ),
                                    helper.make_node(
                                        "Shape", [base_output], [node.output[0] + ".shape"]
                                    ),
                                    helper.make_node(
                                        "Reshape",
                                        [
                                            node.output[0] + ".correction.flat",
                                            node.output[0] + ".shape",
                                        ],
                                        [node.output[0] + ".correction"],
                                    ),
                                    helper.make_node(
                                        "Add",
                                        [base_output, node.output[0] + ".correction"],
                                        list(node.output),
                                    ),
                                ]
                            )
                    else:
                        nodes.append(
                            helper.make_node(
                                "MatMul",
                                [node.input[0], packed_name],
                                list(node.output),
                                name=node.name,
                            )
                        )
                    used.add(packed_name)
                else:
                    nodes.append(node)
            live = {n for node in nodes for n in node.input}
            inputs = []
            for value in graph.graph.input:
                if value.name.startswith("model."):
                    if value.name in live:
                        dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
                        if value.name not in weights:
                            add(value.name, store.get(value.name).astype(dtype))
                        used.add(value.name)
                else:
                    inputs.append(value)
            initializers.extend(weights[n] for n in sorted(used))
            del graph.graph.node[:]
            graph.graph.node.extend(nodes)
            del graph.graph.input[:]
            graph.graph.input.extend(inputs)
            graph.graph.initializer.extend(initializers)
            opsets = [op for op in graph.opset_import if op.domain != "evo"]
            del graph.opset_import[:]
            graph.opset_import.extend(opsets)
            graph.ir_version = 10
            onnx.save_model(graph, output / f"{part}.onnx")
            print(f"Bound {part}: {len(nodes)} nodes, {len(used)} tensors", flush=True)


def export(source, output, bits=8, gguf=None, gguf_zero_points="correction"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    config = json.loads((source / "transformer/config.json").read_text())
    if config.get("_class_name") != "QwenImage21Transformer2DModel" or not config.get(
        "causal_condition"
    ):
        raise ValueError("Expected a causal QwenImage21 transformer")
    output.mkdir(parents=True)
    trace_graphs(config, output)
    if gguf:
        from evo.image_gguf import GGUFWeightStore

        store = GGUFWeightStore(gguf, floating_zero_points=gguf_zero_points == "float")
    else:
        store = WeightStore(source / "transformer")
    bind_weights(output, store, bits)
    manifest = {
        "task": "image-generation",
        "architecture": "QwenImage21Transformer2DModel",
        "source": json.loads((source / "evo-source.json").read_text()),
        "transformer_config": config,
        "sample_sigmas": json.loads((source / "model_index.json").read_text())["sample_sigmas"],
        "weight_bits": bits or 16,
        "activation_dtype": "float16",
        "block_size": 32,
        "prefix_cache": True,
        "shared_weights": True,
        "components": "ONNX denoiser; original text encoder and VAE are shared components",
        "validation": "pending",
        "artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": hash_file(output / name)}
            for name in ("prefix.onnx", "denoise.onnx", "weights.data")
        },
    }
    if gguf:
        manifest["weight_bits"] = "GGUF Q4_K/Q5_K/Q6_K/Q8_0 preserved integer codes"
        manifest["denoiser_source"] = {
            "format": "gguf",
            "file": Path(gguf).name,
            "sha256": hash_file(gguf),
        }
        manifest["gguf_translation"] = store.validation
        manifest["requires_float_zero_points"] = gguf_zero_points == "float"
    write_json(output / "image_config.json", manifest)
    prepare_capture_graphs(output)
    return json.loads((output / "image_config.json").read_text())


def prepare_capture_graphs(package):
    """Move unlearned Sin/Cos timestep features out of the captured GPU graph."""
    import onnx
    from onnx import TensorProto, helper

    package = Path(package)
    for part in ("prefix", "denoise"):
        model = onnx.load(package / f"{part}.onnx", load_external_data=False)
        first = next(
            n for n in model.graph.node if n.name.endswith("timestep_embedder/linear_1/Linear")
        )
        first.input[0] = "time_projection"
        live = {v.name for v in model.graph.output}
        kept = []
        for node in reversed(model.graph.node):
            if live.intersection(node.output):
                kept.append(node)
                live.update(n for n in node.input if n)
        inputs = [v for v in model.graph.input if v.name in live]
        inputs.append(
            helper.make_tensor_value_info("time_projection", TensorProto.FLOAT16, [1, 256])
        )
        initializers = [v for v in model.graph.initializer if v.name in live]
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept))
        del model.graph.input[:]
        model.graph.input.extend(inputs)
        del model.graph.initializer[:]
        model.graph.initializer.extend(initializers)
        onnx.save_model(model, package / f"{part}-capture.onnx")
    manifest_path = package / "image_config.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["host_time_projection"] = True
    for part in ("prefix", "denoise"):
        name = f"{part}-capture.onnx"
        manifest["artifacts"][name] = {
            "bytes": (package / name).stat().st_size,
            "sha256": hash_file(package / name),
        }
    write_json(manifest_path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", choices=(0, 4, 8), type=int, default=8)
    parser.add_argument(
        "--gguf", type=Path, help="Use GGUF denoiser weights instead of source safetensors"
    )
    parser.add_argument("--gguf-zero-points", choices=("correction", "float"), default="correction")
    parser.add_argument("--dependencies", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    import sys

    for path in args.dependencies:
        sys.path.insert(0, str(path.resolve()))
    export(args.source, args.output, args.bits, args.gguf, args.gguf_zero_points)


if __name__ == "__main__":
    main()
