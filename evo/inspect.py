"""Structural inspection without materializing external tensor weights."""

from collections import Counter
from pathlib import Path, PureWindowsPath

import onnx
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto

from evo.schemas import ort_legacy_schemas


def _graphs(graph: GraphProto):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from _graphs(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from _graphs(child)


def _tensors(graph: GraphProto):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                yield attr.t
            elif attr.type == AttributeProto.TENSORS:
                yield from attr.tensors
            elif attr.type == AttributeProto.SPARSE_TENSOR:
                yield attr.sparse_tensor.values
                yield attr.sparse_tensor.indices
            elif attr.type == AttributeProto.SPARSE_TENSORS:
                for sparse in attr.sparse_tensors:
                    yield sparse.values
                    yield sparse.indices


def _model_graphs(model: ModelProto):
    yield from _graphs(model.graph)
    # Local functions may contain constants and nested control-flow graphs too.
    for function in model.functions:
        graph = GraphProto()
        graph.node.extend(function.node)
        yield from _graphs(graph)


def _value_info(value) -> dict:
    if not value.type.HasField("tensor_type"):
        return {"name": value.name, "type": "non-tensor", "shape": None}
    tensor = value.type.tensor_type
    shape = None
    if tensor.HasField("shape"):
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim
        ]
    return {"name": value.name, "type": TensorProto.DataType.Name(tensor.elem_type), "shape": shape}


def inspect_model(path: Path) -> dict:
    path = path.resolve()
    model = onnx.load(str(path), load_external_data=False)
    errors: list[str] = []
    warnings: list[str] = []
    operators: Counter = Counter()
    dtypes: Counter = Counter()
    external: dict[str, dict] = {}
    for graph in _model_graphs(model):
        for node in graph.node:
            operators[f"{node.domain or 'ai.onnx'}::{node.op_type}"] += 1
        for value in (*graph.input, *graph.output, *graph.value_info):
            if value.type.HasField("tensor_type"):
                dtypes[TensorProto.DataType.Name(value.type.tensor_type.elem_type)] += 1
        for tensor in _tensors(graph):
            dtypes[TensorProto.DataType.Name(tensor.data_type)] += 1
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            metadata = {entry.key: entry.value for entry in tensor.external_data}
            location = metadata.get("location", "")
            data_path = (path.parent / location).resolve()
            windows_location = PureWindowsPath(location)
            if (
                not location
                or Path(location).is_absolute()
                or windows_location.drive
                or ".." in windows_location.parts
                or ".." in Path(location).parts
                or not data_path.is_relative_to(path.parent)
            ):
                errors.append(f"External tensor {tensor.name!r} has unsafe location: {location!r}")
                continue
            info = external.setdefault(
                location,
                {
                    "location": location,
                    "exists": data_path.is_file(),
                    "bytes": data_path.stat().st_size if data_path.is_file() else None,
                },
            )
            if not info["exists"]:
                errors.append(f"Missing external data: {location}")
                continue
            try:
                offset = int(metadata.get("offset", "0"))
                length = int(metadata["length"]) if "length" in metadata else None
                if (
                    offset < 0
                    or offset > info["bytes"]
                    or (length is not None and (length < 0 or offset + length > info["bytes"]))
                ):
                    raise ValueError("range exceeds file")
            except ValueError:
                errors.append(f"Invalid external data range for {tensor.name!r} in {location}")
    # The path form avoids ONNX's protobuf size limit and checks external data in situ.
    # Never ask the checker to follow external paths that failed the checks above.
    schema_info = {"registered": [], "runtime_version": None}
    if not errors:
        try:
            standard_ops = {
                name.split("::", 1)[1] for name in operators if name.startswith("ai.onnx::")
            }
            opset = next((op.version for op in model.opset_import if op.domain == ""), 1)
            with ort_legacy_schemas(standard_ops, opset) as schema_info:
                onnx.checker.check_model(str(path))
        except Exception as exc:
            errors.append(f"ONNX checker: {exc}")
    if schema_info["registered"]:
        warnings.append(
            "Used legacy operator schemas from installed ORT: "
            + ", ".join(schema_info["registered"])
            + ". This validates declarations, not kernel availability."
        )
    if dtypes.get("FLOAT16"):
        warnings.append("Contains FP16 tensors: validate shader-f16 support on the target GPU.")
    if dtypes.get("BFLOAT16"):
        warnings.append(
            "Contains BF16 tensors: WebGPU compatibility requires runtime investigation."
        )
    custom = sorted(op for op in operators if not op.startswith("ai.onnx::"))
    if custom:
        warnings.append(
            "Nonstandard operators require matching runtime kernels; ONNX checking does not "
            "validate their WebGPU support."
        )
    io = [_value_info(value) for value in (*model.graph.input, *model.graph.output)]
    if any(v["shape"] is None or any(not isinstance(d, int) for d in v["shape"]) for v in io):
        warnings.append("Dynamic/unknown I/O shapes: graph capture needs a runtime-specific check.")
    if path.stat().st_size >= 2 * 1024**3:
        warnings.append("Large protobuf: externalize weights for browser loading.")
    return {
        "model": str(path),
        "ir_version": model.ir_version,
        "opsets": {op.domain or "ai.onnx": op.version for op in model.opset_import},
        "inputs": [_value_info(value) for value in model.graph.input],
        "outputs": [_value_info(value) for value in model.graph.output],
        "operators": dict(sorted(operators.items())),
        "tensor_type_counts": dict(sorted(dtypes.items())),
        "external_data": list(external.values()),
        "structure": "failed" if errors else "passed",
        "ort_legacy_schemas": schema_info,
        "webgpu": "not_run",
        "accuracy": "not_run",
        "performance": "not_run",
        "errors": sorted(set(errors)),
        "warnings": warnings,
    }
