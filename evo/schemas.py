"""Allow ONNX checking to recognize legacy ORT operators in the ONNX domain."""

import threading
from contextlib import contextmanager

import onnx

_LOCK = threading.RLock()


@contextmanager
def ort_legacy_schemas(operators: set[str], opset: int):
    """Temporarily import actual ORT schemas, never alter an exported graph.

    ORT's SimplifiedLayerNormalization predates standard ONNX RMSNormalization
    and lives in the default domain. The plain ONNX checker cannot recognize it.
    These schemas validate node signatures/attributes, not execution kernels.
    """
    with _LOCK:
        missing = {name for name in operators if not onnx.defs.has(name, opset, "")}
        registered = []
        source = None
        try:
            if missing:
                try:
                    import onnxruntime
                    from onnxruntime.capi import _pybind_state
                except ImportError:
                    yield {"registered": [], "runtime_version": None}
                    return
                schema_type = onnx.defs.OpSchema
                candidates = {}
                for schema in _pybind_state.get_all_operator_schema():
                    if (
                        schema.domain == ""
                        and schema.name in missing
                        and schema.since_version <= opset
                    ):
                        current = candidates.get(schema.name)
                        if current is None or current.since_version < schema.since_version:
                            candidates[schema.name] = schema

                def parameter(value):
                    option = str(value.option).rsplit(".", 1)[-1]
                    return schema_type.FormalParameter(
                        value.name,
                        value.typeStr,
                        param_option=getattr(schema_type.FormalParameterOption, option),
                    )

                for schema in candidates.values():
                    definition = schema_type(
                        schema.name,
                        "",
                        schema.since_version,
                        inputs=[parameter(v) for v in schema.inputs],
                        outputs=[parameter(v) for v in schema.outputs],
                        type_constraints=[
                            (v.type_param_str, v.allowed_type_strs, "")
                            for v in schema.type_constraints
                        ],
                        attributes=[
                            schema_type.Attribute(
                                name,
                                getattr(schema_type.AttrType, str(v.type).rsplit(".", 1)[-1]),
                                required=v.required,
                            )
                            for name, v in schema.attributes.items()
                        ],
                    )
                    onnx.defs.register_schema(definition)
                    registered.append((schema.name, schema.since_version))
                source = onnxruntime.__version__
            yield {"registered": [name for name, _ in registered], "runtime_version": source}
        finally:
            for name, version in registered:
                onnx.defs.deregister_schema(name, version, "")
