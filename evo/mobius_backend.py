"""Use Mobius's complete EP-aware public build/save pipeline."""

import inspect
import json
import logging
import sys


def run(recipe):
    import mobius

    logging.basicConfig(level=logging.INFO)
    options = dict(recipe["mobius_options"])
    reserved = {"model_id", "dtype", "execution_provider", "load_weights", "trust_remote_code"}
    if reserved & options.keys():
        raise ValueError("mobius_options cannot override source, dtype, EP or weight loading")
    options.setdefault("trace_optimization", True)
    options.update(
        dtype={"fp16": "f16", "fp32": "f32"}[recipe["precision"]],
        execution_provider="webgpu",
        load_weights=True,
        trust_remote_code=False,
    )
    inspect.signature(mobius.build).bind(recipe["source"], **options)
    package = mobius.build(recipe["source"], **options)
    save = dict(recipe["mobius_save"])
    if any(k.startswith("_") for k in save) or {"directory", "check_weights"} & save.keys():
        raise ValueError("Use public Mobius save options; weight checking cannot be disabled")
    save.setdefault("external_data", "onnx")
    inspect.signature(package.save).bind(recipe["output"], **save)
    package.save(recipe["output"], **save)


if __name__ == "__main__":
    run(json.loads(sys.argv[1]))
