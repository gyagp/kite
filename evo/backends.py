"""Adapters use upstream CLIs in a selectable Python environment."""

import json
import sys

from evo.config import Recipe

BACKEND_PACKAGES = {"genai": "onnxruntime-genai", "mobius": "mobius-onnx"}


def command_for(recipe: Recipe, python: str = sys.executable) -> list[str]:
    if recipe.backend == "genai":
        source_flag = "-i" if recipe.source_kind == "local" else "-m"
        command = [
            python,
            "-m",
            "onnxruntime_genai.models.builder",
            source_flag,
            recipe.source,
            "-o",
            str(recipe.output),
            "-p",
            recipe.precision,
            "-e",
            "webgpu",
            "--extra_options",
            f"block_size={recipe.block_size}",
            f"use_webgpu_fp32={str(recipe.activation_dtype == 'fp32').lower()}",
            f"enable_webgpu_graph={str(recipe.graph_capture).lower()}",
            "hf_remote=false",
        ]
        for key, value in recipe.genai_options.items():
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, (dict, list)):
                value = json.dumps(value, separators=(",", ":"))
            command.append(f"{key}={value}")
        for key, value in recipe.genai_config.items():
            command.extend([f"--{key}", json.dumps(value)])
        return command
    if recipe.backend == "mobius":
        return [
            python,
            "-m",
            "evo.mobius_backend",
            json.dumps(recipe.to_dict()),
        ]
    raise ValueError(f"Unknown backend: {recipe.backend}")


def make_plan(recipe: Recipe, python: str = sys.executable) -> dict:
    notes = [
        "Export targets WebGPU; placement, accuracy and performance need runtime validation.",
        "Architecture support depends on the installed exporter version.",
    ]
    if recipe.source_kind == "huggingface":
        notes.append(
            "The Hugging Face source is not revision-pinned. For repeatable builds, use a local "
            "snapshot at an immutable commit and record that commit alongside the recipe."
        )
    if recipe.activation_dtype == "fp16":
        notes.append("FP16 execution requires WebGPU shader-f16 on the target device.")
    if recipe.graph_capture:
        notes.append(
            "GenAI graph-capture export changes the runtime contract. ORT Web graph capture "
            "also requires suitable shapes, GPU-resident buffers and WebGPU kernel coverage."
        )
    return {"recipe": recipe.to_dict(), "argv": command_for(recipe, python), "notes": notes}
