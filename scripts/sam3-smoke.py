"""Numerical D3D12/CPU comparison of real multi-frame SAM 3 tracking."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.sam3 import MemoryBank, Sam3Engine

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Explicit correctness reference only")
parser.add_argument("--frames", type=int, default=4)
parser.add_argument(
    "--library",
    default="local/artifacts/runtimes/ort-webgpu-main-86d60730c1/onnxruntime_providers_webgpu.dll",
)
parser.add_argument("--label", default="sam3-gpu")
args = parser.parse_args()
folder = Path("test-results") / args.label
folder.mkdir(parents=True, exist_ok=True)
objects = [{"id": 1, "name": "Jumper", "points": [{"x": 0.405, "y": 0.38, "label": 1}]}]
engine = Sam3Engine(
    {
        "model": {"id": "sam3", "path": "local/artifacts/sam3-video-onnx"},
        "library": str(Path(args.library).resolve()),
        "profile_directory": str(folder),
    },
    cpu_reference=args.cpu,
)
banks = [MemoryBank(engine.constants, args.frames)]
timings = {}
for index, path in enumerate(
    sorted(Path("local/artifacts/sam3-research/frames").glob("*.jpg"))[: args.frames]
):
    started = time.perf_counter()
    rgb = np.asarray(Image.open(path).convert("RGB"))
    masks, scores = engine.frame(rgb, objects, banks, index, timings)
    np.save(folder / f"mask-{index}.npy", masks[0])
    overlay = rgb.copy()
    overlay[masks[0]] = overlay[masks[0]] * 0.5 + np.asarray([40, 125, 255]) * 0.5
    Image.fromarray(overlay).save(folder / f"frame-{index}.png")
    print(
        "FRAME",
        index,
        "seconds",
        time.perf_counter() - started,
        "pixels",
        int(masks[0].sum()),
        scores,
        flush=True,
    )
engine.finish_profile()
(folder / "metrics.json").write_text(
    json.dumps({"phases": timings, "load_ms": engine.load_ms, "runtime": engine.runtime}, indent=2)
)
print(timings, flush=True)
