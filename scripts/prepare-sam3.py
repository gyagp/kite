"""Install the pinned SAM 3 video graphs and exact learned memory constants.

Developer setup: local/.venv/Scripts/python.exe scripts/prepare-sam3.py
Weights stay under local/. Uses existing Hugging Face credentials when required.
"""

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import get_token, hf_hub_download

GRAPH_REPO = "jax-image-tools/sam3-video-onnx"
GRAPH_REVISION = "e942689a73d942f83c69b5965881cb3a0f6153ba"
WEIGHT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
GRAPHS = ("vision_encoder", "mask_decoder", "memory_encoder", "memory_attention")


def install(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for name in [*(g + ".onnx" for g in GRAPHS), "constants.json"]:
        # hf_hub_download checks existing files against the pinned repository.
        print(f"Downloading {name}…", flush=True)
        hf_hub_download(GRAPH_REPO, name, revision=GRAPH_REVISION, local_dir=folder)
    url = f"https://huggingface.co/facebook/sam3/resolve/{WEIGHT_REVISION}/model.safetensors"
    token = get_token()

    def read_range(start, end):
        headers = {"Range": f"bytes={start}-{end}"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(url + f"?range={start}-{end}", headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("The model server did not honor the tensor range request.")
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("Incomplete memory tensor download.")
            return data

    print("Reading SAM 3 learned temporal memory constants…", flush=True)
    size = struct.unpack("<Q", read_range(0, 7))[0]
    if not 0 < size < 16 * 1024 * 1024:
        raise ValueError("Invalid checkpoint header.")
    header = json.loads(read_range(8, 7 + size))
    tensors = {}
    for name in (
        "memory_temporal_positional_encoding",
        "no_memory_embedding",
        "temporal_positional_encoding_projection_layer.weight",
        "temporal_positional_encoding_projection_layer.bias",
    ):
        spec = header["tracker_model." + name]
        if spec["dtype"] != "F32":
            raise ValueError("Unexpected memory tensor precision.")
        start, end = spec["data_offsets"]
        tensors[name] = np.frombuffer(read_range(8 + size + start, 7 + size + end), "<f4").reshape(
            spec["shape"]
        )
    np.savez(folder / "memory_constants.npz", **tensors)
    hf_hub_download("facebook/sam3", "LICENSE", revision=WEIGHT_REVISION, local_dir=folder)
    artifacts = {}
    for name in [*(g + ".onnx" for g in GRAPHS), "constants.json", "memory_constants.npz"]:
        path = folder / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": digest}
    config = {
        "task": "video-segmentation",
        "architecture": "Sam3TrackerVideo",
        "version": 1,
        "name": "SAM 3 · Video",
        "precision": "FP32",
        "graphs": list(GRAPHS),
        "source": {"repository": GRAPH_REPO, "revision": GRAPH_REVISION},
        "memory_source": {"repository": "facebook/sam3", "revision": WEIGHT_REVISION},
        "artifacts": artifacts,
    }
    (folder / "segmentation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Ready to import: {folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "local/artifacts/sam3-video-onnx",
    )
    install(parser.parse_args().directory)
