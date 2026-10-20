"""Exercise complete image generation on the packaged D3D12 backend."""

import argparse
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("base")
parser.add_argument("--label", required=True)
parser.add_argument("--edit", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--capture", action="store_true")
parser.add_argument("--baseline", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]


def call(path, body=None):
    request = urllib.request.Request(
        args.base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


payload = {
    "kind": "image",
    "prompt": "A tiny cobalt blue glass kite floating above a white marble table, soft morning sunlight, delicate reflections, minimalist product photograph.",
    "width": 512,
    "height": 512,
    "seed": 181104,
    "capture": args.capture,
    "profile": args.profile,
}
if args.edit:
    presets = json.loads((root / "demo/image-studio/presets.json").read_text(encoding="utf-8"))
    preset = next(item for item in presets if item["id"] == "sketch-to-sea")
    reference = root / "demo/image-studio" / preset["reference"]
    item = call(
        "/api/attachments",
        {"name": reference.name, "data": base64.b64encode(reference.read_bytes()).decode()},
    )
    payload.update({key: preset[key] for key in ("prompt", "seed", "width", "height")})
    payload["attachments"] = [item["id"]]
job = call("/api/jobs", payload)
print("Started", args.label, job["id"], flush=True)
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    job = call("/api/jobs/" + job["id"])
    if job["status"] != "running":
        break
    time.sleep(1)
(root / "test-results" / (args.label + ".json")).write_text(json.dumps(job, indent=2))
assert job["status"] == "completed", job
result = job["result"]
for stage in ("prefix", "denoise"):
    assert result["phases"][stage]["runtime"]["provider_options"]["dawnBackendType"] == "D3D12"
assert "D3D12" in result["timing_method"]
image_hash = hashlib.sha256(Path(result["path"]).read_bytes()).hexdigest()
if args.baseline:
    baseline = json.loads(args.baseline.read_text())["result"]
    assert hashlib.sha256(Path(baseline["path"]).read_bytes()).hexdigest() == image_hash
if args.profile:
    assert result["profile"]["gpu_events"] > 0
    assert not result["profile"]["warnings"]
print(
    json.dumps(
        {
            "id": job["id"],
            "backend": "D3D12",
            "request_s": result["request_ms"] / 1000,
            "denoise_s": result["phases"]["denoise"]["inference_ms"] / 1000,
            "image_sha256": image_hash,
            "gpu_events": (result.get("profile") or {}).get("gpu_events"),
        }
    ),
    flush=True,
)
