"""Download and verify the small SmolLM2 package from the real public repository."""

import json
import os
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("KITE_TEST_URL", "http://127.0.0.1:4318")


def api(path, data=None):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


catalog = api("/api/hub/catalog")
assert any(m["name"] == "SmolLM2-135M-Instruct" for m in catalog["models"])
packages = api(
    "/api/hub/packages", {"model": "SmolLM2-135M-Instruct", "revision": catalog["revision"]}
)
plan = next(p for p in packages if p["variant"] == "onnx-webgpu")
assert plan["size"] < 150 * 1024 * 1024 and all("gguf" not in f["name"] for f in plan["files"])
directory = Path(__file__).resolve().parent.parent / "test-results/hub-downloads"
job = api(
    "/api/hub/downloads",
    {"path": plan["path"], "revision": plan["revision"], "directory": str(directory)},
)
deadline = time.monotonic() + 480
while time.monotonic() < deadline:
    state = next(d for d in api("/api/hub/downloads") if d["id"] == job["id"])
    if state["status"] not in ("downloading", "verifying"):
        break
    time.sleep(1)
assert state["status"] == "completed", state
model = next(m for m in api("/api/models") if m["id"] == state["model"])
assert model["source"]["revision"] == catalog["revision"]
response = api(
    "/api/jobs",
    {
        "kind": "chat",
        "model": model["id"],
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 24,
        "temperature": 0,
        "save": False,
    },
)
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    generated = api("/api/jobs/" + response["id"])
    if generated["status"] != "running":
        break
    time.sleep(0.25)
assert generated["status"] == "completed", generated
assert generated["result"]["completion_tokens"] > 0
report = {
    "passed": True,
    "repository": catalog["repository"],
    "revision": catalog["revision"],
    "package": plan["path"],
    "downloaded_bytes": plan["size"],
    "files_verified": len(plan["files"]),
    "model_path": model["path"],
    "generation": generated["result"],
    "checks": [
        "Live model search and revision-pinned package discovery",
        "Actual complete ONNX package download, per-file Git/LFS checksums, automatic import",
        "Downloaded model loads and generates through native WebGPU",
    ],
}
target = Path(__file__).resolve().parent.parent / "test-results/hub-report.json"
target.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
