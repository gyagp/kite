import base64
import json
import shutil
import time
import urllib.request
from pathlib import Path

base = "http://127.0.0.1:4325"


def call(path, data=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


presets = json.loads(Path("demo/image-studio/presets.json").read_text(encoding="utf-8"))
reports = []
for preset in presets:
    payload = {k: preset[k] for k in ["prompt", "seed", "width", "height"]}
    payload["kind"] = "image"
    if preset.get("reference"):
        p = Path("demo/image-studio") / preset["reference"]
        a = call(
            "/api/attachments", {"name": p.name, "data": base64.b64encode(p.read_bytes()).decode()}
        )
        payload["attachments"] = [a["id"]]
    job = call("/api/jobs", payload)
    print("Started", preset["id"], job["id"], flush=True)
    while True:
        snapshot = call("/api/jobs/" + job["id"])
        if snapshot["status"] != "running":
            break
        time.sleep(3)
    if snapshot["status"] != "completed":
        raise RuntimeError(snapshot)
    result = snapshot["result"]
    shutil.copy2(result["path"], Path("demo/image-studio") / (preset["id"] + ".png"))
    preset["preview"] = preset["id"] + ".png"
    preset["provenance"] = preset["id"] + ".json"
    (Path("demo/image-studio") / preset["provenance"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reports.append(result)
    Path("demo/image-studio/presets.json").write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Completed", preset["id"], round(result["request_ms"] / 1000, 2), "seconds", flush=True)
Path("test-results/image-generation-report.json").write_text(json.dumps(reports, indent=2))
