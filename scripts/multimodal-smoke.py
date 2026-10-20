"""Verify real Evo file inputs through Kite's HTTP API in disposable state."""

import base64
import json
import math
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.server import create_server


def main():
    import numpy as np

    temporary = tempfile.TemporaryDirectory(prefix="kite-multimodal-")
    server = create_server(0, temporary.name)
    owner = threading.Thread(target=server.serve_forever, daemon=True)
    owner.start()
    base = f"http://127.0.0.1:{server.server_port}"
    report = {"passed": False, "checks": [], "file_cosines": {}}

    def api(path, data=None, headers=None):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def run(data):
        identifier = api("/api/jobs", data)["id"]
        job = server.service.jobs[identifier]
        with job.condition:
            assert job.condition.wait_for(
                lambda: any(e["type"] == "done" for e in job.events), timeout=180
            )
        assert job.status == "completed", job.error
        return job.result

    def upload(path):
        return api(
            "/api/attachments",
            {"name": path.name, "data": base64.b64encode(path.read_bytes()).decode()},
        )

    try:
        boot = api("/api/bootstrap")
        model = next(m for m in boot["models"] if m.get("embedding_task") == "multimodal-embedding")
        text_model = next(
            m for m in boot["models"] if m["kind"] == "embedding" and m["modalities"] == ["text"]
        )
        files = (
            Path(boot["settings"]["asset_root"])
            / "artifacts"
            / "embeddinggemma-2-multimodal-validation"
            / "file-smoke"
        )
        image, audio, video = [
            upload(files / name)
            for name in ("shapes.png", "tone-stereo-44100.wav", "shapes-2s.mp4")
        ]
        samples = [
            {"text": "", "attachments": [image]},
            {"text": "", "attachments": [audio]},
            {"text": "", "attachments": [video]},
            {
                "text": "Picture <|image|> clip <|video|> sound <|audio|>",
                "attachments": [image, video, audio],
            },
        ]
        result = run(
            {
                "kind": "embed",
                "model": model["id"],
                "samples": samples,
                "task": "Document",
                "dimensions": 768,
            }
        )
        for name, vector in zip(("image", "audio", "video", "mixed"), result["vectors"]):
            assert len(vector) == 768 and all(math.isfinite(v) for v in vector)
            assert abs(sum(v * v for v in vector) - 1) < 1e-5
            expected = np.load(files / f"{name}.npy")[0]
            cosine = float(
                np.dot(vector, expected) / (np.linalg.norm(vector) * np.linalg.norm(expected))
            )
            assert cosine > 0.99999, (name, cosine)
            report["file_cosines"][name] = cosine
        report["checks"].append(
            "PNG, stereo 44.1-kHz WAV, MP4, and mixed input match Evo's native CLI outputs at cosine >0.99999"
        )
        reference = np.load(
            Path(boot["settings"]["asset_root"])
            / "artifacts/embeddinggemma-2-multimodal-fixtures/image-280.npz"
        )["expected"][0]
        cosine = float(
            np.dot(result["vectors"][0], reference)
            / (np.linalg.norm(result["vectors"][0]) * np.linalg.norm(reference))
        )
        assert cosine >= 0.999
        report["independent_fp32_image_cosine"] = cosine
        report["checks"].append(
            "Image embedding agrees with independent official FP32 fixture at cosine >=0.999"
        )
        assert result["inputs"][2]["video_frames"] == [2]
        report["runtime"] = api("/api/status")["loaded"]["runtime"]
        assert report["runtime"]["embedding_graphs"] == ["model", "vision", "audio"]
        cropped = api(
            "/v1/embeddings",
            {"model": model["id"], "input": samples[0], "dimensions": 128},
            {"Authorization": "Bearer " + boot["api_key"]},
        )
        assert len(cropped["data"][0]["embedding"]) == 128
        assert abs(sum(v * v for v in cropped["data"][0]["embedding"]) - 1) < 1e-5
        report["checks"].append(
            "Multimodal embeddings API supports structured input and 128d renormalization"
        )
        search = run(
            {
                "kind": "search",
                "model": model["id"],
                "query": "",
                "query_attachments": [image],
                "documents": [
                    "A red rectangle next to a blue circle.",
                    {"text": "", "attachments": [video]},
                ],
            }
        )
        assert len(search["ranking"]) == 2 and all(
            math.isfinite(r["score"]) for r in search["ranking"]
        )
        assert "path" not in json.dumps(search)
        report["checks"].append(
            "Media-only queries search text and media documents without exposing server file paths"
        )
        for identifier in (item["id"] for item in api("/api/examples")):
            example = api("/api/examples/" + identifier, {})
            request = {
                "kind": "search" if example["mode"] == "search" else "embed",
                "model": model["id"],
            }
            if request["kind"] == "search":
                request.update(
                    query=example["query"],
                    query_attachments=example["query_attachments"],
                    documents=example["samples"],
                )
            else:
                request.update(samples=example["samples"], task=example["task"])
            output = run(request)
            report.setdefault("examples", {})[identifier] = {
                "tokens": output["tokens"],
                "dimensions": output["dimensions"],
                "inputs": output.get("inputs"),
                "ranking": [
                    {"index": r["index"], "score": r["score"]} for r in output.get("ranking", [])
                ],
            }
        report["checks"].append(
            "All bundled image/audio/video/mixed and multilingual examples run with actual model outputs"
        )
        for invalid in [
            {"kind": "embed", "model": text_model["id"], "samples": [samples[0]]},
            {
                "kind": "embed",
                "model": model["id"],
                "samples": [{"text": "<|image|> <|image|>", "attachments": [image]}],
            },
            {
                "kind": "embed",
                "model": model["id"],
                "samples": [{"attachments": [{"id": "missing"}]}],
            },
        ]:
            try:
                api("/api/jobs", invalid)
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
            else:
                raise AssertionError("Invalid media request was accepted")
        request = urllib.request.Request(
            base + f"/api/attachments/{video['id']}", headers={"Range": "bytes=10-39"}
        )
        with urllib.request.urlopen(request) as response:
            assert (
                response.status == 206
                and response.read() == (files / "shapes-2s.mp4").read_bytes()[10:40]
            )
        report["checks"].append(
            "Unsupported media and placeholder mismatch are rejected; video byte-range playback works"
        )
        report["passed"] = True
    finally:
        server.service.stop()
        server.service.hub.shutdown()
        server.shutdown()
        owner.join()
        server.server_close()
        server.service.store.close()
        target = Path(__file__).resolve().parent.parent / "test-results/multimodal-report.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.cleanup()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
