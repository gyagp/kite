"""Real GPU/API verification in a disposable workspace; run with Evo's Python."""

import json
import math
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.server import create_server


def main():
    directory = tempfile.TemporaryDirectory(prefix="kite-native-")
    server = create_server(0, directory.name)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    report = {"passed": False, "checks": []}

    def api(path, data=None, key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            base + path,
            data=json.dumps(data, ensure_ascii=False).encode() if data is not None else None,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    def job(data):
        identifier = api("/api/jobs", data)["id"]
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            value = api("/api/jobs/" + identifier)
            if value["status"] != "running":
                # status is assigned just before persistence and the final event.
                with server.service.jobs[identifier].condition:
                    server.service.jobs[identifier].condition.wait_for(
                        lambda: server.service.active is None, timeout=5
                    )
                assert value["status"] == "completed", value
                return value["result"]
            time.sleep(0.2)
        raise TimeoutError(identifier)

    try:
        boot = api("/api/bootstrap")
        embedding = next(
            m for m in boot["models"] if m["kind"] == "embedding" and m["modalities"] == ["text"]
        )
        chat = next(m for m in boot["models"] if m["kind"] == "chat")
        english = "What causes the northern lights?"
        chinese = "什么原因导致北极光？"
        result = job(
            {
                "kind": "embed",
                "model": embedding["id"],
                "texts": [english, chinese],
                "task": "SearchQuery",
                "dimensions": 768,
            }
        )
        assert len(result["vectors"]) == 2
        for vector in result["vectors"]:
            assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
            assert abs(sum(x * x for x in vector) - 1) < 1e-5
        import numpy as np

        fixtures = Path(boot["settings"]["asset_root"]) / "artifacts" / "embeddinggemma-2-fixtures"
        cosines = []
        for vector, fixture in zip(result["vectors"], ["text-0.npz", "text-4.npz"]):
            expected = np.load(fixtures / fixture)["expected"][0]
            cosine = float(
                np.dot(vector, expected) / (np.linalg.norm(vector) * np.linalg.norm(expected))
            )
            assert cosine >= 0.999, cosine
            cosines.append(cosine)
        report["reference_cosines"] = cosines
        report["checks"].append(
            "English/Chinese 768d embeddings finite, unit normalized, cosine >= 0.999 against independent Evo FP32 fixtures"
        )
        report["runtime"] = api("/api/status")["loaded"]["runtime"]
        short = api(
            "/v1/embeddings",
            {"model": embedding["id"], "input": [english] * 5, "dimensions": 128},
            boot["api_key"],
        )
        assert len(short["data"]) == 5
        for item in short["data"]:
            assert (
                len(item["embedding"]) == 128
                and abs(sum(x * x for x in item["embedding"]) - 1) < 1e-5
            )
        report["checks"].append(
            "Authenticated embeddings API, batching across 4-item boundary, 128d renormalization"
        )
        ranked = job(
            {
                "kind": "search",
                "model": embedding["id"],
                "query": english,
                "documents": [
                    "Charged particles from the sun cause the northern lights.",
                    "Mars is known as the Red Planet.",
                ],
            }
        )
        assert ranked["ranking"][0]["index"] == 0
        report["checks"].append("Semantic search ranks the relevant document first")
        response = job(
            {
                "kind": "chat",
                "model": chat["id"],
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France? Answer with just the city name.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            }
        )
        assert "Paris" in response["text"]
        assert 0 < response["prefill_ms"] <= response["ttft_ms"] <= response["total_ms"]
        assert (
            response["decode_tokens"]
            == response["completion_tokens"] - response["first_step_tokens"]
        )
        assert response["decode_tokens"] > 0 and response["decode_ms"] > 0
        assert math.isclose(
            response["prefill_tokens_per_second"],
            response["prompt_tokens"] * 1000 / response["prefill_ms"],
        )
        assert math.isclose(
            response["decode_tokens_per_second"],
            response["decode_tokens"] * 1000 / response["decode_ms"],
        )
        assert api("/api/threads")[0]["messages"][-1]["content"] == response["text"]
        report["chat"] = response
        report["checks"].append(
            "Native Qwen chat response and server-side conversation persistence"
        )
        report["checks"].append(
            "Synchronized prefill, TTFT, decode time and throughput agree with their token counts and timing boundaries"
        )
        one = job(
            {
                "kind": "chat",
                "model": chat["id"],
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 1,
                "temperature": 0,
                "save": False,
            }
        )
        assert one["completion_tokens"] == 1 and one["decode_tokens"] == 0
        assert one["decode_tokens_per_second"] is None and one["decode_ms_per_token"] is None
        assert one["model_load_ms"] == 0
        report["checks"].append(
            "One-token response reports no decode rate; reused model reports zero load time"
        )
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(
                {
                    "model": chat["id"],
                    "messages": [{"role": "user", "content": "Reply with the word hello."}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "stream": True,
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + boot["api_key"],
            },
        )
        with urllib.request.urlopen(request, timeout=60) as stream:
            events = stream.read().decode()
        assert "chat.completion.chunk" in events and "data: [DONE]" in events
        report["checks"].append("OpenAI-compatible streaming chunks and terminal DONE event")
        active = api(
            "/api/jobs",
            {
                "kind": "chat",
                "model": chat["id"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from one to one thousand, one number per line. Continue without stopping.",
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0,
            },
        )["id"]
        time.sleep(0.5)
        api("/api/stop", {})
        deadline = time.monotonic() + 10
        while api("/api/status")["active"] and time.monotonic() < deadline:
            time.sleep(0.1)
        assert api("/api/jobs/" + active)["status"] == "cancelled"
        assert api("/api/status")["loaded"] is None
        assert not api("/api/status")["worker_running"]
        again = job(
            {
                "kind": "embed",
                "model": embedding["id"],
                "texts": ["A fresh start"],
                "dimensions": 256,
            }
        )
        assert len(again["vectors"][0]) == 256
        report["checks"].append(
            "Stop cancels active generation, releases worker, then reloads and embeds successfully"
        )
        report["passed"] = True
    finally:
        server.service.stop()
        server.shutdown()
        thread.join()
        server.service.store.close()
        server.server_close()
        folder = Path(__file__).resolve().parent.parent / "test-results"
        folder.mkdir(exist_ok=True)
        (folder / "native-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        directory.cleanup()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
