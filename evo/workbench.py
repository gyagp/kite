"""Local model registry and evidence index used by the UI."""

import hashlib
import json
import math
import statistics
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import onnx

from evo.inspect import _graphs, _value_info, inspect_model
from evo.paths import remap_paths
from evo.pipeline import hash_file


def read_json(path):
    return remap_paths(json.loads(Path(path).read_text(encoding="utf-8")))


def graph_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if (path / "model.onnx").is_file():
            return path / "model.onnx"
        graphs = list(path.rglob("*.onnx"))
        if len(graphs) != 1:
            raise ValueError("Choose a specific ONNX file when the directory has multiple graphs")
        path = graphs[0]
    if not path.is_file() or path.suffix.lower() not in {".onnx", ".gguf"}:
        raise ValueError("Choose an existing .onnx/.gguf file or ONNX package directory")
    return path


def attr_value(attribute):
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:2000]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, onnx.TensorProto):
        return {
            "tensor": value.name,
            "dtype": onnx.TensorProto.DataType.Name(value.data_type),
            "shape": list(value.dims),
        }
    if isinstance(value, onnx.GraphProto):
        return {"graph": value.name, "nodes": len(value.node)}
    if isinstance(value, (list, tuple)):
        return [str(x)[:300] if not isinstance(x, (float, int, str)) else x for x in value[:128]]
    return value if isinstance(value, (str, int, float, bool)) else str(value)[:1000]


class Workbench:
    def __init__(self, root, directory=None):
        self.root = Path(root).resolve()
        self.directory = Path(directory) if directory else self.root / "artifacts" / "workbench"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "models.json"
        self.lock = threading.RLock()
        self.cache = {}
        self.models = read_json(self.state_path) if self.state_path.exists() else []
        legacy_labels = {
            "Evo · INT4": "Qwen3.5 · INT4",
            "Evo · contrib MLP": "Qwen3.5 · contrib MLP",
        }
        for model in self.models:
            if model.get("name") in legacy_labels:
                model["name"] = legacy_labels[model["name"]]
        self.wishlist_path = self.directory / "wishlist.json"
        self.wishlist = read_json(self.wishlist_path) if self.wishlist_path.exists() else []
        self.performance_path = self.directory / "performance.json"
        self.performance = (
            read_json(self.performance_path) if self.performance_path.exists() else []
        )
        if not self.state_path.exists():
            seeds = [
                (
                    self.root.parent / "webgfx-china/ai-models/Qwen3.5-4B/onnx-webgpu",
                    "WebGFX · previous",
                ),
                (self.root / "artifacts/qwen35-4b-optimized", "Qwen3.5 · INT4"),
                (self.root / "artifacts/qwen35-4b-contrib-mlp", "Qwen3.5 · contrib MLP"),
            ]
            for path, name in seeds:
                if path.exists():
                    self.add(path, name)

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.models, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def add(self, path, name=None, origin=None):
        path = graph_path(path)
        identifier = hashlib.sha256(str(path).lower().encode()).hexdigest()[:16]
        with self.lock:
            existing = next((m for m in self.models if m["id"] == identifier), None)
            if existing:
                if name:
                    existing["name"] = str(name)[:120]
                self.cache.pop(identifier, None)
                if origin:
                    existing["origin"] = origin
                self.save()
                return existing
            model = {
                "id": identifier,
                "name": str(name or path.parent.name)[:120],
                "path": str(path),
                "origin": origin or {"operation": "import", "created": time.time()},
            }
            self.models.append(model)
            self.save()
            return model

    def save_wish(self, data):
        with self.lock:
            identifier = data.get("id")
            item = next((w for w in self.wishlist if w["id"] == identifier), None)
            if identifier and item is None:
                raise ValueError("Unknown wishlist item")
            title = str(data.get("title", item["title"] if item else "")).strip()
            if not title:
                raise ValueError("A wishlist title is required")
            status = data.get("status", item["status"] if item else "open")
            priority = data.get("priority", item["priority"] if item else "medium")
            if status not in {"open", "investigating", "done", "archived"} or priority not in {
                "high",
                "medium",
                "low",
            }:
                raise ValueError("Invalid wishlist status or priority")
            if item is None:
                item = {"id": uuid.uuid4().hex[:12], "created": time.time()}
                self.wishlist.append(item)
            item.update(
                title=title[:200],
                description=str(data.get("description", item.get("description", "")))[:8000],
                status=status,
                priority=priority,
                updated=time.time(),
                operator=str(data.get("operator", item.get("operator", "")))[:200],
                model_id=data.get("model_id", item.get("model_id")),
                kind=str(data.get("kind", item.get("kind", "optimization"))),
            )
            temporary = self.wishlist_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.wishlist, indent=2), encoding="utf-8")
            temporary.replace(self.wishlist_path)
            return dict(item)

    def remove(self, identifier):
        with self.lock:
            self.models = [m for m in self.models if m["id"] != identifier]
            self.cache.pop(identifier, None)
            self.save()  # Registry only; never delete model files.

    def record(self, identifier):
        with self.lock:
            record = next((m for m in self.models if m["id"] == identifier), None)
            if record is None:
                raise ValueError("Unknown model")
            return dict(record)

    def details(self, identifier):
        record = self.record(identifier)
        path = Path(record["path"])
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(identifier)
        if cached and cached[0] == stamp:
            return {**cached[1], "name": record["name"]}
        if path.suffix.lower() == ".gguf":
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError("Invalid GGUF file header")
            result = {
                **record,
                "sha256": hash_file(path),
                "bytes": stat.st_size,
                "graph_bytes": stat.st_size,
                "node_count": 0,
                "contrib_count": 0,
                "quantization": {},
                "genai": False,
                "opportunities": [],
                "inspection": {"structure": "GGUF", "external_data": []},
            }
            self.cache[identifier] = (stamp, result)
            return result
        inspection = inspect_model(path)
        model = onnx.load(path, load_external_data=False)
        nodes, initializers, types = [], {}, {}
        for index, graph in enumerate(_graphs(model.graph)):
            for item in (*graph.input, *graph.output, *graph.value_info):
                types[item.name] = _value_info(item)
            for tensor in graph.initializer:
                initializers[tensor.name] = {
                    "name": tensor.name,
                    "shape": list(tensor.dims),
                    "type": onnx.TensorProto.DataType.Name(tensor.data_type),
                }
            for node in graph.node:
                nodes.append(
                    {
                        "index": len(nodes),
                        "name": node.name or f"{node.op_type}_{len(nodes)}",
                        "scope": graph.name or str(index),
                        "op": node.op_type,
                        "domain": node.domain or "ai.onnx",
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attributes": {a.name: attr_value(a) for a in node.attribute},
                    }
                )
        bit_counts = Counter(
            str(n["attributes"].get("bits", 4))
            for n in nodes
            if n["op"] in {"MatMulNBits", "MatMulNBitsMlp", "MatMulNBitsQkv"}
        )
        config = (
            read_json(path.parent / "genai_config.json")
            if (path.parent / "genai_config.json").exists()
            else None
        )
        patterns = []
        try:
            import onnx_ir as ir

            from evo.contrib_rules import mlp_patterns

            patterns = [match[0].name for match in mlp_patterns(ir.load(path))]
        except ImportError:
            pass
        result = {
            **record,
            "sha256": hash_file(path),
            "graph_bytes": stat.st_size,
            "bytes": stat.st_size + sum(v["bytes"] or 0 for v in inspection["external_data"]),
            "node_count": len(nodes),
            "contrib_count": sum(n["domain"] != "ai.onnx" for n in nodes),
            "quantization": dict(bit_counts),
            "inspection": inspection,
            "nodes": nodes,
            "initializers": initializers,
            "values": types,
            "genai": bool(config),
            "genai_config": config,
            "opportunities": [
                {
                    "rule": "packed-mlp-to-contrib",
                    "target": "com.microsoft::MatMulNBitsMlp",
                    "matches": len(patterns),
                    "nodes": patterns,
                    "status": "Available; validate numerical and runtime results",
                }
            ]
            if patterns
            else [],
        }
        self.cache[identifier] = (stamp, result)
        return result

    def summaries(self):
        result = []
        for record in list(self.models):
            try:
                detail = self.details(record["id"])
                result.append(
                    {
                        key: detail[key]
                        for key in (
                            "id",
                            "name",
                            "path",
                            "sha256",
                            "bytes",
                            "node_count",
                            "contrib_count",
                            "quantization",
                            "genai",
                            "opportunities",
                        )
                    }
                    | {
                        "structure": detail["inspection"]["structure"],
                        "origin": record.get("origin"),
                    }
                )
            except Exception as exc:
                result.append({**record, "error": str(exc), "structure": "unavailable"})
        return result

    def fixtures(self):
        return [str(path) for path in sorted((self.root / "benchmarks").glob("*.json"))]

    def save_performance(self, data):
        """Save one observation, preserving distinct prompts and machines."""
        model = self.record(data.get("model_id"))
        prompt = str(data.get("prompt", "")).strip()
        machine = str(data.get("machine", "")).strip()
        if not prompt or not machine:
            raise ValueError("A prompt and machine are required")
        task = data.get("task", "generation")
        if task not in {"generation", "embedding", "image"}:
            raise ValueError("Choose text generation, embedding, or image measurements")
        metrics = {}
        fields = {
            "embedding": ("latency_ms", "embeddings_per_second"),
            "image": ("latency_ms", "images_per_second"),
            "generation": ("ttft_ms", "decode_tps"),
        }[task]
        for key in fields:
            value = data.get(key)
            value = None if value in (None, "") else float(value)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Performance values must be finite and greater than zero")
            metrics[key] = value
        if all(value is None for value in metrics.values()):
            raise ValueError("Enter latency or throughput")
        with self.lock:
            identifier = data.get("id")
            item = next((r for r in self.performance if r["id"] == identifier), None)
            if identifier and item is None:
                raise ValueError("Unknown performance record")
            if item is None:
                item = {"id": uuid.uuid4().hex[:12], "created": time.time()}
                self.performance.append(item)
            item.update(
                model_id=model["id"],
                model_name=model["name"],
                prompt=prompt,
                machine=machine,
                notes=str(data.get("notes", "")),
                source="manual",
                task=task,
                updated=time.time(),
                **metrics,
            )
            temporary = self.performance_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.performance, indent=2), encoding="utf-8")
            temporary.replace(self.performance_path)
            return dict(item)

    def benchmark_paths(self):
        paths = list((self.root / "artifacts/benchmarks").glob("*.json"))
        paths += list((self.root / "artifacts").glob("comparison-*/*.json"))
        paths += list((self.directory / "jobs").glob("*/result.json"))
        return sorted(set(paths))

    def performance_records(self):
        """Flat records for the UI; never average different prompts or machines."""
        with self.lock:
            records = [dict(r) for r in self.performance]
        models = {m["id"]: m for m in self.models}
        for record in records:
            record["model_name"] = models.get(record["model_id"], {}).get(
                "name", record["model_name"]
            )
        fixtures = {}
        for path in (self.root / "benchmarks").glob("*.json"):
            try:
                spec = read_json(path)
                if not isinstance(spec, dict):
                    continue  # Prompt lists are reference inputs, not tokenized benchmark cases.
                fixtures[hash_file(path)] = {c["name"]: c for c in spec.get("cases", [])}
            except (ValueError, KeyError, OSError):
                continue
        by_path = {str(Path(m["path"]).resolve()).lower(): m for m in self.models}
        hashes = {}
        for path in self.benchmark_paths():
            try:
                data = read_json(path)
                if not isinstance(data, dict):
                    continue
                if data.get("measurement_status") in {
                    "contaminated",
                    "rejected",
                    "unverified",
                    "running",
                    "incomplete",
                }:
                    continue
                package = Path(data["package"])
                if not package.is_absolute():
                    package = self.root / package
                model = by_path.get(str((package / "model.onnx").resolve()).lower())
                if model is None:
                    continue
                if model["id"] not in hashes:
                    hashes[model["id"]] = hash_file(Path(model["path"]))
                if data["model_sha256"] != hashes[model["id"]]:
                    continue
                device = (
                    ", ".join(
                        d.get("metadata", {}).get("Description", "Unknown device")
                        for d in data.get("devices", [])
                    )
                    or "Unknown device"
                )
                machine = data.get("machine", {}).get("name")
                machine = f"{machine} · {device}" if machine else f"{device} (host not recorded)"
                cases = fixtures.get(data.get("cases_sha256"), {})
                for i, row in enumerate(data["results"]):
                    samples = row["samples"]
                    if not samples:
                        continue
                    case = cases.get(row.get("name"), {})
                    embedding = data.get("task") == "embedding"
                    if embedding:
                        seconds = statistics.median(s["latency_seconds"] for s in samples)
                        metrics = {
                            "latency_ms": seconds * 1000,
                            "embeddings_per_second": row.get("batch_size", 1) / seconds,
                        }
                        note = (
                            f"Median of {len(samples)} samples · embedding · "
                            f"ORT {data['versions'].get('onnxruntime', '?')}"
                        )
                    else:
                        metrics = {
                            "ttft_ms": 1000 * statistics.median(s["ttft_seconds"] for s in samples),
                            "decode_tps": statistics.median(
                                s["decode_tokens_per_second"] for s in samples
                            ),
                        }
                        note = (
                            f"Median of {len(samples)} samples · "
                            f"{data['new_tokens']} output tokens · "
                            f"ORT {data['versions'].get('onnxruntime', '?')}"
                        )
                    runtime = data.get("runtime_library", {})
                    if runtime.get("sha256"):
                        variant = runtime.get("build", {}).get("variant")
                        runtime_name = variant or data["versions"].get(
                            "onnxruntime-ep-webgpu", "custom"
                        )
                        note += f" · WebGPU {runtime_name} ({runtime['sha256'][:12]})"
                    if data.get("provider_options", {}).get("enableGraphCapture") == "1":
                        capture = data.get("capture_validation", {})
                        note += (
                            " · capture replay verified"
                            if capture.get("replay_verified")
                            else " · capture requested; replay not verified"
                        )
                        if capture.get("quality_equivalent") is False:
                            note += " · capture differs from uncaptured control"
                    if data.get("protocol"):
                        note += " · " + str(data["protocol"])
                    if data.get("measurement_status") == "screen":
                        note += " · exploratory screen"
                    records.append(
                        {
                            "id": hashlib.sha256(f"{path}:{i}".encode()).hexdigest()[:16],
                            "model_id": model["id"],
                            "model_name": model["name"],
                            "prompt": row.get("prompt")
                            or case.get("text")
                            or row.get("name", "Unrecorded prompt"),
                            "prompt_tokens": row["prompt_tokens"],
                            "machine": machine,
                            "task": "embedding" if embedding else "generation",
                            **metrics,
                            "created": data.get("created", path.stat().st_mtime),
                            "source": str(path),
                            "notes": note,
                        }
                    )
            except (ValueError, KeyError, OSError, TypeError):
                continue
        return sorted(records, key=lambda r: -r["created"])

    def benchmarks(self, detail):
        paths = self.benchmark_paths()
        groups = {}
        for path in paths:
            try:
                data = read_json(path)
                if not isinstance(data, dict):
                    continue
                if data.get("measurement_status") in {
                    "contaminated",
                    "rejected",
                    "unverified",
                    "running",
                    "incomplete",
                }:
                    continue
                if data.get("model_sha256") != detail["sha256"] or not data.get("results"):
                    continue
                # A protobuf hash alone does not identify external tensor contents.
                # Do not attach another package's old measurements merely because
                # its graph protobuf is identical.
                if data.get("package") and detail.get("path"):
                    recorded = Path(data["package"])
                    if not recorded.is_absolute():
                        recorded = self.root / recorded
                    if (recorded / "model.onnx").resolve() != Path(detail["path"]).resolve():
                        continue
                if not all("samples" in row for row in data["results"]):
                    continue
                run = (
                    "Paired comparison"
                    if path.parent.name == "comparison-previous-onnx"
                    else "Recorded runs"
                )
                key = json.dumps(
                    [
                        run,
                        data["cases_sha256"],
                        data["new_tokens"],
                        data["max_length"],
                        data["versions"],
                        data["provider_options"],
                        data["search_options"],
                        data["devices"],
                        data.get("reuse_generator", False),
                        data.get("synchronization", "get_next_tokens"),
                        data.get("runtime_library", {}).get("sha256"),
                    ],
                    sort_keys=True,
                )
                group = groups.setdefault(
                    key,
                    {
                        "id": hashlib.sha256(key.encode()).hexdigest()[:12],
                        "protocol_id": hashlib.sha256(
                            json.dumps(json.loads(key)[1:], sort_keys=True).encode()
                        ).hexdigest()[:12],
                        "name": run,
                        "cases_sha256": data["cases_sha256"],
                        "new_tokens": data["new_tokens"],
                        "max_length": data["max_length"],
                        "versions": data["versions"],
                        "devices": data["devices"],
                        "rows": {},
                        "files": [],
                    },
                )
                group["files"].append(str(path))
                for row in data["results"]:
                    group["rows"].setdefault(row["prompt_tokens"], []).extend(row["samples"])
            except (ValueError, KeyError, OSError, TypeError):
                continue
        for group in groups.values():
            group["rows"] = [
                {
                    "tokens": length,
                    "samples": len(samples),
                    "ttft_ms": 1000 * statistics.median(s["ttft_seconds"] for s in samples),
                    "decode_tps": statistics.median(s["decode_tokens_per_second"] for s in samples),
                    "ids": samples[0]["generated_ids"],
                }
                for length, samples in sorted(group["rows"].items())
            ]
        return list(groups.values())

    def quality(self, detail):
        paths = [
            self.root / "artifacts/comparison-previous-onnx/quality.json",
            self.root / "artifacts/quality-prefill.json",
        ]
        for path in paths:
            if path.exists():
                data = read_json(path)
                match = next(
                    (
                        m
                        for m in data.get("packages", [])
                        if m.get("model_sha256") == detail["sha256"]
                        and (
                            Path(m["path"])
                            if Path(m["path"]).is_absolute()
                            else self.root / m["path"]
                        ).resolve()
                        == Path(detail["path"]).parent
                    ),
                    None,
                )
                if match:
                    return {
                        "reference": data["reference"],
                        "scope": data["scope"],
                        "results": match["results"],
                        "file": str(path),
                    }
        return None

    def profiles(self, detail):
        folders = [
            self.root / "artifacts/comparison-previous-onnx/profile-previous",
            self.root / "artifacts/comparison-previous-onnx/profile-evo",
            self.root / "artifacts/profile-prefill",
        ]
        folders += [
            p.parent for p in (self.root / "artifacts/ui-outputs").glob("*/profile-info.json")
        ]
        for folder in folders:
            info = folder / "profile-info.json"
            if info.exists() and read_json(info).get("model_sha256") == detail["sha256"]:
                data = read_json(info)
                events = read_json(data["profile"])
                runs = [e for e in events if e.get("name") == "model_run"]
                start = runs[1]["ts"] if len(runs) > 1 else 0
                counts = Counter(
                    f"{e['args']['provider']}::{e['args']['op_name']}"
                    for e in events
                    if e.get("ts", 0) >= start
                    and e.get("cat") == "Node"
                    and e.get("args", {}).get("provider")
                    and e["args"].get("op_name")
                )
                measured = max(1, len(runs) - 1)
                return {
                    "counts_per_run": {k: v / measured for k, v in counts.items()},
                    "runs": measured,
                    "file": data["profile"],
                    "device": data.get("device"),
                }
        return None

    def compare(self, a, b, group_id=None):
        left, right = self.details(a), self.details(b)
        lg, rg = self.benchmarks(left), self.benchmarks(right)
        common = [g for g in lg if any(r["id"] == g["id"] for r in rg)]
        unpaired = not common
        if unpaired:
            common = [g for g in lg if any(r["protocol_id"] == g["protocol_id"] for r in rg)]
        common.sort(key=lambda g: g["name"] != "Paired comparison")
        chosen = next((g for g in common if g["id"] == group_id), common[0] if common else None)
        group = None
        if chosen:
            other = next(
                g
                for g in rg
                if g["protocol_id" if unpaired else "id"]
                == chosen["protocol_id" if unpaired else "id"]
            )
            group = {
                **chosen,
                "paired_collection": not unpaired and chosen["name"] == "Paired comparison",
                "name": "Compatible records · different sessions" if unpaired else chosen["name"],
                "right_files": other["files"],
                "rows": [
                    {
                        "tokens": row["tokens"],
                        "left": row,
                        "right": next(v for v in other["rows"] if v["tokens"] == row["tokens"]),
                    }
                    for row in chosen["rows"]
                    if any(v["tokens"] == row["tokens"] for v in other["rows"])
                ],
            }

        def side(model):
            return {
                key: model[key]
                for key in ("id", "name", "bytes", "node_count", "sha256", "opportunities")
            }

        return {
            "left": side(left),
            "right": side(right),
            "benchmark": group,
            "groups": [
                {
                    "id": g["id"],
                    "name": "Compatible records · different sessions" if unpaired else g["name"],
                }
                for g in common
            ],
            "operators": [
                {
                    "id": key,
                    "left": left["inspection"]["operators"].get(key, 0),
                    "right": right["inspection"]["operators"].get(key, 0),
                }
                for key in sorted(
                    left["inspection"]["operators"].keys() | right["inspection"]["operators"].keys()
                )
            ],
            "quality": {"left": self.quality(left), "right": self.quality(right)},
            "profiles": {"left": self.profiles(left), "right": self.profiles(right)},
        }
