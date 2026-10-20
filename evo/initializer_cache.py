"""Share read-only file mappings through ORT's external-initializer buffer API."""

import mmap
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class MappedInitializers:
    # ORT borrows these buffers. Every live session must retain this owner, even
    # after the cache is cleared or a newer file replaces a cached mapping.
    mappings: tuple
    metadata: dict


class InitializerFileCache:
    def __init__(self):
        self._lock = threading.RLock()
        self._graphs = {}
        self._files = {}

    def bind(self, model, options):
        if not hasattr(options, "add_external_initializers_from_files_in_memory"):
            return None
        import onnx

        started = time.perf_counter()
        model = Path(model).resolve()
        with self._lock:
            graph_signature = fingerprint(model)
            cached_graph = self._graphs.get(model)
            if not cached_graph or cached_graph[0] != graph_signature:
                graph = onnx.load(model, load_external_data=False)
                names = sorted(
                    {
                        item.value
                        for tensor in graph.graph.initializer
                        for item in tensor.external_data
                        if item.key == "location"
                    }
                )
                self._graphs[model] = graph_signature, names
            else:
                names = cached_graph[1]
            mappings, reused = [], 0
            for name in names:
                path = (model.parent / name).resolve()
                if not path.is_relative_to(model.parent):
                    raise ValueError(
                        "External initializer files must remain in the model directory."
                    )
                signature = fingerprint(path)
                cached = self._files.get(path)
                if cached and cached[0] == signature:
                    mapping = cached[1]
                    reused += 1
                else:
                    with path.open("rb") as stream:
                        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                    self._files[path] = signature, mapping
                mappings.append(mapping)
            if names:
                options.add_session_config_entry(
                    "session.use_external_initializer_file_buffers_directly", "1"
                )
                options.add_external_initializers_from_files_in_memory(
                    names, mappings, [len(mapping) for mapping in mappings]
                )
            return MappedInitializers(
                tuple(mappings),
                {
                    "files": len(mappings),
                    "reused_files": reused,
                    "mapped_bytes": sum(len(mapping) for mapping in mappings),
                    "setup_ms": (time.perf_counter() - started) * 1000,
                    "read_only": True,
                },
            )

    def clear(self):
        # Drop only cache references; active sessions still own their borrowed buffers.
        with self._lock:
            self._files.clear()
            self._graphs.clear()
