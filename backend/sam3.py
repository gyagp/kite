"""SAM 3 video tracking: four ORT graphs and a bounded temporal memory bank.

The orchestration follows Transformers' Sam3TrackerVideoModel (SAM License).
No PyTorch model or optical-flow tracker is used for inference.
"""

import json
import time
import zipfile
from pathlib import Path

import numpy as np

GRAPH_NAMES = ("vision_encoder", "mask_decoder", "memory_encoder", "memory_attention")
COLORS = [(40, 125, 255), (255, 153, 45), (180, 92, 240), (32, 190, 168)]


def flatten_feature(value):
    return np.ascontiguousarray(value.reshape(1, value.shape[1], -1).transpose(2, 0, 1))


class MemoryBank:
    """One object's conditioning frame, six spatial memories, and recent pointers."""

    def __init__(self, constants, total_frames):
        self.constants = constants
        self.max_pointers = min(total_frames, 16)
        self.frames = {}

    def add(self, index, memory, position, pointer):
        self.frames[index] = (memory, position, pointer.reshape(1, 256))
        for old in list(self.frames):
            if old != 0 and old < index - 15:
                del self.frames[old]

    def inputs(self, index):
        memories, positions = [], []
        selected = [(0, 0), *((index - age, age) for age in range(6, 0, -1) if index - age > 0)]
        for frame, age in selected:
            if frame not in self.frames:
                continue
            memory, position, _ = self.frames[frame]
            memories.append(memory)
            positions.append(
                position + self.constants["memory_temporal_positional_encoding"][age - 1]
            )
        pointer_frames = [
            0,
            *(index - age for age in range(1, self.max_pointers) if index - age > 0),
        ]
        pointers = np.stack([self.frames[f][2] for f in pointer_frames])
        offsets = np.asarray([index - f for f in pointer_frames], np.float32) / max(
            self.max_pointers - 1, 1
        )
        dim = np.arange(128, dtype=np.float32)
        phase = offsets[:, None] / (10000 ** (2 * np.floor(dim / 2) / 128))
        pe = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
        pe = pe @ self.constants["temporal_positional_encoding_projection_layer.weight"].T
        pe += self.constants["temporal_positional_encoding_projection_layer.bias"]
        return {
            "spatial_memory": np.concatenate(memories),
            "spatial_memory_position_embeddings": np.concatenate(positions),
            "pointer_memory": pointers.reshape(-1, 1, 64),
            "pointer_memory_position_embeddings": np.repeat(pe[:, None, :], 4, axis=0).astype(
                np.float32
            ),
        }


class Sam3Engine:
    def __init__(self, request, *, cpu_reference=False, emit=None):
        import onnxruntime as ort

        from .native import emit as native_emit
        from .native import register_runtime

        self.emit = emit or native_emit
        self.package = Path(request["model"]["path"])
        self.spec = request["model"]
        self.sessions = {}
        self.profile_directory = request.get("profile_directory")
        self.options = {
            "dawnBackendType": "D3D12",
            "powerPreference": "high-performance",
            "validationMode": "basic",
            "enableGraphCapture": "0",
        }
        started = time.perf_counter()
        self.runtime = {"backend": "CPU reference"}
        devices = []
        if not cpu_reference:
            devices, self.runtime = register_runtime(request["library"])
            self.runtime["provider_options"] = self.options
        with np.load(self.package / "memory_constants.npz", allow_pickle=False) as data:
            self.constants = {k: data[k] for k in data.files}
        for name in GRAPH_NAMES:
            self.emit("status", message=f"Loading SAM 3 {name.replace('_', ' ')}…")
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.intra_op_num_threads = 8
            if self.profile_directory:
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(self.profile_directory) / name)
            if not cpu_reference:
                options.add_provider_for_devices(devices[:1], self.options)
            session = ort.InferenceSession(
                str(self.package / (name + ".onnx")),
                options,
                **({"providers": ["CPUExecutionProvider"]} if cpu_reference else {}),
            )
            if not cpu_reference and not any(
                "webgpu" in p.lower() for p in session.get_providers()
            ):
                raise RuntimeError(f"SAM 3 {name} did not activate WebGPU.")
            self.sessions[name] = session
        self.load_ms = (time.perf_counter() - started) * 1000
        self.emit(
            "loaded",
            model=self.spec["id"],
            kind="segmentation",
            runtime=self.runtime,
            load_ms=self.load_ms,
            provider_options=self.options,
        )

    def run_graph(self, name, feeds, timings, outputs=None):
        before = time.perf_counter()
        session = self.sessions[name]
        keys = outputs or [o.name for o in session.get_outputs()]
        result = session.run(keys, {k: np.ascontiguousarray(v) for k, v in feeds.items()})
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - before) * 1000
        return dict(zip(keys, result))

    def frame(self, rgb, objects, banks, index, timings):
        import cv2
        from PIL import Image

        # SAM 3 resizes directly to a square, bilinear, normalized to [-1, 1].
        resized = np.asarray(
            Image.fromarray(rgb).resize((1008, 1008), Image.Resampling.BILINEAR), dtype=np.float32
        )
        pixels = (resized / 127.5 - 1.0).transpose(2, 0, 1)[None]
        features = self.run_graph(
            "vision_encoder",
            {"pixel_values": pixels},
            timings,
            ["feats0", "feats1", "feats2", "pos2"],
        )
        raw = features["feats2"]
        masks, scores = [], []
        for obj, bank in zip(objects, banks):
            if index == 0:
                conditioned = raw + self.constants["no_memory_embedding"].reshape(1, 256, 1, 1)
                points = [[p["x"] * 1008, p["y"] * 1008] for p in obj["points"]]
                labels = [p["label"] for p in obj["points"]]
                if obj.get("box"):
                    x0, y0, x1, y1 = obj["box"]
                    points += [[x0 * 1008, y0 * 1008], [x1 * 1008, y1 * 1008]]
                    labels += [2, 3]
                if not obj.get("box"):
                    points.append([0, 0])
                    labels.append(-1)
            else:
                memory = self.run_graph(
                    "memory_attention",
                    {
                        "current_vision_features": flatten_feature(raw),
                        "current_vision_position_embeddings": flatten_feature(features["pos2"]),
                        **bank.inputs(index),
                    },
                    timings,
                )["conditioned_features"]
                conditioned = (
                    memory.reshape(1, 5184, 256).transpose(0, 2, 1).reshape(1, 256, 72, 72)
                )
                points, labels = [[0, 0]], [-1]
            decoded = self.run_graph(
                "mask_decoder",
                {
                    "feats0": features["feats0"],
                    "feats1": features["feats1"],
                    "feats2": conditioned,
                    "input_points": np.asarray(points, np.float32)[None, None],
                    "input_labels": np.asarray(labels, np.int32)[None, None],
                },
                timings,
            )
            high = decoded["high_res_masks"].reshape(1, 1, 1008, 1008)
            memory = self.run_graph(
                "memory_encoder",
                {
                    "vision_features": raw,
                    "pred_masks_high_res": high,
                    "object_score_logits": decoded["object_score_logits"].reshape(1, 1),
                    "is_mask_from_pts": np.asarray([index == 0], dtype=np.bool_),
                },
                timings,
            )
            bank.add(
                index, memory["memory_tokens"], memory["memory_pos_enc"], decoded["object_pointer"]
            )
            masks.append(
                cv2.resize(high[0, 0], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                > 0
            )
            scores.append(
                {
                    "id": obj["id"],
                    "iou": float(decoded["iou_scores"].max()),
                    "present": bool(decoded["object_score_logits"].item() > 0),
                }
            )
        return masks, scores

    def track(self, request):
        import cv2
        import imageio_ffmpeg
        from PIL import Image

        started = time.perf_counter()
        folder = Path(request["directory"])
        folder.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(request["path"])
        if not cap.isOpened():
            raise ValueError("Could not decode this video. Try an MP4 with H.264 video.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError("The clip has no valid frame rate.")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = int(request["start_time"] * fps + 1e-6)
        if start >= count:
            cap.release()
            raise ValueError("The selected start frame is outside the clip.")
        stride = request["stride"]
        total = min(request["max_frames"], (count - start + stride - 1) // stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        banks = [MemoryBank(self.constants, total) for _ in request["objects"]]
        timings, frames, writers = {}, [], []
        try:
            for index in range(total):
                before = time.perf_counter()
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Video decoding failed at frame {start + index * stride}.")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                if index == 0:
                    if width * height > 3840 * 2160:
                        raise ValueError("Use a video at 4K resolution or below.")
                    for name in ("overlay.mp4", "masks.mp4"):
                        writer = imageio_ffmpeg.write_frames(
                            str(folder / name),
                            (width, height),
                            fps=fps / stride,
                            codec="libx264",
                            pix_fmt_out="yuv420p",
                            macro_block_size=2,
                            output_params=["-movflags", "+faststart", "-crf", "18"],
                        )
                        writer.send(None)
                        writers.append(writer)
                    Image.fromarray(rgb).save(folder / "source.png")
                timings["video_decode"] = (
                    timings.get("video_decode", 0) + (time.perf_counter() - before) * 1000
                )
                self.emit("status", message=f"Tracking frame {index + 1} of {total}…")
                masks, scores = self.frame(rgb, request["objects"], banks, index, timings)
                before = time.perf_counter()
                overlay, labels, color_masks = (
                    rgb.copy(),
                    np.zeros((height, width), np.uint8),
                    np.zeros_like(rgb),
                )
                for obj_index, mask in enumerate(masks):
                    color = np.asarray(COLORS[obj_index], np.float32)
                    overlay[mask] = (0.52 * rgb[mask] + 0.48 * color).astype(np.uint8)
                    color_masks[mask] = color.astype(np.uint8)
                    labels[mask] = obj_index + 1
                    scores[obj_index]["pixels"] = int(mask.sum())
                Image.fromarray(labels).save(folder / f"mask-{index:05d}.png")
                # Lossless object IDs live in PNGs. MP4 is only a visual preview.
                Image.fromarray(overlay).save(folder / f"frame-{index:05d}.jpg", quality=85)
                writers[0].send(np.ascontiguousarray(overlay))
                writers[1].send(np.ascontiguousarray(color_masks))
                frames.append(
                    {
                        "index": index,
                        "source_frame": start + index * stride,
                        "time": (start + index * stride) / fps,
                        "objects": scores,
                    }
                )
                timings["output"] = timings.get("output", 0) + (time.perf_counter() - before) * 1000
                self.emit(
                    "tracking_progress",
                    completed=index + 1,
                    total=total,
                    data={"preview_url": f"/api/segments/{request['id']}/frame-{index:05d}.jpg"},
                )
                for _ in range(stride - 1):
                    cap.grab()
        finally:
            cap.release()
            for writer in writers:
                writer.close()
        elapsed = (time.perf_counter() - started) * 1000
        result = {
            "id": request["id"],
            "frame_count": len(frames),
            "source_fps": fps,
            "output_fps": fps / stride,
            "start_time": start / fps,
            "stride": stride,
            "objects": request["objects"],
            "frames": frames,
            "phases": timings,
            "total_ms": elapsed,
            "inference_ms": sum(timings.get(n, 0) for n in GRAPH_NAMES),
            "processing_fps": len(frames) * 1000 / elapsed,
            "runtime": self.runtime,
            "width": width,
            "height": height,
            "video_url": f"/api/segments/{request['id']}/overlay.mp4",
            "masks_url": f"/api/segments/{request['id']}/masks.mp4",
            "metadata_url": f"/api/segments/{request['id']}/result.json",
            "timing_method": "ORT WebGPU D3D12; synchronous graph execution includes transfers. Setup reported separately.",
        }
        (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with zipfile.ZipFile(folder / "masks.zip", "w", zipfile.ZIP_STORED) as archive:
            for path in sorted(folder.glob("mask-*.png")):
                archive.write(path, path.name)
        self.emit("result", data=result)

    def finish_profile(self):
        for session in self.sessions.values():
            session.end_profiling()
        self.emit("result", data={"saved": True})
