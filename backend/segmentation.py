"""Request and package contracts for SAM 3 video tracking."""

import hashlib
import math
from pathlib import Path

GRAPHS = ["vision_encoder", "mask_decoder", "memory_encoder", "memory_attention"]


def inspect_segmentation(root, config):
    if config.get("architecture") != "Sam3TrackerVideo" or config.get("version") != 1:
        raise ValueError("Unsupported SAM 3 video package contract.")
    if config.get("graphs") != GRAPHS:
        raise ValueError(
            "SAM 3 video needs vision, mask, memory encoder, and memory attention graphs."
        )
    required = [*(name + ".onnx" for name in GRAPHS), "constants.json", "memory_constants.npz"]
    for name in required:
        path = (root / name).resolve()
        spec = config.get("artifacts", {}).get(name, {})
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != spec.get("bytes")
        ):
            raise ValueError(f"Missing or incomplete SAM 3 artifact: {name}")
    return {
        "id": hashlib.sha256(str(root).lower().encode()).hexdigest()[:16],
        "name": config.get("name", "SAM 3 · Video"),
        "path": str(root),
        "kind": "segmentation",
        "architecture": "Sam3TrackerVideo",
        "modalities": ["video", "image"],
        "size": sum((root / name).stat().st_size for name in required),
        "precision": "FP32",
        "context": 0,
        "tasks": ["video-tracking", "point-segmentation", "box-segmentation"],
        "graphs": required,
        "source": config.get("source", {}),
        "variant": root.name,
    }


def validate_tracking(service, data):
    attachment = service.store.get("attachments", data.get("attachment", ""))
    if not attachment or attachment["kind"] != "video" or not Path(attachment["path"]).is_file():
        raise ValueError("Upload a video to track.")
    start = data.get("start_time", 0)
    if type(start) not in (int, float) or not math.isfinite(start) or not 0 <= start <= 86400:
        raise ValueError("Choose a valid start time.")
    for name, default, low, high in (("max_frames", 60, 1, 600), ("stride", 1, 1, 30)):
        value = data.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
        data[name] = value
    objects = data.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= 4:
        raise ValueError("Select between one and four objects.")

    def coordinate(v):
        return type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1

    clean = []
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError("Invalid object selection.")
        points, box = obj.get("points", []), obj.get("box")
        if not isinstance(points, list) or len(points) > 32:
            raise ValueError("Use at most 32 points per object.")
        for point in points:
            if (
                not isinstance(point, dict)
                or not all(coordinate(point.get(k)) for k in ("x", "y"))
                or type(point.get("label")) is not int
                or point["label"] not in (0, 1)
            ):
                raise ValueError("Points must be inside the frame and marked include or exclude.")
        if box is not None and (
            not isinstance(box, list)
            or len(box) != 4
            or not all(coordinate(v) for v in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise ValueError("Draw a nonempty box inside the frame.")
        if not box and not any(p["label"] == 1 for p in points):
            raise ValueError(f"Object {i + 1} needs an include point or a box.")
        clean.append(
            {
                "id": i + 1,
                "name": str(obj.get("name", f"Object {i + 1}"))[:60],
                "points": [{k: p[k] for k in ("x", "y", "label")} for p in points],
                "box": box,
            }
        )
    return {
        "path": attachment["path"],
        "start_time": start,
        "max_frames": data["max_frames"],
        "stride": data["stride"],
        "objects": clean,
    }
