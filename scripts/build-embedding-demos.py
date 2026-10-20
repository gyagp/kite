"""Create small, original synthetic media for Kite's offline multimodal examples."""

import json
import math
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local" / ".cache" / "embeddinggemma2" / "deps"))
import av  # noqa: E402 -- use Evo's isolated media dependencies
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

folder = ROOT / "demo"
folder.mkdir(exist_ok=True)


def shapes(offset=0):
    image = Image.new("RGB", (320, 224), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 40, 130, 186), radius=7, fill="#ec525c")
    draw.ellipse((168 + offset, 66, 262 + offset, 160), fill="#438bea")
    return image


shapes().save(folder / "shapes.png")
tree = Image.new("RGB", (320, 224), "#eaf6ff")
draw = ImageDraw.Draw(tree)
draw.rectangle((0, 178, 320, 224), fill="#dbecd6")
draw.rectangle((150, 99, 172, 190), fill="#aa7450")
for bounds in [(92, 45, 172, 130), (145, 33, 226, 137), (114, 85, 209, 156)]:
    draw.ellipse(bounds, fill="#53a66b")
tree.save(folder / "tree.png")

rate = 16000
with wave.open(str(folder / "beeps.wav"), "wb") as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(rate)
    output = bytearray()
    for i in range(rate * 2):
        t = i / rate
        beat = t % 0.5
        gain = min(1.0, beat / 0.01, max(0, (0.25 - beat) / 0.01)) if beat < 0.25 else 0
        output += struct.pack("<h", int(7000 * gain * math.sin(2 * math.pi * 660 * t)))
    stream.writeframes(output)

with av.open(str(folder / "moving-circle.mp4"), mode="w") as container:
    stream = container.add_stream("libx264", rate=12)
    stream.width, stream.height, stream.pix_fmt = 320, 224, "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}
    for i in range(36):
        frame = av.VideoFrame.from_ndarray(
            np.asarray(shapes(round(26 * math.sin(i / 35 * 2 * math.pi)))), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

manifest = [
    {
        "id": "image-search",
        "title": "Match words to pictures",
        "description": "Search for the red rectangle and blue circle. Compare a shapes image, a tree image, and a text description in one embedding space.",
        "modalities": ["text", "image"],
        "mode": "search",
        "query": "A red rectangle next to a blue circle on a white background.",
        "samples": [
            {"text": "", "files": ["shapes.png"]},
            {"text": "", "files": ["tree.png"]},
            {"text": "A green tree standing on grass under a blue sky.", "files": []},
        ],
    },
    {
        "id": "audio-search",
        "title": "Give a sound some meaning",
        "description": "Listen to four synthetic beeps, then compare the audio with descriptions. This clip is an electronic tone, not speech.",
        "modalities": ["text", "audio"],
        "mode": "search",
        "query": "",
        "query_files": ["beeps.wav"],
        "samples": [
            {"text": "A sequence of short electronic beeps separated by silence.", "files": []},
            {"text": "A person speaking a sentence.", "files": []},
            {"text": "A dog barking outside.", "files": []},
        ],
    },
    {
        "id": "video-search",
        "title": "Look through a short video",
        "description": "Search a three-second silent animation of a moving blue circle and a red rectangle. The model samples visual frames; video sound is not included.",
        "modalities": ["text", "image", "video"],
        "mode": "search",
        "query": "A blue circle moving beside a red rectangle.",
        "samples": [
            {"text": "", "files": ["moving-circle.mp4"]},
            {"text": "", "files": ["tree.png"]},
            {"text": "A still red rectangle and blue circle.", "files": []},
        ],
    },
    {
        "id": "mixed-input",
        "title": "Bring all four together",
        "description": "Create one vector from text, an image, a video, and audio. Compare it with the individual modalities, or switch to Raw embeddings to inspect the vectors.",
        "modalities": ["text", "image", "audio", "video"],
        "mode": "compare",
        "query": "",
        "task": "SentenceSimilarity",
        "samples": [
            {
                "text": "A red rectangle and blue circle, with a short electronic beep.",
                "files": ["shapes.png", "moving-circle.mp4", "beeps.wav"],
            },
            {"text": "A red rectangle and blue circle.", "files": []},
            {"text": "", "files": ["shapes.png"]},
            {"text": "", "files": ["moving-circle.mp4"]},
            {"text": "", "files": ["beeps.wav"]},
        ],
    },
]
(folder / "examples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Created original PNG, WAV, H.264 MP4, and four example recipes in", folder)
