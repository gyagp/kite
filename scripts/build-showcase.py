"""Build the Orbit collection from credited NASA images and Azure Speech narration.

Use --fetch once to download the listed NASA image assets. Subsequent builds
reuse the local photographs. No image-generation service or private media is used.
"""

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local/.cache/embeddinggemma2/deps"))
import av  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FOLDER = ROOT / "demo/showcase"
ASSETS = {
    "earth": "PIA18033",
    "spacewalk": "S84-27031",
    "launch": "NHQ202211160030",
    "saturn": "PIA14922",
    "pluto": "PIA19952",
    "horizon": "iss040e080833",
}


def fetch():
    FOLDER.mkdir(parents=True, exist_ok=True)
    credits = []
    for name, identifier in ASSETS.items():
        endpoint = (
            "https://images-api.nasa.gov/search?media_type=image&nasa_id="
            + urllib.parse.quote(identifier)
        )
        with urllib.request.urlopen(endpoint, timeout=30) as response:
            result = json.load(response)
        item = next(
            i for i in result["collection"]["items"] if i["data"][0]["nasa_id"] == identifier
        )
        data = item["data"][0]
        links = [
            v for v in item["links"] if v.get("render") == "image" and v["href"].endswith(".jpg")
        ]
        source = next((v["href"] for v in links if "~medium.jpg" in v["href"]), links[0]["href"])
        with urllib.request.urlopen(source, timeout=30) as response:
            raw = response.read()
        target = FOLDER / (name + ".jpg")
        target.write_bytes(raw)
        with Image.open(target) as image:
            image = image.convert("RGB")
            image.thumbnail((1440, 1440))
            image.save(target, quality=91, optimize=True)
        description = html.unescape(re.sub("<[^>]+>", "", data.get("description", "")))
        credits.append(
            {
                "file": name + ".jpg",
                "nasa_id": identifier,
                "title": data["title"],
                "credit": data.get("photographer")
                or data.get("secondary_creator")
                or "NASA (see source description for full credit)",
                "description": description,
                "source_url": "https://images.nasa.gov/details/" + identifier,
                "download_url": source,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        print("Downloaded", name, identifier, flush=True)
    (FOLDER / "credits.json").write_text(
        json.dumps(
            {
                "images": credits,
                "usage_guidelines": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
                "note": "NASA imagery is credited to its original creators. NASA does not endorse Kite. The film is an authored photo montage; narration is generated with Azure AI Speech.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def cover(image, width, height, zoom=1, x=0.5, y=0.5):
    scale = max(width / image.width, height / image.height) * zoom
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    left = round((resized.width - width) * x)
    top = round((resized.height - height) * y)
    return resized.crop((left, top, left + width, top + height))


def film():
    photos = {name: Image.open(FOLDER / (name + ".jpg")).convert("RGB") for name in ASSETS}
    width, height, fps = 960, 540, 24
    font = ImageFont.truetype("C:/Windows/Fonts/segoeuil.ttf", 46)
    label = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 14)
    scenes = [
        ("launch", "01 / DEPARTURE", "Leave the familiar."),
        ("earth", "02 / PERSPECTIVE", "See the bigger picture."),
        ("spacewalk", "03 / DISCOVERY", "Reach beyond words."),
    ]
    # Design a real motion sequence from photographs, explicitly credited as a montage.
    with av.open(str(FOLDER / "orbital-film.mp4"), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "22", "preset": "fast"}
        for frame_index in range(12 * fps):
            t = frame_index / fps
            index = min(2, int(t / 4))
            progress = (t - index * 4) / 4
            ease = progress * progress * (3 - 2 * progress)
            name, kicker, title = scenes[index]
            image = cover(
                photos[name], width, height, zoom=1.02 + 0.10 * ease, x=0.46 + 0.08 * ease, y=0.46
            )
            if progress < 0.15 and index > 0:
                prior = cover(
                    photos[scenes[index - 1][0]], width, height, zoom=1.12, x=0.54, y=0.46
                )
                image = Image.blend(prior, image, progress / 0.15)
            layer = Image.new("RGBA", (width, height))
            draw = ImageDraw.Draw(layer)
            for y in range(290, height):
                alpha = round(195 * ((y - 290) / (height - 290)) ** 1.2)
                draw.line((0, y, width, y), fill=(8, 19, 39, alpha))
            draw.rounded_rectangle((34, 30, 181, 60), radius=15, fill=(230, 242, 255, 225))
            draw.text((48, 36), "KITE / ORBIT", font=label, fill=(24, 66, 119, 255))
            draw.text((38, 378), kicker, font=label, fill=(166, 205, 255, 255))
            draw.text((36, 405), title, font=font, fill=(255, 255, 255, 255))
            draw.text(
                (38, 490),
                "NASA IMAGERY  /  AN AUTHORED PHOTO FILM",
                font=label,
                fill=(210, 224, 242, 255),
            )
            draw.line((width - 193, 499, width - 38, 499), fill=(255, 255, 255, 75), width=2)
            draw.line(
                (width - 193, 499, width - 193 + round(155 * t / 12), 499),
                fill=(159, 204, 255, 255),
                width=3,
            )
            image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
            if frame_index in (0, 4 * fps, 8 * fps):
                image.save(FOLDER / f"film-scene-{index + 1}.jpg", quality=91)
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for name, image in photos.items():
        cover(image, 720, 440).save(FOLDER / (name + "-cover.jpg"), quality=88, optimize=True)
    print("Rendered 12-second H.264 photo film and preview assets", flush=True)


def audio_metadata():
    # Measure the actual generated WAV; the preview must follow the current voice.
    import soundfile as sf

    audio, rate = sf.read(FOLDER / "mission-brief.wav")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    bins = np.array_split(np.abs(audio), 72)
    peaks = np.array([np.max(b) if len(b) else 0 for b in bins])
    (FOLDER / "audio.json").write_text(
        json.dumps(
            {
                "duration": len(audio) / rate,
                "peaks": (peaks / max(float(peaks.max()), 1e-9)).round(3).tolist(),
            }
        )
    )
    print("Updated narration waveform and duration", flush=True)


def manifest():
    audio = json.loads((FOLDER / "audio.json").read_text())
    speech = json.loads((FOLDER / "speech.json").read_text(encoding="utf-8"))
    credits = json.loads((FOLDER / "credits.json").read_text(encoding="utf-8"))
    credits["audio"] = speech
    credits["note"] = (
        "NASA imagery is credited to its original creators. NASA does not endorse Kite. The film is an authored photo montage; narration is generated with Azure AI Speech. Playback and embedding inference run locally."
    )
    (FOLDER / "credits.json").write_text(
        json.dumps(credits, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def sample(text="", *files, label=None):
        return {"text": text, "files": ["showcase/" + f for f in files], "label": label or text}

    images = [
        sample("", "earth.jpg", label="The blue marble"),
        sample("", "saturn.jpg", label="Saturn & Titan"),
        sample("", "pluto.jpg", label="Pluto in color"),
        sample("", "spacewalk.jpg", label="An untethered spacewalk"),
    ]
    transcript = speech["transcript"]
    shared = {
        "collection": "The Orbit Collection",
        "credit": "NASA imagery · Azure AI Speech narration",
        "audio_credit": f"{speech['provider']} · {speech['voice']}",
        "duration": "12-second film",
        "featured": False,
    }
    presets = [
        {
            **shared,
            "id": "image-search",
            "title": "A planet without a keyword",
            "eyebrow": "01 / VISUAL DISCOVERY",
            "description": "Find Earth among distant worlds using a description, not a filename.",
            "story": "Four real NASA images. One natural-language query. Discover how a visual idea finds its place in an embedding space.",
            "cover": "showcase/earth-cover.jpg",
            "accent": "blue",
            "modalities": ["text", "image"],
            "mode": "search",
            "query": "The blue planet Earth seen from space, with oceans and swirling white clouds.",
            "suggestions": [
                "A pale planet with rings and a moon in front of it.",
                "An astronaut in a white spacesuit floating above Earth.",
            ],
            "samples": images,
        },
        {
            **shared,
            "id": "audio-search",
            "title": "Hear it. Find it.",
            "eyebrow": "02 / AUDIO → IMAGE",
            "description": "A spoken mission briefing becomes a visual search. Press play, then let the model listen.",
            "story": "There is no text query here. The model embeds the actual spoken audio and compares it with NASA photographs.",
            "cover": "showcase/spacewalk-cover.jpg",
            "accent": "violet",
            "preview_audio": "showcase/mission-brief.wav",
            "transcript": transcript,
            "waveform": audio["peaks"],
            "audio_duration": audio["duration"],
            "modalities": ["audio", "image", "text"],
            "mode": "search",
            "query": "",
            "query_files": ["showcase/mission-brief.wav"],
            "samples": [
                images[3],
                images[0],
                sample("", "launch.jpg", label="Artemis I liftoff"),
                images[1],
            ],
        },
        {
            **shared,
            "id": "video-search",
            "title": "Twelve seconds to orbit",
            "eyebrow": "03 / MOVING PICTURES",
            "description": "Launch. Earth. Spacewalk. Search a cinematic three-scene photo film by its story.",
            "story": "A locally authored film adds slow camera moves, transitions, and chapter titles to NASA photographs. The embedding model samples its visual frames; this is not original motion footage.",
            "cover": "showcase/film-scene-1.jpg",
            "preview_video": "showcase/orbital-film.mp4",
            "accent": "amber",
            "modalities": ["text", "video", "image"],
            "mode": "search",
            "query": "A journey showing a rocket launch, the blue Earth from space, and an astronaut on a spacewalk.",
            "suggestions": [
                "A rocket lifting off at night with a bright flame.",
                "A blue planet covered with white clouds.",
            ],
            "samples": [
                sample("", "orbital-film.mp4", label="Departure → perspective → discovery"),
                images[1],
                images[2],
                sample("", "launch.jpg", label="One moment: liftoff"),
            ],
        },
        {
            **shared,
            "id": "mixed-input",
            "title": "One mission. Every dimension.",
            "eyebrow": "04 / THE FULL EXPERIENCE",
            "featured": True,
            "description": "Text, a photograph, a spoken briefing, and a film—woven into one memory.",
            "story": "Compare a combined mission memory with each of its ingredients. Each row is a real embedding, and every square in the matrix is a computed cosine similarity.",
            "cover": "showcase/horizon-cover.jpg",
            "preview_video": "showcase/orbital-film.mp4",
            "preview_audio": "showcase/mission-brief.wav",
            "accent": "blue",
            "modalities": ["text", "image", "audio", "video"],
            "mode": "compare",
            "query": "",
            "task": "SentenceSimilarity",
            "samples": [
                sample(
                    "A space mission: launching from Earth, orbiting the blue planet, and an astronaut exploring above the clouds.",
                    "spacewalk.jpg",
                    "orbital-film.mp4",
                    "mission-brief.wav",
                    label="The complete mission",
                ),
                sample(
                    "An astronaut floats in a white spacesuit above the blue Earth.",
                    label="The written idea",
                ),
                sample("", "spacewalk.jpg", label="The photograph"),
                sample("", "mission-brief.wav", label="The spoken briefing"),
                sample("", "orbital-film.mp4", label="The visual journey"),
            ],
        },
        {
            **shared,
            "id": "multilingual-search",
            "title": "Same world. Different words.",
            "eyebrow": "05 / ACROSS LANGUAGES",
            "description": "Ask in Chinese. Find the same blue planet in a collection of images.",
            "story": "Try the English and Chinese prompts in turn. This is cross-language image retrieval through the shared embedding space—not translation before inference.",
            "cover": "showcase/earth-cover.jpg",
            "accent": "teal",
            "modalities": ["text", "image"],
            "mode": "search",
            "query": "从太空看蓝色的地球，海洋上覆盖着白色的云层。",
            "suggestions": [
                "The blue Earth covered in white clouds, photographed from space.",
                "一个穿着白色宇航服的宇航员漂浮在地球上方。",
            ],
            "samples": images,
        },
        {
            **shared,
            "id": "visual-neighbors",
            "title": "A map of distant worlds",
            "eyebrow": "06 / SEMANTIC CONSTELLATIONS",
            "description": "Place Earth, Saturn, Pluto, a rocket, and a spacewalker on one similarity map.",
            "story": "Some connections are obvious; others are surprising. Explore the actual matrix instead of assuming that similar colors mean similar content.",
            "cover": "showcase/saturn-cover.jpg",
            "accent": "violet",
            "modalities": ["image"],
            "mode": "compare",
            "query": "",
            "task": "SentenceSimilarity",
            "samples": [*images, sample("", "launch.jpg", label="Artemis I launch")],
        },
    ]
    (ROOT / "demo/examples.json").write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Created six Orbit experiences", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Refresh audio metadata and examples without re-rendering the film",
    )
    args = parser.parse_args()
    if args.fetch or args.fetch_only:
        fetch()
    if not args.fetch_only:
        if not args.audio_only:
            film()
        audio_metadata()
        manifest()
