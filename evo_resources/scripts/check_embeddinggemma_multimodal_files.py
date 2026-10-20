"""Exercise actual PNG/WAV/MP4 decoding and multimodal CLI inference locally."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--transformers-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.transformers_path.resolve()))
    import av
    import numpy as np
    import soundfile as sf
    from PIL import Image, ImageDraw

    from evo.embedding_multimodal import prepare_files

    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (144, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 65, 80), fill="red")
    draw.ellipse((78, 20, 138, 80), fill="blue")
    image_path = args.output / "shapes.png"
    image.save(image_path)
    samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    audio_path = args.output / "tone-stereo-44100.wav"
    sf.write(audio_path, np.stack([samples, samples * 0.5], axis=-1), 44100)
    video_path = args.output / "shapes-2s.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 144, 96, "yuv420p"
        for i in range(20):
            frame_image = image if i < 10 else image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            frame = av.VideoFrame.from_ndarray(np.asarray(frame_image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    processed = prepare_files(args.package, videos=[video_path])
    assert processed["num_frames_per_video"].tolist() == [2]
    cases = {
        "image": ["--image", str(image_path)],
        "audio": ["--audio", str(audio_path)],
        "video": ["--video", str(video_path)],
        "mixed": [
            "--text",
            "Picture <|image|> clip <|video|> sound <|audio|>",
            "--task",
            "Document",
            "--image",
            str(image_path),
            "--audio",
            str(audio_path),
            "--video",
            str(video_path),
        ],
    }
    results = []
    for name, flags in cases.items():
        output = args.output / f"{name}.npy"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evo",
                "embed",
                str(args.package),
                "--transformers-path",
                str(args.transformers_path),
                *flags,
                "--output",
                str(output),
            ],
            check=True,
        )
        vector = np.load(output)
        assert vector.shape == (1, 768) and np.isfinite(vector).all()
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1) < 1e-6
        results.append({"case": name, "shape": list(vector.shape), "norm": norm, "passed": True})
    (args.output / "smoke.json").write_text(
        json.dumps({"passed": True, "video_sampled_frames": 2, "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
