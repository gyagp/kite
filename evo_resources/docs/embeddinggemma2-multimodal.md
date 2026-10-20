# EmbeddingGemma 2 multimodal ONNX

Published package: [webai-community/ai-models — EmbeddingGemma-2/onnx-webgpu](https://huggingface.co/webai-community/ai-models/tree/main/EmbeddingGemma-2/onnx-webgpu).
Publication commit: `10ae6d737f902ef8ed51776168285f21f9bfda2b`.
Generation instructions were added to both Hugging Face READMEs in commit
`82ff1a3bd221d7b42369eb4a0486251a6431bbab`, including pinned setup, source download,
provenance preparation, the multimodal export command, and validation commands.
The README checksum was updated; all model artifacts remain unchanged.
All 23 uploaded files were verified against remote sizes and LFS/Git hashes;
existing unrelated repository files remained unchanged. The publication includes
a standalone `inference.py` helper, processor dependencies, validation results,
and checksums. Its mixed text/image/video/audio output exactly matched Evo.

```bash
hf download webai-community/ai-models --revision 10ae6d737f902ef8ed51776168285f21f9bfda2b --include "EmbeddingGemma-2/*" --local-dir ./ai-models
```

The full multimodal package is in `artifacts/embeddinggemma-2-multimodal-fp16/`.
It supports text, images, audio, video frames, and interleaved combinations in
the same 768-dimensional embedding space. All three encoders use the official
checkpoint `google/embeddinggemma-2`, revision
`914f7f89142e33e77833254d9c9b90c3cef7303b` (Apache 2.0).

The three ONNX graphs and their external weights total **1.492 GB** (decimal).
The tokenizer adds approximately 32 MB. Keep each `.onnx.data` beside its graph.
Learned projections use FP16; normalization and sensitive accumulation retain
FP32 operations. The text backbone includes 194 native ORT RMS normalizations.

| Graph | Purpose | Inputs | Output |
| --- | --- | --- | --- |
| `model.onnx` | Shared backbone, pooling, normalization | int64 `input_ids`, `attention_mask` `[B,S]`; FP16 `media_embeddings` `[B,S,512]` | FP32 `sentence_embedding` `[B,768]` |
| `vision.onnx` | Image and video-frame soft tokens | FP16 `pixel_values` `[N,P,768]`; int64 `pixel_position_ids` `[N,P,2]`; int64 `pool_indices` `[N,L,9]` | FP16 `soft_tokens` `[N,L,512]` |
| `audio.onnx` | Audio soft tokens | FP16 `input_features` `[N,T,128]`; bool `input_features_mask` `[N,T]`; bool `blocked_mask` `[N,1,ceil(ceil(T/4)/12),12,24]` | FP16 `soft_tokens` `[N,ceil(T/4),512]`; bool `soft_token_mask` |

The official processor handles resizing, patchification, mel features, and
special tokens. Evo preserves the per-modality placeholder order, removes
padded soft tokens, and merges them before the shared backbone. Image pooling
uses 2D coordinates; audio uses the official strict local-window bounds.
The graphs support dynamic batch and sequence dimensions.

## Run

Run from `E:/workspace/project/evo`. The existing isolated processing environment
is `.cache/embeddinggemma2/deps`; `--transformers-path` selects it explicitly.

```powershell
# Image
.venv\Scripts\python -m evo embed artifacts/embeddinggemma-2-multimodal-fp16 `
  --transformers-path .cache/embeddinggemma2/deps --image photo.jpg --output image.npy

# Audio (mono or stereo; the helper resamples to 16 kHz)
.venv\Scripts\python -m evo embed artifacts/embeddinggemma-2-multimodal-fp16 `
  --transformers-path .cache/embeddinggemma2/deps --audio speech.wav --output audio.npy

# Video (official defaults: 1 sampled frame/s, at most 32 frames)
.venv\Scripts\python -m evo embed artifacts/embeddinggemma-2-multimodal-fp16 `
  --transformers-path .cache/embeddinggemma2/deps --video clip.mp4 --output video.npy

# One embedding for interleaved text and media
.venv\Scripts\python -m evo embed artifacts/embeddinggemma-2-multimodal-fp16 `
  --transformers-path .cache/embeddinggemma2/deps `
  --text "A product <|image|> demonstrated in <|video|> with sound <|audio|>" `
  --task Document --image product.jpg --video demo.mp4 --audio narration.wav `
  --output combined.npy
```

Repeat media flags to supply several files in one sample. Their placeholders
are filled in order within each modality. If a text description has no media
placeholders, the helper appends them. Media-only inputs receive no text task
prefix. Text-only batches use repeated `--text`; mixed CLI inputs use one text
sample. For mixed batches, pass the official processor's arrays to
`evo.embedding_multimodal.MultimodalSession.run()`.

`--dimensions 128`, `256`, or `512` truncates and renormalizes the output.
The combined context limit is 8192 tokens, including media tokens. Video uses
the visual frames; add an `--audio` input when audio content is also needed.
All model execution uses the native ORT runtime selected in `.evo/runtime.json`.
Preprocessing and the transfer of soft tokens between graphs currently use CPU
memory. Some ONNX shape/index operations execute on CPU as recorded in profiles.

## Validation

On 2026-10-12, **12/12 end-to-end cases passed** using the self-built WebGPU
plugin from `E:/workspace/project/ws-ort/onnxruntime`, revision
`86d60730c14f546ba1a4ba06b4887b53285117a5`, ORT host 1.31.0, RTX 5080, D3D12.
DLL SHA256:
`3082fe6fe9a425bcc49f821ee5b28daad8aace3aa622547f554aeac29af2eaf2`.

The independent reference runs the unmodified official Transformers FP32
multimodal model, followed by its configured masked mean pooling and L2 norm.
Fixtures include text, 70/280-token-budget images, portrait images, padded image
and audio batches, two interleaved images, 1/2-second audio, two video frames,
text+image+audio, and all four modalities together.

- Minimum cosine agreement: **0.999813749**.
- Maximum absolute embedding error: **0.002780655**.
- All outputs finite and unit-normalized.
- Acceptance thresholds: cosine >=0.999, absolute error <=0.005,
  norm error <0.001, chosen before native validation.
- Authoring wrappers independently matched the FP32 reference before export;
  the export-specific audio window rewrite matched exactly.
- Profiles confirm native GPU execution in all three graphs.

Evidence is retained in `artifacts/embeddinggemma-2-multimodal-fixtures/` and
`artifacts/embeddinggemma-2-multimodal-validation/webgpu-fp16.json` with all
graph/weight hashes and execution profiles. These deterministic synthetic
media cases test conversion parity, not retrieval quality or the full context
limit. Graph capture was disabled for correctness validation. **Captured
performance has not been qualified for this multimodal package.**

File-based smoke checks additionally exercise PNG, stereo 44.1-kHz WAV, and
MP4 decoding, official frame sampling, and the CLI's combined-modality path.
Their inputs and results are in
`artifacts/embeddinggemma-2-multimodal-validation/file-smoke/`.

## Reproduce

Export and preprocessing use Transformers source revision
`536ecc007387a50e77603bb5d92100e9b07514cc`, PyTorch 2.14.1+cpu, and torchvision
0.29.1. The isolated processor dependencies are pinned in
`requirements/embeddinggemma2-processing.txt`. With Evo's existing PyTorch and
native-runtime environment installed, recreate the processing layer with:

```powershell
.venv\Scripts\python -m pip install --target .cache/embeddinggemma2/deps `
  --no-deps -r requirements/embeddinggemma2-processing.txt

.venv\Scripts\python -m evo embedding-export models/embeddinggemma-2 `
  --multimodal --precision fp16 --output artifacts/embeddinggemma2-multimodal-new `
  --transformers-path .cache/embeddinggemma2/deps

.venv\Scripts\python -m evo embedding-validate artifacts/embeddinggemma2-multimodal-new `
  --fixtures artifacts/embeddinggemma-2-multimodal-fixtures --output validation.json
```

Regenerate independent references with `embedding-validate models/embeddinggemma-2
--prepare --multimodal --fixtures NEW_DIRECTORY --transformers-path
.cache/embeddinggemma2/deps`. Use a new model output directory for every export.
The native normalization registration makes this an ORT-targeted package.
