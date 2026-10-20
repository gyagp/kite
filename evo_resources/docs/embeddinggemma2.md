# EmbeddingGemma 2 for native ORT WebGPU

For text, images, audio, video, and mixed inputs, use the
[full multimodal package](embeddinggemma2-multimodal.md). The package documented
below is the smaller text-only variant.

The converted **text encoder** is in `artifacts/embeddinggemma-2-text-fused-fp16/`.
It contains the ONNX graph, external weights, tokenizer, task prefixes, and
`embedding_config.json`. Graph plus weights total approximately **543 MB**.
Image, audio, and video encoders are not included in this text package.

The source is `google/embeddinggemma-2` at immutable revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. Its downloaded checkpoint is in
`models/embeddinggemma-2/`. The model's license is Apache 2.0.
The export uses Transformers source revision
`536ecc007387a50e77603bb5d92100e9b07514cc` (5.19.0.dev0), installed separately in
`.cache/embeddinggemma2/deps`, and PyTorch 2.14.1 CPU. The existing GenAI
environment was not upgraded to this Transformers development version.

## Use

```powershell
.venv\Scripts\evo embed artifacts/embeddinggemma-2-text-fused-fp16 `
  --text "What causes the northern lights?" --task SearchQuery `
  --output embeddings.npy
```

Use `--task Document` for documents, or another task in the included
`config_sentence_transformers.json`. Repeat `--text` to make a padded batch.
`--dimensions 128`, `256`, or `512` truncates the vector and re-normalizes it.
Full vectors have 768 dimensions. This is an embedding model, without token
generation or KV caches. The helper rejects inputs longer than the model card's
8192-token context; the validation below covers up to 1024 tokens.

The graph takes `input_ids` and `attention_mask`, both int64 `[batch, sequence]`,
and returns float32 `sentence_embedding` `[batch, 768]`. Pooling averages
non-padding token outputs, including the task prefix, then L2-normalizes.
Attention remains bidirectional, with an inclusive local radius of 512 and
the configured full-attention layers. FP16 projections use FP32 accumulation
inside 194 native RMS normalizations.

## Validation

Following the WebGFX repository's testing guidance, validation and timing were
collected separately. Native tests explicitly select D3D12, record the actual
machine and GPU, retain input/model hashes, synchronize output readback, and
exclude model loading, tokenization, and warmup from inference timing.

The independent reference runs the official Transformers text model in FP32,
followed by masked mean pooling and normalization. All **13 cases passed**:
English, Chinese, code, a prefix-only input, a four-item padded batch, and exact
128/512/513/1024-token inputs. The 512/513 cases exercise the local-window edge.

- FP32 ONNX on CPU: maximum absolute embedding error below `2.4e-7`.
- Selected native FP16 graph: minimum cosine agreement **0.9998619**;
  maximum absolute error **0.0022523**. Outputs were finite and unit-normalized.
- Acceptance thresholds were set to cosine ≥0.999 and absolute error ≤0.005
  before validation. Native execution is evidenced by the retained ORT profile.
- This is conversion parity on a limited fixture set, not an MTEB or full
  retrieval-quality qualification. Some shape/scalar work runs on CPU.

Evidence: `artifacts/embeddinggemma-2-fixtures/` and
`artifacts/embeddinggemma-2-validation/{cpu-fp32,fused-fp16}.json`.

Revalidated on 2026-10-12 with your self-built native WebGPU plugin from
`E:/workspace/project/ws-ort/onnxruntime`, revision
`86d60730c14f546ba1a4ba06b4887b53285117a5`, using the ORT 1.31.0 host and
RTX 5080 D3D12 backend. All 13 cases passed again, with minimum cosine
`0.999861876` and maximum absolute error `0.002252247` against the FP32 reference.
The selected runtime is recorded in `.evo/runtime.json`; its DLL SHA256 is
`3082fe6fe9a425bcc49f821ee5b28daad8aace3aa622547f554aeac29af2eaf2`.
The report and execution profile are retained at
`artifacts/embeddinggemma-2-validation/latest-webgpu-86d60730c1.json` and its
referenced profile path. The CLI smoke test also returned a finite, unit-length
`[1, 768]` vector. These are correctness checks with graph capture disabled;
they do not establish captured performance on the new runtime.

## Measured performance

Machine **webgfx-100**, RTX 5080, Windows build 26300; ORT 1.31.0 and native
WebGPU plugin 0.4.0. Batch 1, 2 warmups plus 5 measurements per process.
These historical measurements used graph capture disabled and predate the
self-built runtime above. They do not satisfy the requested capture-enabled
performance comparison.
Order: ordinary FP16 → native-normalization FP16 → native-normalization FP16 →
ordinary FP16. The table combines 10 samples per model/prompt by median.
The baseline includes the completed Mobius encoder cleanup passes.

| Input tokens | Ordinary FP16 latency | Selected latency | Selected embeddings/s |
| ---: | ---: | ---: | ---: |
| 128 | 26.95 ms | 20.62 ms | 48.50 |
| 512 | 34.74 ms | 25.89 ms | 38.62 |
| 1024 | 40.18 ms | 39.26 ms | 25.47 |

The selected graph reduces median latency by 23.5% and 25.5% at the two shorter
lengths. The 2.3% difference at 1024 tokens is small. Ordinary FP16 had more
variation at shorter lengths; raw samples are retained. Latency includes CPU
input upload and the final embedding readback. Profiling was disabled.
No CPU builds or other GPU benchmarks ran during these measurements.

Evidence: `artifacts/benchmarks/embeddinggemma2-{fp16,fused}-{a,b}.json` and
`artifacts/embeddinggemma-2-validation/performance-summary.json`.

## Reproduce

```powershell
.venv\Scripts\python -m pip install --no-deps --target .cache/embeddinggemma2/deps `
  https://github.com/huggingface/transformers/archive/536ecc007387a50e77603bb5d92100e9b07514cc.zip

.venv\Scripts\evo embedding-export models/embeddinggemma-2 `
  --output artifacts/embeddinggemma2-new --precision fp16 --native-norms `
  --transformers-path .cache/embeddinggemma2/deps

.venv\Scripts\evo embedding-validate artifacts/embeddinggemma2-new `
  --fixtures artifacts/embeddinggemma-2-fixtures --output validation.json

.venv\Scripts\evo embedding-benchmark artifacts/embeddinggemma2-new `
  --fixtures artifacts/embeddinggemma-2-fixtures --output perf.json --repeats 5
```

Use a new output directory for each export. To regenerate reference fixtures,
run `embedding-validate models/embeddinggemma-2 --prepare --fixtures NEW_DIRECTORY`
with the same `--transformers-path`. The selected graph relies on ORT's legacy
`ai.onnx::SimplifiedLayerNormalization` registration. This is intentional for
the native ORT target; it is not a runtime-independent ONNX claim.

The full Mobius pipeline on the native-normalization graph was stopped during
symbolic shape simplification, with a diagnostic stack in
`.cache/embeddinggemma2/optimizer-diagnostic.log`. The selected artifact is the
validated direct native-normalization export; those incomplete Mobius outputs
are not recommended or registered in the UI.
