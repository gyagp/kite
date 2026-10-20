# Kite

A Windows desktop workspace for **your source-built ONNX Runtime WebGPU models**.
React + Electron UI, a loopback Python service, and isolated native inference.
Inference, conversations, attachments, and prompts remain on this computer. Model authoring can download explicitly selected source checkpoints.

**Kite 0.2 adds Qwen-Image Studio and integrates the model development engine.** Explore the [integration guide](docs/evo-integration.md) for the feature map, environment setup, and validation. Image Studio includes four original, locally generated demos: a floating cloud city, a bilingual poster, a glass camera, and sketch-to-photo editing. The GUI focuses on running and managing models; conversion and authoring remain developer tools.

[Initial verification results](docs/verification.md) record the tested runtime,
numerical checks, GUI checks, and remaining multimodal qualification boundary.

## Run on this machine

Double-click **`start.bat`**, or run:

```powershell
npm install
npm start
```

Kite is independent of the former `E:\workspace\project\evo` directory.
Its asset library is **`E:\workspace\project\kite\local`**:

- `local/models/` — original checkpoints and GGUF sources.
- `local/artifacts/` — ONNX exports, runtime DLLs, fixtures and benchmark evidence.
- `local/.venv/` and `local/.cache/` — the working Python/exporter/media environments.
- `gitignore/images/` — automatically saved generated PNGs, prompts and metrics; excluded from Git.
- `gitignore/profiles/` — opt-in chat and embedding traces and summaries; excluded from Git.

Image Studio shows live loading messages, four generation stages, denoising
steps, elapsed time, and cancellation. Progress survives switching pages during
generation. Its text encoder retains the original CPU weights between images,
avoiding a redundant GPU-to-CPU weight copy.

Enable **Profile generation**, **Profile response**, or **Profile embeddings**
before running to see the slowest shaders/operators and download ORT JSON traces
for Perfetto/Chrome tracing. Image profiles are stored beside the generated PNG.
Chat and embedding profiling starts a fresh model session and releases that worker
after saving its trace. Profiling adds overhead. GPU shader timestamps require a
runtime with the native WebGPU profiling bridge; otherwise Kite shows CPU operator
durations (including GPU dispatch/waits) and explicitly marks GPU timings unavailable.
The selected Image Studio DLL provides GPU timestamps; the currently selected
chat/embedding main DLL provides CPU operator traces. Runtime choices are preserved.

The Electron launcher finds `local/kite-assets.json` and uses the local Python.
The default chat/embedding runtime is selected by `local/.evo/runtime.json`;
Image Studio has its own runtime setting. Historical paths are resolved in memory
without rewriting hashed model/benchmark evidence. Your existing models,
conversations and image-gallery references are migrated with a database backup.

The desktop service chooses a free loopback port. Its exact URL and local API key
are in **Settings**. For a stable browser/API endpoint:

```powershell
npm run build
python -m backend.server --port 4317
# Open http://127.0.0.1:4317
```

`start-web.bat` performs these steps. The HTTP host requires Python 3.11+ and no
third-party Python packages. Native inference and authoring use a separate, compatible
environment with `numpy`, `transformers`, `tokenizers`, `onnxruntime`, and
`onnxruntime-genai`. The relocated local environment already provides them. Kite
does not install a replacement ORT wheel, rebuild ORT, or modify Evo.

## Workspaces

- **Image Studio:** generate and edit images with Qwen-Image. Each saved image has
  an end-to-end timing breakdown with phase percentages, the largest phase,
  model setup, verification, execution, saving, and remaining overhead. New runs
  also separate text/VAE initialization, input preparation, GPU transfers and
  cleanup. CUDA timing boundaries wait for work to finish; older runs display
  only measurements they actually recorded. Phase totals include their indented
  steps, so the breakdown does not double-count model loading.
- **Embeddings:** EmbeddingGemma 2 task prefixes; batch inference; 768/512/256/128
  dimensions with L2 renormalization; semantic document ranking; pairwise cosine
  matrix; vector inspection; JSON export. Supports text, images, audio, video, and
  mixed inputs with Evo's full export. Six photographic experiences show these modes.
  Text-file import splits on blank lines. Every result includes a performance
  card with inference latency, token/input throughput, preprocessing, and total
  request time. Detailed measurements are saved in Activity and JSON exports.
- **Chat:** streaming SLM responses; preserved package chat templates; system
  prompt, temperature, top-P, output limit and thinking controls; stop/regenerate;
  GitHub-flavored Markdown (headings, lists, tables, tasks, links), highlighted
  code blocks with copy buttons; saved conversations, rename, delete, and export.
  Each response shows live and saved prefill/decode statistics, token counts,
  first-token latency, and generation time. Maximum new tokens defaults to **4096**;
  changes are saved automatically for new chats and application restarts.
- **My models:** browse models you have chosen, whether or not their files are installed;
  import a package or scan a directory; inspect metadata; load/unload. **Delete
  from disk** removes the confirmed package folder and keeps the model card,
  which changes to **Not downloaded**. Downloading it again preserves its identity
  and saved conversations.
  **Discover models** searches `webai-community/ai-models` on Hugging Face,
  displays ONNX variants and sizes, downloads and verifies files, and imports
  complete packages. Local packages are referenced directly in place.
- **Activity:** retained outcomes, errors, token counts, timings, and JSON export.
- **Settings:** select inference Python and a source-built WebGPU DLL, configure
  context, view actual runtime/device provenance, export diagnostics, and connect
  API clients.

The app loads one model at a time and serializes GPU jobs. Loading a different
model releases the previous worker. Stop terminates the complete worker process
tree, including Windows venv launchers; the next request starts a clean session.
Job objects also release workers when their host process exits.

### Embedding timing definitions

**Embedding performance** appears above search, comparison, and raw-vector results,
and in **Activity → Run details**. It records:

- **Inference:** synchronous model execution and CPU readback. For multimodal
  inputs this includes vision/audio encoders and Evo's graph orchestration.
- **Inference throughput:** non-padding input tokens or embedded inputs divided
  by inference time. Tokens include task prefixes, special tokens, and expanded
  media tokens. Padding slots are reported separately in Timing details.
- **Preprocessing:** text tokenization or media decoding, sampling, resampling,
  processor work, and input preparation, measured before model execution.
- **Output processing:** output truncation, L2 normalization, validation, and
  conversion to serializable vectors.
- **Encoding total:** all worker encoding work, excluding model setup. Search
  adds query and document encoding; its details table reports both phases.
- **Total request:** backend job start through completed inference and search
  ranking, including worker startup/model setup when needed and IPC overhead.
  File uploads, browser rendering, and saving Activity are outside this boundary.
  Setup is zero for a reused model. Native model loading is a subset of setup.

Input counts, batch counts, dimensions, and sampled video frames describe the
measured workload, including media-only queries. The text export runs batches of
up to four; the current multimodal adapter processes one input per batch. Token
throughput varies with modality, input length, padding, and shader warmup. These
are host wall-clock measurements, not GPU kernel profiling. Embeddings have no
autoregressive decode phase. Older Activity records retain their available
timings and show missing measurements as unavailable.

### Chat timing definitions

The **Response performance** card records actual host wall-clock observations:

- **Prefill:** input submission through synchronized `get_logits()` CPU readback,
  before first-token sampling. Includes input upload, prompt evaluation (plus
  media encoders when present), and logits readback. Throughput is all prompt
  tokens—including system/history/template tokens—divided by this duration.
- **First token (TTFT):** inference start through the first `get_next_tokens()`
  readback. Excludes model loading, input preparation, and generator setup.
- **Decode:** first to last generated-token readback. Throughput excludes all
  tokens emitted by the first step and includes the loop's decoding/streaming
  overhead. Zero/one-step responses show no invented decode rate.
- **Generation:** input submission through completion of the generation loop.
  **Timing details** separately shows input preparation, generator setup, model
  loading for this request, and average decode milliseconds per token.

Statistics update while streaming and persist with messages and Activity records.
Stopped responses retain the last available snapshot, marked partial. Special
tokens such as EOS can contribute to output counts. Older conversations without
separate prefill data show that metric as unavailable. These are interactive
measurements, not GPU kernel profiling; the extra prefill synchronization has a
small readback cost. The measurement boundary follows GenAI's `AppendTokens`,
`SetInputs`, and `GetLogits` contracts in `src/generator/generators.cpp` and
`src/ort_genai_c.cpp` of the local source checkout.

## EmbeddingGemma 2 multimodal

Select **EmbeddingGemma 2 · Multimodal** in Embeddings, or click
**Multimodal examples** to explore the **Orbit Collection**:

1. **A planet without a keyword** — find NASA photographs with natural language.
2. **Hear it. Find it.** — use a spoken mission briefing to retrieve images.
3. **Twelve seconds to orbit** — search an authored three-scene photo film.
4. **One mission. Every dimension.** — compare text, image, audio, video, and one
   combined input in a 5×5 cosine matrix.
5. **Same world. Different words.** — search the same images in Chinese or English.
6. **A map of distant worlds** — compare Earth, Saturn, Pluto, a spacewalker, and a rocket.

Preview photography, play the film or narration, and edit the query before
choosing **Open in workspace** or **Run demo**. The bright blue interface uses
larger navigation, chat, and form text.

Files are bundled in `demo/`; no sample download is needed. Scores come from
actual inference, with no preset similarity values. The film is a montage of NASA
photographs, and the narration is generated with Azure AI Speech. Bundled audio
plays locally; running the examples does not call Azure. Image credits and source
links are included in the gallery and `demo/showcase/credits.json`. These examples
show input handling, not a retrieval benchmark. NASA does not endorse Kite.

Each input row produces **one embedding**, even if it combines text with several
media files. Search queries can also be media-only. Attach PNG/JPEG/WebP images,
WAV/FLAC/MP3 audio, or MP4/MOV video (64 MB/file, up to 8 files/input). Inputs
must fit the combined 8192-token model context after media-token expansion.
Video uses visual frames sampled at 1 fps, at most 32 frames; its soundtrack is
not automatically included. Attach an audio file separately to include sound.

Kite uses its integrated Evo `prepare_files` and `MultimodalSession` with all learned
computation in the selected source-built ORT WebGPU provider. All three graphs
(`model`, `vision`, `audio`) are checked and loaded. The isolated dependencies in
`local/.cache/embeddinggemma2/deps` supply the official Transformers processor,
image/audio/video decoders, and audio resampling. Kite does not load PyTorch
model weights. The helpers are now included in Kite’s `evo/` package. Changing the asset directory in Settings changes where external dependencies and model assets are found.

Mixed text can use `<|image|>`, `<|audio|>`, and `<|video|>` placeholders to control
interleaving; counts must match each modality's attachments. Without placeholders,
media is placed after text. Pure-media inputs receive no text task prefix.
Truncated output vectors are L2-normalized again. Result timing separates CPU
preprocessing from native graph inference/readback. Model load is excluded from
those two figures.

The smaller text-only export remains available. The Qwen3.5 package supplied by Evo
is still a text-only **chat** model; embedding support does not add vision to it.

### GenAI multimodal chat

Kite implements an image/audio chat integration for **compatible ORT GenAI
multimodal packages**: capability discovery from declared vision/speech graphs,
PNG/JPEG and WAV upload/preview, package chat templates, architecture-specific
media markers, native `MultiModalProcessor`, expanded media-token context
accounting, `Generator.set_inputs`, and streaming generation. Marker contracts
follow upstream GenAI `examples/python/common.py`: Phi-3 vision, Phi-4 multimodal,
Qwen VL, Mistral vision, and structured Gemma-style content. The selected WebGPU
build must support the package's operators. All graphs, preprocessing assets,
tokenizer, and templates must be included by the exporter.

EmbeddingGemma multimodal inference is end-to-end verified against Evo outputs;
other GenAI image/audio chat packages require their own qualification. Chat video,
microphone recording, tool calling, and remote inference are
not implemented. Model conversion is a developer-tool workflow. The UI enables chat attachments only when the package declares
the necessary encoder graphs.

## Hugging Face model downloads

My Models contains only explicitly added models. Hugging Face discovery results
are not automatically added to the library. Bundled repository metadata provides
download links for matching library entries and an offline discovery fallback;
it is not a list of architectures validated by Kite. Deleting files keeps the
chosen model in My Models so it can be downloaded again.

In **My models → Discover models**, search the public
[`webai-community/ai-models`](https://huggingface.co/webai-community/ai-models)
repository, select a model and ONNX variant, choose a local directory, then click
**Download & import**. GGUF and non-ONNX formats are excluded. Package contents
and total size are shown before download. Runtime compatibility is checked when
loading; presence in the repository does not guarantee every architecture or
operator is supported by the selected DLL/GenAI host.

The downloader pins the repository commit, verifies Git blob or LFS SHA-256
hashes for every file, and records provenance in `kite-hub-source.json`. Complete
packages are imported automatically. Downloads run independently of inference;
cancelled/interrupted files remain under `.kite-downloads/` in the chosen
directory. Select the same variant/revision/directory to resume. Existing
unrelated directories are never overwritten. Only data/configuration assets are
downloaded; repository Python code is not executed. Model-specific licenses and
usage details are available through **View source**.

## Local API

Supported endpoints:

| Endpoint                    | Behavior                                                                  |
| --------------------------- | ------------------------------------------------------------------------- |
| `GET /v1/models`            | Registered model IDs and capabilities                                     |
| `POST /v1/embeddings`       | Strings or Kite multimodal inputs, float output, optional dimensions/task |
| `POST /v1/chat/completions` | Text or compatible image/audio content, streaming or complete response    |

Use `Authorization: Bearer <local API key from Settings>`. Python client examples
are generated with your actual model IDs in Settings. Embedding vectors use the
package's `SearchQuery` task by default; set `extra_body={"task": "Document"}`
when indexing documents with the OpenAI Python client. The client is just an HTTP
client here; requests are sent to `127.0.0.1`.

This is a **subset** of the OpenAI-compatible API, not a complete server
replacement. Streaming emits content chunks and a terminal `[DONE]`. Remote image
URLs are not fetched; use PNG/JPEG base64 data URLs. Audio accepts WAV. Tools,
structured output, and custom stop sequences are rejected explicitly. A busy
GPU returns HTTP 409. Disconnected streaming API clients cancel their job.

For multimodal embedding inputs, upload media through `POST /api/attachments`
with `{ "name": "photo.png", "data": "<base64>" }`, then reference returned IDs:

```json
{
  "model": "<multimodal model ID>",
  "input": [{ "text": "A picture", "attachments": [{ "id": "<attachment ID>" }] }],
  "dimensions": 768,
  "task": "Document"
}
```

This structured `input` is a Kite extension. The GUI job API also supports
`samples` for embedding jobs and `query_attachments`/structured `documents` for
cross-modal search. File paths supplied by clients are not trusted; media is
resolved from registered attachment IDs.

## Development and verification

```powershell
# Terminal 1 (HTTP service)
python -m backend.server --port 4317
# Terminal 2 (Vite; proxies local API routes)
npm run dev
# Open http://127.0.0.1:5175

npm run build
npm test
npm run test:ui
npm run test:chat-ui
# Full browser checks that load the real local models:
node scripts/ui-smoke.mjs --inference
# Native checks in a separate disposable application database:
local\.venv\Scripts\python.exe scripts/native-smoke.py
local\.venv\Scripts\python.exe scripts/multimodal-smoke.py
# Against a disposable service on 4318 (downloads ~96 MB for verification):
python scripts/hub-smoke.py
node scripts/multimodal-ui-smoke.mjs --inference

# Produce release\win-unpacked\Kite.exe and its resources:
npm run pack
# Exercise the packaged app with disposable state:
npm run test:desktop
```

UI tests use a fresh, headless Edge profile; they never connect to personal
browser sessions. Set `KITE_TEST_URL` to choose a test instance. Inference UI
checks create a test conversation in that instance. Native checks compare
English/Chinese vectors with Evo's independently generated FP32 fixtures, check
renormalization, search ranking, real generation, API streaming, cancellation,
and reload. Reports/screenshots are written to `test-results/`. UI timing values
are observations from interactive runs, **not controlled performance benchmarks**.

The unpacked Windows distribution includes the GUI and Python service; it does
not bundle Python, model weights, or the user's runtime DLLs. It includes Evo's
implementation, documentation, recipes, and CLI resources. Keep its resource
directory beside the executable. On a different machine, install a compatible
inference environment and configure the runtime in Settings.

## Configuration and persistence

| Setting           | Purpose                                                |
| ----------------- | ------------------------------------------------------ |
| `KITE_ASSET_ROOT` | Override the local asset library directory             |
| `KITE_IMAGE_DIR`  | Override automatic image output directory               |
| `KITE_PYTHON`     | Python executable used to launch the desktop HTTP host |
| `KITE_DATA_DIR`   | Override browser-server data directory                 |
| `--data-dir PATH` | Explicit HTTP server data directory                    |

Development/browser data lives in `.kite/`; packaged desktop data lives in
Electron's per-user application-data directory. `kite.db` stores settings, model
references, conversations, and the last 100 operation summaries. Attachments are
stored in its `attachments/` subdirectory. Removing a conversation currently
retains uploaded media on disk. Completed job results/events have a bounded
32-job in-memory replay window; export embedding results before restarting.

The server binds only to loopback, checks Host/Origin, requires JSON mutations,
and authenticates `/v1` requests. The GUI API is intended for the local app, not
untrusted local users; it can select executable Python/DLL paths. Do not expose
it through a public tunnel/reverse proxy. Electron uses a sandboxed renderer,
context isolation, a narrow file-picker bridge, and no Node integration. No
remote model code is executed; tokenizers use local files only.

Runtime sidecar hashes are checked before registration. Chat/embedding graph capture remains
off; Image Studio's optional capture mode uses FP32 accumulation on its D3D12 runtime. CPU shape/control kernels can coexist with WebGPU execution; Kite does not
claim that every graph node runs on the GPU. The exported model's numerical and
task-quality limitations still apply.

## Image generation performance

All Kite ORT WebGPU inference uses **D3D12**, including Image Studio, chat,
embeddings and current profiling/performance scripts. Historical Vulkan benchmark
results do not describe the current backend.

Image Studio uses native D3D12 WebGPU FP16 denoising,
with graph capture off by default. The optional capture compatibility mode retains
FP32 accumulation because full-image tests still expose inconsistencies in FP16
capture on this runtime. Experimental runtime/model candidates that failed those
checks are not selected.

The current NVIDIA D3D12 runtime uses constant-index accumulator accesses in
MatMulNBitsWideTile. Warm 512×512 images measured **15.60–15.98 s overall /
6.32–6.40 s denoising**, down from 29.55 s / 19.83 s, with identical generated pixels.
See [measurements and validation](docs/qwen-webgpu-profile.md) for scope and limits.

The official Qwen VLM backbone produces the same prompt features without loading
its unused 1.25 GB vocabulary head, retaining intermediate hidden layers, or
building an autoregressive KV cache. CPU text/VAE components remain available
between images, and unchanged model-file hashes are reused after initial
verification. **Unload model** or switching models releases that worker and its
CPU cache. Stop also terminates the worker tree.

New images save automatically under `gitignore/images/<run id>/image.png`, alongside
request, timing and provenance JSON. Your existing user-generated image was migrated
there. Generated-image folders and the local asset library are excluded from Git.
See [standalone migration and performance verification](docs/standalone.md).
