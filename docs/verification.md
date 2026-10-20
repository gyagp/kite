# Initial Windows verification — 2026-10-12

**Latest update:** six Orbit Collection experiences, larger type, guarded model
file deletion, and a remembered 4096-token chat default are verified. The final
section records the checks; earlier release limitations below are historical.

Kite was built and tested against the existing Evo artifacts on Windows with an
NVIDIA GeForce RTX 5080. The packaged executable is
`release/win-unpacked/Kite.exe`.

| Check                                     | Result                            |
| ----------------------------------------- | --------------------------------- |
| TypeScript and Vite production build      | Passed                            |
| Python backend tests                      | 17 passed                         |
| Ruff / Prettier                           | Passed                            |
| Production npm dependency audit           | 0 vulnerabilities                 |
| Real native embedding and chat API checks | 6 passed                          |
| Browser GUI checks, including inference   | 9 passed, no renderer exceptions  |
| Packaged Windows application checks       | 4 passed                          |
| Vite development API proxy                | Health and model discovery passed |

Development tooling has 8 moderate transitive npm advisories at the recorded
dependency resolution. The production dependency audit is reported separately;
it is not a claim that the entire toolchain is advisory-free. Electron is 41.10.7.

## Native evidence

- Runtime: source-built `onnxruntime_providers_webgpu.dll`, source revision
  `86d60730c14f546ba1a4ba06b4887b53285117a5`.
- DLL SHA-256:
  `3082fe6fe9a425bcc49f821ee5b28daad8aace3aa622547f554aeac29af2eaf2`.
- Host ORT 1.31.0, GenAI 0.17.1, D3D12, graph capture disabled.
- EmbeddingGemma 2 English/Chinese 768-dimensional vectors were finite and
  normalized. Cosine agreement with Evo's independent FP32 fixtures:
  **0.9999836404** and **0.9999708547**.
- Authenticated API batching crossed the four-item worker batch boundary;
  128-dimensional truncated outputs were renormalized. Reload also returned
  normalized 256-dimensional output.
- Semantic search ranked the aurora document above the Mars distractor.
- Qwen3.5-4B generated `Paris` for the capital-of-France check. Streaming chunks,
  the terminal `[DONE]`, and saved conversation retrieval passed.
- Cancelling active generation released the worker; another model loaded and
  ran successfully afterwards.

These are integration and limited numerical checks, not broad language-model
quality evaluations or controlled performance measurements.

## UI and distribution

The browser tests exercised semantic search, actual pairwise cosine results,
JSON download, model import errors, capability-gated attachments, streaming chat,
conversation restoration after reload, activity, settings, and a compact desktop
viewport. Screenshots were inspected.

The actual packaged `Kite.exe` launched with disposable state, discovered Evo,
ran semantic search with the correct source-built WebGPU DLL, exposed the narrow
native file-picker bridge without Node integration, and stopped its HTTP service
when the application closed. No packaged application or inference processes
remained after the test.

Reports and images are retained locally under `test-results/`:

- `native-report.json`
- `ui-report.json`
- `desktop-report.json`
- `desktop.png`, `semantic-search-result.png`, `chat-result.png`, and other UI views

## Qualification boundary

At initial release, both installed Evo packages were text-only. Image/audio GenAI integration and
model-specific media marker contracts are implemented, with unit coverage for
capability detection, missing graphs, marker order, and rejection of media by
text-only exports. **No real multimodal ONNX package was available for end-to-end
image/audio verification.** Video and multimodal embedding inference are not
implemented at that point. The multimodal embedding update below supersedes this
initial limitation. Other GenAI chat packages still require their own qualification.

## Blue theme, Markdown, and chat timing update

The subsequent user-requested update replaced the green theme with blue,
including application icons and embedding visualizations. Chat gained highlighted
code blocks with copy controls and an expanded Markdown layout, plus live/saved
prefill and decode statistics.

- 20 Python tests passed, including fixed-clock timing arithmetic, zero/one-step
  decoding, and exclusion of the entire first generated step from decode rates.
- The native smoke suite passed with synchronized prefill/TTFT/decode checks,
  one-token generation, model reuse, streaming, persistence, and cancellation.
- The chat UI fixture checked headings, emphasis, code highlighting/copy, lists,
  tables, tasks, safe links, streamed Markdown, and the performance card. Fixture
  values are explicitly synthetic UI inputs, not performance measurements.
- The packaged app generated and rendered a real Markdown Python example and
  saved its measured statistics. That individual 50-input/23-output-token run
  observed 406.84 ms prefill, 407.18 ms TTFT, and 50.84 decode tokens/s over the
  22 tokens after the first step. These are interactive cold-run observations,
  not a controlled benchmark or a general performance claim.
- Production build, Ruff, Prettier, and production dependency audit passed.

Evidence: `test-results/chat-ui-report.json`, `native-report.json`,
`desktop-report.json`, `desktop-chat-metrics.json`, and `desktop-chat.png`.
Timing definitions and their synchronization overhead are documented in README.

## Full multimodal embeddings, sample gallery, model downloads, and light theme

Evo's `artifacts/embeddinggemma-2-multimodal-fp16` is now detected and used directly
without copying or changing its weights. Kite reuses the official preprocessing
and graph orchestration from Evo, with each of its three ONNX sessions explicitly
bound to the selected source-built WebGPU DLL.

- **27 unit tests passed**, including graph/dependency validation, attachment-ID
  resolution, media placeholder counts, download path containment, Git/LFS file
  hashes, cancellation with HTTP-range resume, and protection of existing folders.
- **Real file-based inference passed** for PNG, stereo 44.1-kHz WAV, MP4, and mixed
  text/image/audio/video input. Cosine agreement with Evo's native CLI outputs was
  above 0.99999 for all four. The image also agreed with the independent official
  FP32 reference at cosine **0.999934924**. Outputs were finite and normalized;
  128-dimensional truncation was renormalized through the authenticated API.
- **All four bundled examples ran successfully**: image/text search, audio/text
  comparison, short-video search, and a five-input mixed-modality cosine matrix.
  Actual video sampling was checked (two frames for Evo's two-second fixture;
  three for Kite's three-second demo). Video contributes frames, not soundtrack.
- **Hugging Face search/download/import/generation passed** with the actual
  `SmolLM2-135M-Instruct/onnx-webgpu` package at repository revision
  `6d5cd933ae1008d4cb78f88730248c464d3a50d1`: 95,984,192 bytes across nine verified
  files. The downloaded package then generated text on the native WebGPU runtime.
  This is an integration check, not a qualification of every repository model.
- **GUI checks passed** for the light palette, sample gallery, uploaded image
  previews, media-only audio queries, H.264 video playback, real image-search
  results, the real 5×5 mixed-modality matrix, and live repository browsing.
- **The packaged Windows executable passed** startup, direct Evo discovery,
  bundled-media loading and multimodal inference, Markdown chat with measured
  timings, and shutdown with no orphan service. Existing conversations/settings
  use the same persistent application data.
- TypeScript/Vite production build, Ruff, Prettier, and the production npm audit
  passed. No performance benchmark claim is made from these interactive checks.

New evidence: `test-results/multimodal-report.json`, `multimodal-ui-report.json`,
`hub-report.json`, and `desktop-report.json`; screenshots include
`multimodal-examples.png`, `mixed-similarity-result.png`,
`huggingface-model-browser.png`, and `desktop-multimodal.png`.

## Orbit Collection, model deletion, and remembered chat limit — 2026-10-13

The gallery now includes six experiences with NASA photography, local synthetic
narration, a 12-second authored photo montage, and Chinese/English queries.
The interface has larger text while retaining its bright blue palette. Credits
and source URLs are bundled; the montage is identified as authored photo media.

- **All 33 backend tests passed**, including exact-path disk deletion, library-only
  removal, overlapping model protection, protected workspace roots, declared
  example asset serving, and persistent chat preferences.
- **All six experiences ran on the actual native WebGPU runtime.** Image, audio,
  video, and multilingual searches returned measured scores; mixed input and
  image comparison returned 768-dimensional vectors. File parity with Evo's
  image/audio/video/mixed CLI fixtures remained above 0.99999 cosine; the
  independent FP32 image check was 0.999934924.
- **Browser checks passed** for six gallery cards, modality filters, audio
  playback/waveform/transcript, 12-second video metadata, editable multilingual
  prompts, one-click audio-to-image inference, uploads, live Hugging Face browsing,
  and a compact viewport without horizontal overflow or renderer exceptions.
- **Disk deletion passed through the actual GUI**, using only a script-created
  disposable package. The button required acknowledgement, removed that folder,
  and removed its library entry. No real user model was deleted by testing.
- **The packaged Windows app passed seven checks**, including actual semantic
  search, source-built runtime provenance, bundled mixed-media inference producing
  a 5×5 matrix, generated Markdown chat with measured and saved statistics, and
  service shutdown. A fresh workspace started at **4096** maximum new tokens.
  A change to **1024** survived both a new chat and a full application restart
  using the same disposable database.
- Production build, Windows packaging, Ruff, Prettier, and `git diff --check`
  passed. Screenshots of the gallery, previews, result page, and chat were inspected.

Evidence: `test-results/orbit-ui-report.json`, `multimodal-report.json`,
`multimodal-ui-report.json`, `desktop-report.json`, `orbit-native.log`, and
`orbit-pack.log`. Screenshots include `orbit-gallery.png`,
`orbit-gallery-compact.png`, `orbit-audio-preview.png`, `orbit-film-preview.png`,
`orbit-audio-result.png`, and `desktop-chat.png`.

These are integration checks, not a retrieval-quality or performance benchmark.
Video inference uses sampled visual frames; soundtrack requires a separate audio
attachment. Only the maximum-token preference is persisted by this update.

## Azure Speech narration — 2026-10-13

The Orbit briefing was regenerated with the user's Azure AI Speech resource in
Japan East using `en-US-AvaMultilingualNeural`. The 24-kHz, 16-bit mono WAV is
7.449583 seconds long. Its SHA-256, measured waveform, duration, transcript, and
voice credits are recorded with the bundled assets. The generator retrieves a
key through Azure CLI or reads `SPEECH_KEY`; credentials are not written into
the repository or packaged app. Generation calls Azure; playback and embedding
inference use the resulting local WAV.

The rebuilt Windows package passed focused checks for served audio hash,
playback, duration, waveform, and credits. Actual audio-to-image inference ranked
the astronaut spacewalk image first. The combined text/image/new audio/video
example also produced its real 5×5 similarity matrix on the source-built WebGPU
runtime. Production build, packaging, Ruff, and whitespace checks passed.

Evidence: `test-results/azure-speech-report.json`, `azure-speech-preview.png`,
`azure-speech-ui.log`, and `azure-speech-pack.log`. Voice provenance is in
`demo/showcase/speech.json`. These are focused integration checks, not a general
retrieval-quality evaluation.

## Embedding performance statistics — 2026-10-13

Embedding results now show inference latency and throughput (tokens/s and
inputs/s), preprocessing, total request time, input/token/batch counts, output
dimensions, and sampled video frames. Timing details separate model setup,
native model loading, output processing, and search ranking; searches retain
separate query and document measurements. Activity and JSON exports preserve
these statistics. Older records show unavailable fields without inventing data.

- **35 backend tests passed**, including summed query/document throughput with
  unequal durations, non-padding token counts, and zero-duration rate handling.
- **Five real packaged-app workloads passed**: text search with model loading,
  a reused text model with five variable-length inputs across two batches,
  Azure audio-to-image search, mixed text/image/audio/video embedding, and a
  media-only video search query. Measured phase totals, throughput arithmetic,
  normalized vectors, query metadata, and zero setup for reused models passed.
- **UI checks passed** for the performance card, query/document details table,
  JSON view, restored Activity records after reload, and a compact desktop
  viewport. Screenshots were inspected; no renderer exceptions or page overflow.
- Production build, Windows packaging, Ruff, Prettier, and whitespace checks
  passed. The timing definitions and boundaries are documented in README.

The five-input mixed-media observation processed 3,881 non-padding tokens and
24 video frames, with 2,094.25 ms inference and 9,194.60 ms preprocessing. Its
11,304.12 ms backend request used an already loaded model. This is an interactive
observation on the current workload, not a controlled performance benchmark.

Evidence: `test-results/embedding-performance-report.json`,
`embedding-performance-text.png`, `embedding-performance-multimodal.png`,
`embedding-performance-activity.png`, and `embedding-perf-pack.log`.
Reproduce with `node scripts/embedding-perf-smoke.mjs` after packaging and closing
the normal Kite window; the script uses a separate disposable data directory.

## Evo integration and Qwen-Image Studio — 2026-10-18

Kite 0.2 includes Evo's implementation, Python API/CLI, 21 integrated operation
forms, recipes, requirements, tests and documentation. Source-file hashes are in
`evo_resources/migration.json`; the original directory had no Git revision.
Model weights, runtimes and dependency environments remain referenced in place.
The feature map and remaining environment requirements are documented in
[evo-integration.md](evo-integration.md).

- **41 Kite backend tests passed.** The migrated Evo lightweight suite passed
  **73 tests**, skipping two optional suites; the isolated Qwen image/translation
  suite passed **11 tests**. Graph export/CPU-reference tests exercise tiny models;
  the merge did not re-export every multi-GB checkpoint.
- **Four real Image Studio demos passed:** 768×768 cloud city, 768×1024 bilingual
  poster, 768×768 glass camera, and 512×512 boat-sketch editing. Cold total times
  were 56.59, 62.22, 55.53 and 46.63 seconds. These include model setup, shader
  compilation, process startup and readback; they are not warmed comparisons.
- The first FP16 sky-garden attempt produced a non-finite prediction at step 8.
  FP32 accumulation, already used by Evo's editing workaround, resolved it for
  these runs. Studio applies it to prefix and denoising. Captured versus uncaptured
  sky-garden final latents are exactly equal, maximum absolute difference **0**.
- **Browser and packaged-app UI checks passed:** four model-generated previews,
  reference import, before/after divider, fullscreen viewer, image history and
  phase statistics; all 21 forms, actual ONNX inspection and report artifacts,
  relocated-recipe planning, 66 historical observations/filtering, and wishlist
  editing/persistence. The browser test also checked a 1000×760 viewport. No
  renderer exceptions were recorded.
- **Packaged image inference passed:** stop/retry, completed PNG and phase
  metrics, eight denoising steps, verified FP32 provider option, cleanup of only
  the job's scratch files, persisted gallery, and switching back to the existing
  embedding runtime. That cold 512×512 run took **54.16 seconds** end to end.
- **Existing native regressions passed:** Qwen chat/streaming/metrics and
  persistence; English/Chinese embedding fixture agreement; batching, search,
  cancellation/reload; original media fixtures and all six multimodal examples.
- Production build, Windows packaging, Ruff and whitespace checks passed.

The image denoiser/prefix execute with the selected native Vulkan WebGPU DLL;
text/reference encoding and VAE decoding use local PyTorch/CUDA. All images are
generated by the model, and previews retain its imperfections. These checks do
not establish universal image quality, arbitrary multi-reference editing support,
or a performance advantage over the GGUF baseline.

Evidence: `test-results/image-generation-report.json`,
`capture-control-report.json`, `merged-ui-report.json`,
`merged-ui-desktop-report.json`, `packaged-image-inference-report.json`,
`native-report.json`, `multimodal-report.json`, `merged-backend-tests.log`,
`evo-suite.log`, `evo-image-suite.log`, and `merged-pack.log`.
Reproduce with `scripts/merged-ui-smoke.mjs` (add `--desktop` for the packaged
app) and `scripts/merged-desktop-inference.mjs`. The image generation script
expects a disposable source service at port 4325 and rebuilds the demo previews.


## Standalone migration and selected image optimizations — 2026-10-18

See [standalone.md](standalone.md) for the completed asset/environment migration, automatic image saving, performance measurements, tests, and rejected runtime candidates. All native and packaged checks ran while the original Evo path was unavailable. The final 512×512 packaged sample took 44.35 seconds cold and 16.53 seconds warm, with byte-identical outputs. The new cold image-edit check took 49.95 seconds; no cold-edit improvement is claimed.
