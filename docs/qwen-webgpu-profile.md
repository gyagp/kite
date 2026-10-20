# Qwen-Image WebGPU profiling

## Current backend: D3D12 (2026-10-19)

### Selected optimization: constant row indexing

The selected runtime is now
`local/artifacts/runtimes/ort-webgpu-qwen-image-d3d12-unrolled/`, provider SHA-256
`7e9f131fcd6c31f89755b46523892d7bfbb41b26b90df881bd0bcee11371c133`.
It removes dynamic row indexing from the MatMulNBitsWideTile direct-reduction
loop and output stores, using constant indices for each accumulator. The change
is scoped to NVIDIA D3D12, covers 16/32-row tiles, preserves arithmetic order and
has a distinct shader-cache key. This lets the compiler optimize accumulator
access; register-spill reduction has not been independently verified in GPU ISA.

| 512×512 application observation | Previous | Optimized |
| --- | ---: | ---: |
| Warm total | 29.55 s | **15.60–15.98 s** |
| Warm denoising execution | 19.83 s | **6.32–6.40 s** |
| First worker request, total | 53.11 s | 43.28 s |
| Reference editing, total | 40.25 s | 25.91 s |
| Reference editing, denoising | 20.40 s | 6.84 s |

This is about 46% less warm end-to-end time and 3.1× faster denoising in these
observations. Generation and editing PNGs are byte-identical to their previous
D3D12 baselines. The baseline was not interleaved with the new runs; the editing
validation overlapped lightweight backend tests. A frozen 768×1024 poster fixture
also repeats exactly over two complete eight-step trajectories (21.61/21.38 s
denoising). These fixture times exclude the encoder, prefix and image decoder.

Final packaged profiling recorded 17,951 GPU dispatches and 6.47 s summed GPU
time. Denoiser MatMulNBitsWideTile now takes 5.44 s (84.1%), down from 18.97 s;
FlashAttention is 0.65 s (10.0%). Profile downloads, progress/navigation and gallery
reload passed. FP32 capture mode completed in 19.48 s / 10.05 s denoising and
exactly matched its pre-optimization D3D12 PNG. The second unprofiled warm sample
was 15.60 s overall / 6.40 s denoising, again with identical output.

The final binary passes 38 native numerical cases, including biased batched
matrices and partial tiles, with CPU fallback rejected; profiling lifecycle and
Dawn persistence/corruption/concurrency tests also pass on D3D12. Eight unittest
methods ran without skips. Shader generator: 89 tests passed. Kite backend:
54 tests passed; runtime selection tests: 3 passed. Runtime support files now
receive the same hash and path-containment checks as root DLLs.

An additional K-loop unroll saved only ~0.15 s in a repeat but increased first-use
cost, so it was not selected. Updating DXC alone was not faster. The selected
runtime retains the validated DXC 1.8.2502.8 sidecars and does **not** enable the
experimental Agility SDK.

The Agility investigation exposed SM 6.10 and advertised F16→F16/F32 matrix
capabilities on this GPU, but pipeline creation failed with E_OUTOFMEMORY for
quantized and plain subgroup-matrix shaders, including reduced tiles and FP32
accumulators. No GPU exhaustion was demonstrated; the root cause remains open.
Those isolated candidates and their reproducers were retained in the ORT worktree,
and their experimental shader/build changes were removed from the selected source.
No system DLLs, drivers, or Developer Mode settings were changed.

Evidence: `test-results/d3d12-unroll-validation.json`,
`test-results/d3d12-unrolled-app-*.json`,
ORT worktree `build/d3d12-unroll-final-*.log/json`,
`build/d3d12-unroll-final-poster/`, and `build/runtimes/d3d12-unroll-final/`.

### D3D12 baseline and policy

Kite always uses **ORT WebGPU D3D12** for its ONNX inference, including Image
Studio, chat, embeddings and current profiling/performance scripts. Vulkan was an
earlier experimental selection, not a Qwen-Image requirement. The NVIDIA Vulkan
copy workaround in the retained runtime does not execute on D3D12. The Python
image session helper rejects attempts to select another backend.

All Vulkan measurements below are historical. They must not be presented as
current D3D12 performance. The persistent Dawn cache and the shared mmap/CPU
encoder storage improvements remain enabled. Text/reference encoding and VAE
decoding still use their existing PyTorch CUDA implementation.

The first complete 512×512 D3D12 generation measured 53.11 seconds cold and
29.55 seconds warm, including 19.95/19.83 seconds of denoising. The two PNGs are
byte-identical (SHA-256
`595d33e7c3b1c8604b3aff5ef6aae348b90bebf0dd9ddca1285359cbcf6a5f28`).
Warm phase totals: preparation 2.72 s, prefix 3.11 s, denoising 22.89 s,
decoding/saving 0.83 s. The earlier encoder optimization remains effective;
denoising is now the dominant D3D12 cost. First reference editing measured
40.25 seconds total / 20.40 seconds denoising. These are individual application
observations with eight uncaptured steps, not a controlled cross-backend benchmark.
Backend-specific FP16 arithmetic can produce different PNGs from Vulkan; quality
is checked visually and repeats are compared within D3D12.

Evidence: `test-results/d3d12-app-*.json` and `scripts/image-d3d12-smoke.py`.

Editing repeat: 39.41 s / 20.42 s denoising, byte-identical to the first edit.
The final packaged D3D12 profiling UI test passed with 17,951 GPU events and a
PNG identical to the unprofiled D3D12 generation. It measured 20.18 s summed GPU
time; denoiser MatMulNBitsWideTile consumed 18.97 s (94.0%), FlashAttention 0.65 s
(3.2%). The fresh-worker profiled total was 52.89 s. FP32 capture compatibility
mode also completed successfully (30.14 s total / 21.41 s denoising).

The retained build has `DAWN_USE_AGILITY_SDK=OFF` and `DAWN_USE_BUILT_DXC=ON`.
Dawn's D3D12 subgroup-matrix feature/configuration enumeration is guarded by
`DAWN_USE_AGILITY_SDK`; when enabled it additionally requires SM 6.10 and supported
linear-algebra capabilities. This identifies a concrete acceleration investigation;
it does not establish that enabling the build flag alone is sufficient on this
driver. Keep future optimization on D3D12 as requested.

Final checks: 53 backend tests, generation/edit repeats, image inspection,
profiled GUI workflow, trace downloads, FP32 capture, lint and Windows packaging.
Runtime selections, model entries and token-limit preferences were preserved.
`test-results/image-progress-profile-ui-report.json` now describes D3D12;
the previous Vulkan report is preserved as
`test-results/image-progress-profile-ui-vulkan-final.json`.

## Earlier Vulkan investigation

GPU timestamps identify quantized matrix multiplication as the denoising hotspot.
The NVIDIA Vulkan activation-copy correction below is retained. Startup now uses
a persistent Dawn cache and shared read-only weight mappings; long-repeat GPU
latency spikes remain unresolved. Earlier experiments below are historical.

## Current startup optimization: Dawn cache and shared mappings

### 2026-10-19: faster text/reference preparation and in-app profiling

The text encoder now retains its immutable CPU parameters/buffers while using a
temporary GPU copy. `evo/cpu_weights.py` restores the original CPU storage after
inference, preserving Parameter identity/ties and nonpersistent buffers. This
avoids downloading the same weights from GPU every image; CPU-to-GPU upload is
still needed. It does not keep the large encoder resident alongside the denoiser.

Matched 512×512 fixed-prompt/seed unprofiled application observations:

| Warm run | Total | Text/reference preparation | Encoder cleanup |
| --- | ---: | ---: | ---: |
| Before, 1 | 15.41 s | 6.93 s | 2.91 s |
| Before, 2 | 13.82 s | 5.87 s | 2.95 s |
| Retained CPU weights, 1 | 10.07 s | 2.48 s | 0.39 s |
| Retained CPU weights, 2 | 9.20 s | 2.09 s | 0.41 s |

Warm total median fell from 14.61 to 9.64 seconds in these observations. Baseline
run 1 overlapped lightweight tests/build; OS/driver caches were not reset. The
isolated cleanup reduction is consistent with removing the weight download.
First new-worker generation was 34.03 seconds; initial imports/encoder loading
remain costly. Reference editing took 18.32 seconds, including 8.98 seconds of
text/reference preparation. All new generation PNGs and the reference edit match
their earlier validated outputs byte-for-byte.
Evidence: `test-results/text-preparation-optimization.json` and
`test-results/prep-{baseline,retained}-*.json`.

Image Studio now exposes **Profile generation**. The report separates GPU shader
timestamps from CPU dispatch durations, lists the slowest shaders and their share
of recorded GPU time, and downloads prefix/denoiser traces plus a JSON summary.
The existing total-time breakdown continues to cover text/reference encoding,
loading, transfers, execution, cleanup and saving. Profiling can add overhead and
does not constitute a warmed benchmark. Live progress includes four stages,
loading messages, actual denoising counts, elapsed time and cancellation; unknown
stage duration uses an indeterminate bar. It remains visible after page navigation
and does not claim overall completion before the image is saved.

The initial profiled UI test (image `1eafcb4adfb74cfcbc9c4dd3f1df36ff`) captured 19,727 GPU events with 1,749.5 ms summed GPU shader
time. Denoiser SubgroupMatrixMatMulNBits accounted for 1,013.6 ms (57.9%);
FlashAttention 347.2 ms (19.8%); prefix MatMulNBitsWideTile 105.3 ms (6.0%).
The PNG matched the baseline. Desktop/compact layout, navigation, trace downloads,
gallery reload, cancellation and validation-error recovery were verified.

**Parallel compilation audit:** selected Image Studio provider hash remains
`1433463cb3fa473b10ab1c9fdabf2c5c27879e85ee0c748c47e77f193f598074`.
At its recorded source revision, `ProgramManager::Build` calls
`CreateComputePipelineAsync`; `WebGpuContext::Run` queues deferred dispatches and
waits when draining their bounded window (default 16 dispatches). The bundled
DawnPlatform uses `max(2, hardware_concurrency / 2)` workers: 16 on this 32-logical-
processor machine. Vulkan forwards pipeline creation to Dawn's async task manager.
Parallel pipeline compilation is enabled; there is no additional disabled switch
to turn on. WGSL generation and `CreateShaderModule` parsing remain synchronous.
This verifies configuration/source paths, not a measured peak of 16 concurrent
compilations; dependencies, cache hits and drain points limit actual overlap.

Chat and embeddings also expose profiling controls. Their selected runtime
`ort-webgpu-main-86d60730c1` currently lacks the GPU timestamp bridge: real Qwen3.5
chat and multimodal-embedding search produced usable CPU operator traces, with an
explicit missing-GPU-timestamp notice. No runtime was silently substituted. Raw
traces include session setup and operator details; CPU operator time includes GPU
dispatch/waits and is not labeled GPU execution time. Native GenAI uses its
session `enable_profiling` option and flushes traces on model release. Embedding
sessions use ORT profiling, including all model/vision/audio graphs. Profile files
are retained in `gitignore/profiles/` and linked from results.

Validation: 53 backend tests, production build/packaging, real image generation
and editing, real profiled image/chat/embedding UI runs, exact PNG checks, runtime
choice preservation and trace-download checks. Reports:
`test-results/image-progress-profile-ui-report.json`,
`test-results/all-profiling-ui-report.json`,
`test-results/image-progress-stop-report.json`, `test-results/all-profiling-backend.log`.
Final packaged verification after the encoder change also passed, capturing
19,727 GPU events with exact output (image `8e975397fca84a4cb490021afe817886`).
That fresh-worker profiled job took 42.92 seconds; it is not a warm performance
sample. The earlier UI report is preserved as
`test-results/image-progress-profile-ui-before-retention.json`.

### 2026-10-18: Dawn cache and shared mmap measurements

Kite now selects `local/artifacts/runtimes/ort-webgpu-qwen-image-dawn-cache/`.
The provider SHA-256 is
`1433463cb3fa473b10ab1c9fdabf2c5c27879e85ee0c748c47e77f193f598074`.
It was built in the Qwen worktree from ORT
`c64c9fd76404df85bbbac37485d5c0f2babcd194` plus its existing GGUF, profiling,
copy-correction changes and the new cache. The earlier copy-fix runtime remains
available. Chat/embedding runtime settings and the three chosen models are unchanged.

The design follows Edge's `D:/r/edge/src/gpu/command_buffer/service/`
`dawn_caching_interface.{h,cc}` and `dawn_platform.cc`, inspected at
`fce00a78dba5a91a6b9e5375410d1f2334865363`: supply Dawn's `CachingInterface`
through its platform, with a memory cache backed by persistent storage.
ORT's bundled platform previously supplied no caching interface.

- Set `ORT_WEBGPU_DAWN_CACHE_DIR` before the first native Dawn instance.
  Kite sets it to `gitignore/dawn-cache`; entries live in its `v1` subdirectory.
- Memory budget: 64 MiB. Disk budget: 256 MiB, best-effort across concurrent
  processes. Entries use exclusive temporary files and atomic publication.
- Validate full keys, sizes and checksums before giving opaque blobs to Dawn.
  Corrupt or unavailable files cause recompilation. Checksums detect corruption;
  they are not cryptographic authentication. Dawn supplies version-dependent
  keys; Vulkan pipeline keys include the driver's pipeline-cache UUID.
- The full generation workload currently creates 138 blobs, about 1.6 MB.
  A separate prefix probe observed 108 stores/1 hit on an empty cache and
  0 stores/109 hits in a fresh process using the saved cache.

ORT already maps external ONNX weights. The additional optimization maps each
whole weight file once and shares those read-only mappings across prefix and
denoiser sessions and successive images. `evo/initializer_cache.py` uses
`add_external_initializers_from_files_in_memory` with
`session.use_external_initializer_file_buffers_directly=1`. ORT borrows those
buffers: live sessions retain their mapping owner, even after cache invalidation.
Kite terminates the image worker before deleting its model files on Windows.
Hosts lacking this API use ordinary loading.

This maps 4,850,557,440 bytes of files; that is virtual address space, not a claim
of equal physical RAM consumption or zero-copy access by the discrete GPU.
CPU-to-GPU uploads and session creation still occur. A retained prefix fixture
measured ordinary cached session loads of 2.750/2.813 seconds versus shared-mmap
loads of 2.286 seconds first and 1.209/1.234 seconds subsequently. Execution stayed
around 1.09–1.13 seconds, with identical output hashes. Mapping setup was about
6 ms. These are isolated prefix measurements, not complete-image times.

Packaged 512×512 generation, fixed prompt/seed and eight uncaptured steps:

| Run | Total | Denoising execution |
| --- | ---: | ---: |
| Earlier copy-fix warm observation | 17.89 s | 1.76 s |
| New first run, empty Dawn cache | 48.86 s | 1.74 s |
| New repeat 1 | 14.62 s | 1.69 s |
| New repeat 2 | 13.41 s | 1.68 s |
| New process after reopening, saved Dawn cache | 31.21 s | 1.69 s |
| New repeat 3, after that process's first run | 12.04 s | 1.68 s |

The three warm observations have a 13.41-second median, about 25% below the
earlier single warm observation. This comparison includes mmap and Dawn caching;
it does not isolate the contribution of either change. Reopening preserves disk
blobs but resets worker components/mappings. The 31.21-second restart also
benefits from existing OS/driver caches and is not a measurement of Dawn alone.

The new generation PNGs exactly match the earlier copy-fix PNG. The complete
512×512 reference-image edit took 22.25 seconds and exactly matches the PNG
decoded from the validated copy-fix denoising trajectory. These are individual
application observations, not a controlled claim that cold startup improved:
the earlier detailed cold run was 39.04 seconds. The new first run spent 10.44
seconds initializing libraries and 16.24 seconds loading/transferring the text
encoder. No unrelated compiler process was observed during these app runs;
filesystem and driver caches were not reset.

Repeat 2 breakdown: text preparation 5.86 s, prefix phase 2.34 s, denoising
phase 4.30 s, decoding/saving 0.91 s. Model setup/transfers across phases still
consume 6.92 s. Text-encoder return to CPU/cleanup alone takes 2.65 s. Dawn cache
reduces repeated compilation; it cannot eliminate these transfers or Python
imports. This identifies the next startup work separately from GPU MatMul stalls.

Rejected load experiments: preserving the device, mapping initializer GPU buffers
at creation, Queue.WriteBuffer uploads, and Windows PrefetchVirtualMemory. None
improved load plus first execution consistently; speculative native upload changes
were removed. ONNX graph parsing itself was only 55–66 ms in the sampled probe.

Validation: 49 backend tests; 72 native numerical cases across Vulkan/D3D12;
profiling lifecycle on both backends; the 4,096-repeat Vulkan regression; real
GPU cache persistence, shader-identity changes, corrupt entries, concurrent writers
and unavailable-directory tests on both backends. The native suites ran 15 test
methods with no skips. Standalone cache storage tests also passed. Windows
packaging and complete generation/editing passed. Source snapshots, including
the new header, hashes and validation provenance accompany the selected runtime.

Evidence: `test-results/qwen-cache-app-*.json`,
`test-results/cache-edit-golden/comparison.json`,
`test-results/dawn-cache-backend-tests.log`, and Qwen worktree
`build/cache-final-native-validation.json`, `build/cache-release-build.log`,
`build/load-shared-mmap/`, `build/load-cache-cpu/`, `build/load-cache-{cold,warm}/`.
All model assets, caches, runtime binaries, reports and generated images remain
in ignored folders.

## Follow-up: repeated activation-copy corruption

The dedicated `ws-ort/ort-prs/qwen-image` worktree narrowed the poster failure to
the third block's MLP. Its normalized input stayed identical while both gate and
up projections lost four rows out of every eight. A single real gate projection
with resident inputs reproduced this on call 55 (M3072, K4096, N12288).

The affected `SubgroupMatrixMatMulLayout` shader used a cooperative-matrix load
and store to rearrange activations. Ordinary element copies avoid the corruption
on this RTX 5080 / driver 616.64 / Vulkan configuration. The final workaround is
scoped to NVIDIA Vulkan, has its own shader-cache identity, and preserves the
existing copy path on other adapters. The underlying Dawn/driver cause is not
yet established.

A synthetic column-select MatMulNBits provides exact expected outputs without
model weights: the original runtime failed 465 of 4,096 repetitions; the final
copy fix passed all 4,096. The real projection passed 512 consecutive checks;
64 values independently checked against CPU dequantization/matmul have maximum
absolute error 0.00274 and cosine 0.9999975. The corrected copy candidate produced
identical complete denoising trajectories across six poster, five editing, and
four sky runs. The earlier minimizer inspected only every 128th iteration;
the revised readback test checks every iteration.

**This is a correctness fix, not a demonstrated speedup.** Long stalls remain.
The scalar-copy screen included a 22.23-second poster run and an 11.55-second
editing run. Smaller 64×64 tiles and direct reads from the original activation
buffer also retained stalls and were rejected. Some diagnostic runs overlapped
compilation and are not controlled speed comparisons.

Final runtime: `local/artifacts/runtimes/ort-webgpu-qwen-image-copy-fix/`, built
from `c64c9fd76404df85bbbac37485d5c0f2babcd194` plus the preserved GGUF/profiling
changes and the scoped copy fix. Provider SHA-256:
`a160b63f0dfef7f3f4701ddcd76161ef3c029f31fc0f73e742bb3cd512ce4635`.
Final native validation passes 72 numeric cases across Vulkan/D3D12, profiling
lifecycle tests on both backends, and the 4,096-repeat Vulkan regression.
The WGSL generator's 89 tests also pass. Exact commands, candidate patches,
logs and final model checks are retained in the worktree's `build/` directory.

Packaged Image Studio generated identical 512×512 PNGs in a cold and warm run:
42.07s total / 1.80s denoising, then 17.89s total / 1.76s denoising. These are
individual application observations, not evidence that the copy fix improves
latency. The images and reports are retained under `gitignore/images/` and
`test-results/qwen-copy-fix-app*.json`.

## End-to-end timing breakdown

Image Studio now shows phase percentages and indented, non-overlapping details
for model loading/GPU setup, library initialization, preparation, execution,
saving and cleanup. CUDA intervals wait for completion before ending their
timer. Prefix and denoiser execution include required CPU readback; these are
wall-clock measurements, not shader-only GPU timestamps. Unknown costs remain
in phase overhead; older runs do not acquire invented subphase measurements.

The latest instrumented 512×512 cold run took 39.04s: preparation 19.09s,
prefix 8.28s, denoising phase 9.31s, and decoding/saving 2.35s. Across those phases,
model loading/setup accounted for 22.77s (58.3%) and library initialization 5.19s
(13.3%). Denoising execution itself was 1.88s. This identifies startup/model setup
as the dominant cost in this run, independently of the long-repeat GPU stalls.
Evidence: `test-results/image-timing-ui-report.json`; validated at desktop and
1000px width, with totals checked and legacy-record compatibility. Backend 47
tests pass, including CUDA timer-boundary accounting.

## What was measured

RTX 5080 16 GB, NVIDIA driver 616.64, Windows, native ORT 1.31 host and Vulkan
WebGPU provider built from ORT `c64c9fd76404df85bbbac37485d5c0f2babcd194`, with
the existing GGUF floating-zero-point/BK64 patches. The model is Kite's
`qwen-image-2.1-turbo-gguf-floatzp`, `short-blocks` graph variant.

Saved Image Studio fixtures and prefix caches were replayed for eight denoising
steps, with capture disabled. This isolates the denoiser; it excludes text
encoding, prefix inference, VAE decoding and image saving. Session creation is
reported separately. These times must not be compared directly with the earlier
44-second cold / 16.5-second warm complete-image observations.

The profiling bridge was missing: the plugin's adapter constructed an internal
profiler, but `OrtEp::CreateProfiler` was not implemented. Ordinary ORT traces
therefore contained CPU operator events without GPU shader timings. The isolated
profiling build connects the callback to the existing WebGPU timestamp collector.
Its traces include shader names, specialization keys, shapes and GPU durations.

| GPU work | Healthy 768×768 sky pass | Slow 512×512 edit pass |
| --- | ---: | ---: |
| Quantized tensor-core matrix multiplication | 1.807 s (47.5%) | 11.906 s (93.0%) |
| Flash attention | 1.343 s (35.3%) | 0.639 s (5.0%) |
| All GPU shader events | 3.801 s | 12.801 s |
| Denoising wall time with profiling | 4.027 s | 12.993 s |

The sky table excludes warmup. That warmup overlapped the tail of another test
and is not performance evidence. All operators with recorded provider assignment
in these denoiser traces ran on WebGPU. CPU node durations are dispatch timings,
not GPU compute time; they must not be added to the GPU totals.

## Repeated-run checks and experiments

An unprofiled control using the app's eight Torch threads measured
12.391 / 4.457 / 13.975 seconds after a 2.049-second first edit. All four edit
trajectories were identical in that control. A poster control showed substantial
drift (trajectory cosine 0.884 / 0.928 relative to its first execution), so the
issue is not exclusively profiling or graph capture. This extends the limitations
reported in the earlier, shorter validation runs.

| Candidate | Observation | Decision |
| --- | --- | --- |
| Submission batches of 1 or 64 dispatches | Stalls remained | Not selected |
| Serialized compatibility mode | Sky repeated at ~4.0 s; editing/poster still failed the broader check | Not selected |
| Robust buffer access | Slower ordinary passes; poster still drifted | Not selected |
| Disable storage/uniform buffer caches | Stalls remained | Not selected |
| Device-local staging instead of mapped initializer upload | Stalls and drift remained; no evidence of major shared-memory spill in the sampled Windows counter | Not selected |
| Tensor-core FP32 accumulator prototype | Edit repeated exactly but latency spikes remained | Not selected |
| Decode each matrix once, then ordinary MatMul | 14.82 s edit median; slower | Not selected |
| Materialized attention MatMul/Softmax graph | Slower and did not resolve the main hotspot | Not selected |
| Smaller 64×64 tensor-core tiles | 2.50 s edit median, but large repeat drift (cosine ~0.947) | Not qualified |
| Small tiles with explicit accumulator initialization | Edit repeated exactly at 2.52 s median; poster still drifted (cosine as low as 0.884) | Not qualified |
| Existing FP32 accumulation fallback | Edit 8.483 / 8.466 / 8.399 s; identical trajectories | Consistent compatibility option, slower than a healthy FP16 pass |

Neither GPU-memory placement nor accumulation precision alone explains the
problem. The numerical root cause remains unresolved. A fast first execution,
finite tensors, and small operator tests are insufficient to qualify a fix.
No performance superiority over GGUF is established by these experiments.
Kite's selected BK64 DLL and model weights are unchanged. The speculative native
kernel/memory changes were removed from the working sources; isolated binaries
and diagnostic sources remain available as evidence. Only the profiling bridge
and its regression test are retained as new native changes.

## Reproduce the profile

From the Kite folder, using the retained sky fixture:

```powershell
.\local\.venv\Scripts\python.exe scripts\profile-image.py `
  --fixtures test-results\merged-workspace\images\e6ff01e85b444f68a33af48aed5cb39b `
  --output gitignore\image-profile\my-sky-profile --profile --runs 1
```

Omit `--profile` for unprofiled timing runs. Both modes now default to the retained
Dawn-cache runtime and D3D12. `--library` explicitly selects another DLL.
Historical fixture replay reuses saved prefix tensors; full Image Studio jobs
also regenerate prefixes on D3D12. `--provider-option KEY=VALUE` permits
controlled provider-option experiments. The default thread count matches the
app. Use a new output directory for each run and run GPU experiments sequentially.

`results.json` records provider options, DLL and graph identity, loading time,
per-step times, all-repeat equality and trajectory differences. `ort_*.json` is a
Chrome/Perfetto-compatible trace; `trajectory*.npy` preserves intermediate states.
The profiler groups GPU shader costs separately from CPU events and excludes the
requested warmup executions from GPU shader totals. Profiling changes overhead;
use separate unprofiled runs for performance qualification.

All raw reports are in `gitignore/image-profile/`, excluded from Git. Runtime
experiments are isolated under `local/artifacts/runtimes/`, also excluded.
The public profiling regression verifies GPU events, numerical output, session
reuse and running after profiling ends, on both Vulkan and D3D12.

Final validation: 116 Kite/engine tests passed, two optional suites skipped;
profiling lifecycle tests passed on both backends; 72 native numerical cases
passed with profiling enabled. A full eight-step edit produced 17,816 GPU shader
events using the finalized diagnostic DLL. Its SHA-256 is
`12f6e24cfddea0dff358c1d064edbd39d013f8ea1a0f7770221e2f5bc76fa43b`.
The selected production DLL remains
`6ce1ad3fe6f00f63a608ae7ceca117a9a02213ec406c598e257bb76dd311aff8`.
See `gitignore/image-profile/validation.json` for the verification record.
