# Standalone Kite and Qwen-Image performance

Kite no longer depends on `E:/workspace/project/evo`. The old path was made
unavailable during native and packaged-app verification.

## Asset and data migration

73,653 files (201,245,372,559 bytes) were relocated on the same NTFS volume into
`E:/workspace/project/kite/local`. Every relocated file retained its file identity,
volume and size immediately after the move. This avoided duplicating 201 GB on a
drive with approximately 24 GB free. The original models, exports, runtime DLLs,
benchmarks, fixtures, dependencies and authoring resources were retained.

The relocated Python environment uses its existing CPython 3.11 installation
under the user's UV directory, which is outside the retired Evo project. Its
editable-package paths, activation scripts and 35 console launchers were rebuilt
for Kite. No ORT wheel or source-built runtime DLL was replaced by a stock package.
Kite is independent of Evo; it is not a portable distribution of Windows/Python.

Electron and the backend discover `local/kite-assets.json`; `KITE_ASSET_ROOT`
can override that asset library. The merged engine is `kite/evo/`, with authoring
resources in `evo_resources/`. Legacy paths in benchmark metadata are resolved
in memory to the new asset root, preserving hashed evidence bytes.

The user's four model references, four conversations and one generated image
were preserved. Model IDs and conversation text did not change. The migrated PNG
is byte-identical to its original. Each application database has a
`before-asset-relocation.db` backup. Existing settings now use `asset_root`, and
point to Kite's local Python and DLLs.

**Generated images save automatically to `gitignore/images/<run id>/image.png`.**
The same folder retains request/settings and result/timing JSON. The gallery
continues to use the application database; its records point to these new files.
Both `/gitignore/` and `/local/` are excluded from Git. Curated demo previews remain
application assets under `demo/`.

Keep the whole Kite folder, including `local/`, `gitignore/`, and the packaged
application's `resources/`. The original sibling Evo directory is no longer
needed. Retained historical metadata mentioning Evo is provenance, not a live
dependency. Other unrelated repositories are outside this migration's scope.

## Selected image optimizations

- Native WebGPU Vulkan **FP16 without capture** is the default interactive path.
  The eight-step schedule, resolution and source model remain unchanged.
- The official Qwen3-VL backbone replaces its conditional-generation wrapper
  for feature extraction. The unused 1.245 GB vocabulary projection, intermediate
  hidden-state collection and autoregressive KV cache are omitted. The existing
  Diffusers pre-normalization hook is preserved.
- Text-encoder and VAE weights remain cached on the CPU between images. GPU
  sessions are released between phases. Stop, Unload model, or switching to chat,
  embeddings or another tool releases the image worker and its CPU cache.
- After initial SHA-256 verification, unchanged artifacts reuse the verification
  result. File identity, size, modification time and expected hash are included
  in the fingerprint; changes invalidate it.

The CPU cache holds a large VLM and consumes substantial RAM. First-image loading
is still significant. Text/reference encoding and VAE decoding run PyTorch/CUDA;
the denoiser and prefix run the source-built ORT WebGPU provider.

## Measurements on this RTX 5080

Follow-up [GPU profiling and longer repeat checks](qwen-webgpu-profile.md)
exposed intermittent FP16 denoiser stalls and output drift beyond the shorter
checks below. These historical successful runs do not establish general repeat
stability. The follow-up records the limitation and rejected optimizations.

| Workload | Earlier behavior | Selected path |
| --- | ---: | ---: |
| 768×768 cloud-city denoising | 20.23 s, FP32 capture including first-run work | 3.88–3.98 s, FP16 uncaptured, three identical repeats |
| 512×512 packaged generation, cold | 54.16 s in the preceding packaged check | 44.35 s |
| Same 512×512 generation, warm | Previous app reloaded every phase | 16.53 s |
| 768×768 cloud city, CPU components reused | Earlier cold complete run 56.59 s | 22.48 s before verification-cache optimization |
| 512×512 reference-image edit, cold packaged app | Earlier cold run 46.63 s | 49.95 s |

These are interactive wall-clock observations with explicitly different cold/warm
conditions, not a controlled claim of improvement for every workload. Cold image
editing did not improve in this check. The fastest benefit is denoising and
subsequent generations after the large text encoder is loaded.

The final cold and warm 512×512 outputs are byte-identical. Full-size cloud-city
prompt features, initial latents, rotary data, prefix features and image masks
match the original pipeline exactly. Independent tiny-model tests also show
exact pre-norm text and image features with the vocabulary head omitted.

## Runtime/model candidates deliberately rejected

The existing source-built BK64 provider remains selected. Two native capture
prototypes were built in `ws-ort/ort-prs/evo-qwen-image-gguf`: cooperative staging
barriers, and recording initialization/upload operations for capture replay.
Neither passed repeated full-image validation across the tested prompts.
The latter passed 72 small numerical cases across Vulkan/D3D12, including 240
captured executions, but still showed finite output drift in full images.
Their C++ edits were restored; binaries, patches and logs remain isolated and
are not selected by Kite. The extra operator capture tests remain in that worktree.

Projection-merging and D3D12 alternatives were also screened; they did not provide
a qualified improvement. FP16 capture remains an unresolved runtime interaction.
Kite's optional capture compatibility mode retains FP32 accumulation. Do not infer
image correctness from finite output or small operator tests alone.

## Verification

- 41 Kite backend tests; 75 migrated engine tests, with two optional suites skipped.
- Native chat, English/Chinese embeddings, batching, search, cancellation, media
  fixture agreement, and all six multimodal examples with Evo unavailable.
- Packaged UI: all 21 tools, real ONNX inspection, recipe planning, 66 historical
  performance observations, wishlist persistence, image previews and comparison.
- Packaged image generation/editing, cancellation/retry, automatic ignored-folder
  saving, exact repeat output, component/verification reuse, gallery persistence,
  and switching back to the embedding runtime.
- Working relocated Evo/Mobius entry points and installed authoring capabilities;
  a real tiny Qwen3.5 export completed successfully without downloading weights.
  The underlying GenAI package's direct `--help` has an existing argparse percent
  formatting error; Kite's own `evo build --help` and recipe planning work.
- Runtime build experiments, production build, packaging, lint and whitespace checks.

Evidence lives in `test-results/asset-relocation-report.json`,
`user-data-migration-report.json`, `pruned-features-report.json`,
`packaged-image-inference-report.json`, `standalone-edit-report.json`,
`native-report.json`, `multimodal-report.json`, `merged-ui-desktop-report.json`,
and the matching logs. Optimization samples and latent files are under
`gitignore/image-optimization/`. The migration marker records completion only
after these checks pass.
