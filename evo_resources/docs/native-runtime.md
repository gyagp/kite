# Native WebGPU runtime selection

Current status: build completed successfully; all 20 native GDN kernel checks
pass; `.evo/runtime.json` selects this build. Capture replay has been verified
for all three tested Qwen variants. See [FP16-state results](fp16-state-results.md)
for performance and the remaining capture numerical discrepancy.

The user-requested source checkout is
`E:/workspace/project/ws-ort/onnxruntime`, branch `main`. It was fetched and
fast-forwarded to `86d60730c14f546ba1a4ba06b4887b53285117a5` on 2026-10-12.
Existing untracked installation and Node files were preserved. The source
version is 1.32.0; this does not change the Python host ORT or GenAI packages.

The fresh build uses Release, Ninja, MSVC 14.51, LTO, RTTI disabled, and the
native D3D12 WebGPU plugin target. Build directory:
`E:/workspace/project/ws-ort/onnxruntime/build/evo-webgpu-latest/Release`.
The exact command and result are in `build/evo-webgpu-latest-build.log` and
`build/evo-webgpu-latest-build.exit` in that checkout.

This build contains pristine upstream source. It does not incorporate the
earlier experimental GDN subgroup patch. Earlier benchmark results remain tied
to that patch's own binary hash and cannot be transferred to the new runtime.

## Select a runtime

Evo selects a native WebGPU DLL in this order:

1. An explicit `--ep-library` argument on commands that provide it.
2. The `EVO_WEBGPU_LIBRARY` environment variable.
3. `.evo/runtime.json` in the current working directory.
4. The installed `onnxruntime-ep-webgpu` package.

The project setting stores `webgpu_library`; relative paths are resolved from
the `.evo` directory. An invalid selection fails rather than silently falling
back. The library's `evo-runtime.json` sidecar records source/build provenance
and DLL hashes, which Evo verifies before loading. Explicitly selected libraries
remain useful for reproducing historical results.

`scripts/package_webgpu.py` checks the build exit code, expected source revision,
tracked-tree cleanliness, matching CMake source path, and DXC hashes. It creates
an immutable package. Official DXC v1.8.2502 matches this upstream revision's
Windows plugin packaging workflow. The project default is written only after
the packaged library passes independent native GDN checks.

## Graph capture and FP16 state

The upcoming Qwen performance tests explicitly enable graph capture for both
models using `scripts/benchmark_interleaved.py --graph-capture`. The flag is
recorded as requested configuration. A separate native trace must show replay;
setting `enableGraphCapture=1` alone is not replay evidence. Numerical controls
may run uncaptured to check the behavior of the captured path.

At this revision, GatedDeltaNet accepts FP16 Q/K/V but requires FP32 state and
gate parameters (`TS`). FP16 cache storage can be tested using explicit casts
around the FP32 operator. This saves cache storage but adds dispatches and
scratch memory; it does not implement a native FP16-state kernel.

The two experimental model packages are:

- `artifacts/qwen35-4b-gdn-mlp-fp16-cache`: same GDN + MLP graph with 48 state
  conversion nodes. Internal recurrent computation remains FP32; it rounds at
  each invocation boundary, not each token within a multi-token prefill.
- `artifacts/qwen35-4b-linear-fp16-mlp`: previous native FP16 LinearAttention
  graph with the 32 MLP fusions. This also changes the attention implementation,
  so it cannot isolate the effect of state precision alone.

Validation progress and evidence are recorded in
`artifacts/comparison-latest-capture`. Experimental exports are not automatically
recommended or substituted for the previously validated model.
