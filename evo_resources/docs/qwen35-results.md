# Qwen3.5-4B native WebGPU results

**Scope update:** this report compares Evo with its own initial export. A later
[comparison with the previously tuned WebGFX model](previous-vs-evo.md) found
that prior package faster. The Evo package below is smaller, not a performance
upgrade over that external baseline.

Measured on 2026-10-11: RTX 5080, Windows driver 32.0.16.1664, ORT 1.31.0, GenAI 0.17.1, WebGPU plugin 0.4.0.

Selected package: **`artifacts/qwen35-4b-optimized/`**. Keep the graph, external weights, tokenizer and GenAI config together. Source revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## Final measurements

Batch 1, three repeated-passage prompts, 32 forced generated tokens, one warmup and five measured repetitions per prompt. Models ran sequentially on the same GPU. Medians follow. First-token time includes native API work through synchronized token observation; it is not a pure GPU prefill timer. Decode excludes the first observed token step.

| Prompt tokens | Baseline TTFT | Optimized TTFT | Reduction | Baseline decode | Optimized decode | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 135 | 87.6 ms | 70.4 ms | 19.6% | 74.2 tok/s | 74.5 tok/s | 0.4% |
| 863 | 364.5 ms | 278.8 ms | 23.5% | 68.4 tok/s | 74.3 tok/s | 8.6% |
| 3383 | 1459.5 ms | 1123.1 ms | 23.1% | 55.1 tok/s | 73.3 tok/s | 33.0% |

The short-prompt decode difference is within normal run variation. Model/weight files total 2,578,995,089 bytes for the baseline (including MTP), versus 2,441,906,186 bytes for the selected package.

## Selected configuration

- INT4 block 32, FP16 activation request, accuracy level 4.
- MTP disabled; final-position LM-head pruning and MLP gate/up fusion enabled.
- Graph capture disabled: the tested capture candidate was slower.
- Full Mobius WebGPU passes applied. No additional fusion patterns matched the already-fused GenAI graph; no separate Mobius speedup is claimed.
- Default runtime limit 4096 tokens, avoiding allocation for the full source context of 262144. Callers can increase this with an appropriate memory budget.

This comparison measures the combined export changes, not a separate causal effect for each switch.

## Quality and alternatives

Compared first-token logits on three short prompts against the original FP32 Transformers model. All candidates matched the reference first token and had finite logits. The selected model had cosine similarity 0.9750 / 0.9640 / 0.9747 and reference-to-export KL 0.00042 / 0.00019 / 0.04985, comparable to the original INT4 baseline.

Block 128 was slightly faster but had materially worse distribution agreement: cosine 0.9247 / 0.9121 / 0.9636, KL 0.00049 / 0.10281 / 0.52462. It was rejected. Capture measured approximately 69–72 tok/s in exploration and was also rejected.

Mobius post-processing preserved the checked logits exactly. The final graph and weights are byte-identical to the checked Mobius candidate:

```text
model.onnx      31c3ac5c2e6a1571c91a778a51c7a01211cb8dd57699467efda0845df45b0444
model.onnx.data cdf15219c43fe74bcc60b476601711c8ec239fe776aa4b993eff793d11c4ecab
```

Generated text remained coherent, but the short benchmark prompt changed wording. This is a limited first-token regression screen and generation smoke test, not perplexity, task-quality or long-context accuracy qualification. The preset is measured for this device/workload, not universally optimal.

## Existing ONNX input and runtime suggestions

The standalone ONNX route was exercised on the real graph. A native profile confirmed WebGPU compute plus a small CPU partition containing Cast, ReduceSum, Sub, Shape and Gather. `evo analyze` excludes the first run when multiple model-run events exist, and does not treat ORT event durations as GPU shader timings.

The report distinguishes existing GatedDeltaNet/export integration work from proposals for fused convolution/recurrent preparation and a quantized LM-head/greedy-selection epilogue. See [coverage and source audit](optimization-coverage.md).

## Reproduce

```powershell
.venv\Scripts\evo convert recipes/qwen35-4b-recommended.toml
.venv\Scripts\evo optimize artifacts/qwen35-4b-recommended-export --output artifacts/qwen35-4b-optimized
.venv\Scripts\evo benchmark artifacts/qwen35-4b-optimized --cases benchmarks/qwen35-text.json --output artifacts/my-benchmark.json --repeats 5
```

Use fresh paths when rerunning conversion/optimization. The pinned checkpoint is already under `models/qwen35-4b/`. `scripts/prepare_benchmark.py` regenerates workloads; `scripts/check_qwen_quality.py` reproduces the FP32 comparison.

Evidence: `artifacts/experiment-summary.json`, `artifacts/benchmarks/baseline-final.json`, `artifacts/benchmarks/optimized-final.json`, `artifacts/quality-prefill.json`, `artifacts/qwen35-4b-optimized/optimization.json`, `artifacts/webgpu-analysis.json`.
