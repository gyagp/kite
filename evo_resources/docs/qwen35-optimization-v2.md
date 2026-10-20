# Qwen3.5-4B: follow-up optimization

The previous WebGFX package remains the recommended performance baseline.
The new Evo mixed-precision candidate recovers most of the old Evo decode
regression, but the controlled comparison does **not** establish a speedup over
WebGFX. Exploratory measurements that suggested a gain did not survive the
alternating-order comparison.

The new candidate is `artifacts/qwen35-4b-policy-gates-int8/`. It was exported
with current GenAI using the previous model's INT8 LM head and INT8 output/down
projections in linear-attention layers. Then Evo quantized the 48 small dense
recurrent gate matrices to INT8. This last step changes numerical behavior;
GenAI deliberately keeps those gates dense, so it remains an explicit experiment.

On the older 135/863/3383-token, 32-output-token screen, leaving the 48 matrices
dense yielded about 67–68 tok/s. Quantizing them to INT8 recovered approximately
110–121 tok/s. This isolates a major contributor to the regression, rather than
attributing it solely to serialized node counts or MLP fusion names.

## Final controlled measurements

Testing follows `webgfx-china/webgfx-china/tasks/ai-test.py`: prompt lengths
128/512/1024, 128 generated tokens, generator reuse, maximum context 8192, and
explicit D3D12. Evo uses deterministic token IDs in [0,99] so both models see
identical inputs. Python `is_done` calls follow the native benchmark's per-step
synchronization approach; final token readback is included. Results should not
be mixed with the earlier 32-output-token, separate-generator measurements.

Machine: webgfx-100, RTX 5080. ORT 1.31.0, GenAI 0.17.1, WebGPU plugin 0.4.0.
Basic validation and graph capture off for both. Order: previous → Evo → Evo →
previous, each process with one warmup and five measurements per input length.

| Prompt tokens | Previous TTFT | Evo TTFT | Previous decode | Evo decode |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 54.55 ms | 54.27 ms | 124.20 tok/s | 123.27 tok/s |
| 512 | 163.15 ms | 162.72 ms | 124.38 tok/s | 116.20 tok/s |
| 1024 | 314.94 ms | 314.85 ms | 123.99 tok/s | 123.44 tok/s |

Each cell is a median of ten samples. Evo decode is 0.8%, 6.6%, and 0.4% slower.
The tiny TTFT differences are not a meaningful performance win. Generated token
sequences were deterministic within each measurement group. The candidate is
also larger: 4.446 GB of graph/external files versus 3.165 GB for the previous
package, partly because current mixed-head export retains dense embeddings.

Evidence: `artifacts/comparison-qwen35-final/summary.json` and the four raw
`previous-{a,b}.json` / `evo-{a,b}.json` reports.

## Broader numerical screen

Twelve English/Chinese/code/JSON/reasoning prompts were checked at eight
FP32-reference-generated prefix lengths each: **96 logit comparisons**.
Reference generation uses the official FP32 Transformers model. Each exported
model recomputes a full prefix, so this does not validate accumulation error
over long native recurrent decoding.

| Metric | Previous | Evo |
| --- | ---: | ---: |
| Top-1 agreement with FP32 | 85/96 | 86/96 |
| Median reference-to-export KL | 0.003531 | 0.003299 |
| Maximum KL | 14.2404 | 11.5844 |
| All compared logits finite | Yes | Yes |

The large worst-case KL values and top-token disagreements remain material
quality limitations. One extra top-1 match is not a broad quality improvement.
Evidence: `artifacts/comparison-qwen35-final/quality.json`.

## Other experiments and next work

- Mobius post-processing of the previous graph found no useful extra fusion.
- Forcing accuracy level 0 significantly worsened prefill.
- Graph capture, one CPU thread, disabled validation, and an INT4 head were
  screened without establishing a replacement recommendation.
- The previous runtime already folds 32 MLP norms and eight attention norms
  into fused kernels: the profile retains 24 standalone skip normalizations.
  Explicitly writing the same MLP-norm fusion does not add a new optimization.
- A GatedDeltaNet candidate uses explicit FP32 V-major state and FP32 gates.
  It was rejected at session creation: plugin 0.4.0 has no usable GatedDeltaNet
  kernel. The kernel exists in reviewed upstream ORT source, so this is a
  deployed-runtime gap, not a request to invent an absent operator. That candidate
  requires a newer, source-pinned native plugin and fresh numerical/performance
  validation. Its states are not interchangeable with the original FP16 K-major
  recurrent cache.
- Gate INT8 transplanted into the previous package is retained as a further
  isolated experiment; it is not part of the final comparison above.

Model experiments use independent copies of external weights. ONNX 1.23 rejects
multiply-linked external data files during structural checking; temporary
experimental hardlinks were replaced with independent copies, and the original
package passes structural inspection again.

## Reproduce the current Evo candidate

```powershell
.venv\Scripts\evo convert recipes/qwen35-4b-webgfx-policy.toml
.venv\Scripts\evo quantize artifacts/qwen35-4b-webgfx-policy-v3 `
  --output artifacts/new-gate8-candidate `
  --node-regex 'linear_attn/[ab]_proj/MatMul$' --bits 8
```

Use new output paths. `runtime_provider_options` applies the runtime policy after
export, keeping graph-capture preparation independent from enabling replay.
All source packages remain available.
