# Qwen3.5-4B: first conversion target

The full pretrained model has now been exported, optimized and benchmarked on
native WebGPU. Use [the measured recipe](../../recipes/qwen35-4b-recommended.toml)
and read [results and limitations](../qwen35-results.md). The original recipe
below remains the baseline for comparison.

Source: [Qwen/Qwen3.5-4B configuration at the reviewed revision](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json).
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

This is a multimodal `Qwen3_5ForConditionalGeneration` checkpoint. Its text
backbone has 32 layers: 24 linear-attention layers and 8 full-attention layers.
Hidden size is 2560, intermediate size 9216, and vocabulary size 248320.
The source requests BF16 weights and FP32 recurrent state. Full-attention layers
have 16 attention heads, 4 KV heads and head dimension 256. These are source
configuration facts, not a guarantee about the exported tensor layout.

The chosen first workload is **text generation on native ORT WebGPU**, with a
Mobius-style Python API and CLI. Vision and browser deployment are later work.

## First recipe

`recipes/qwen35-4b-genai-int4.toml` uses the GenAI builder's Qwen3.5 path with
WebGPU, INT4 block 32, FP16 I/O request and graph capture off. In the reviewed
**0.17.1 wheel**, this builder exports the text backbone and marks the runtime
model as `qwen3_5_text` when embeddings are included. It does not supply a full
vision pipeline. This is an export candidate, not a benchmark-qualified preset.

```powershell
.venv\Scripts\evo plan recipes/qwen35-4b-genai-int4.toml
.venv\Scripts\evo convert recipes/qwen35-4b-genai-int4.toml
```

The HF source in the recipe is unpinned at execution time. To use the exact
reviewed checkpoint, first download a local snapshot with the HF CLI:

```powershell
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/qwen35-4b
```

Then copy the recipe, set `source_kind = "local"` and
`source = "../models/qwen35-4b"`, and choose a fresh output directory. Downloading
the full source weights is a multi-gigabyte operation; the initial project setup
does not require that download.

A tiny random Qwen3.5 checkpoint with the same mix of attention mechanisms has
been exported successfully through the real GenAI backend; see
[verification evidence](../verification.md). That test does not qualify the full
4B weights, output quality or GPU execution.

## Critical validation cases

- Bind convolution state and recurrent state for linear-attention layers as well
  as KV caches for full-attention layers. Use exported names, shapes and types;
  a conventional Llama KV-cache loop is insufficient.
- Check kernels used for the recurrent update and causal convolution against the
  selected ORT WebGPU build, alongside attention and MatMulNBits kernels.
- Compare the first prefill, subsequent decode steps, and chunked prefill to the
  reference model. Reset all state between requests. Test short and long prompts.
- Preserve FP32 state where the exporter requires it. Do not apply blanket FP16
  conversion to every tensor in this hybrid graph.
- Preserve the tokenizer and chat template. Validate token IDs before attributing
  generation differences to quantization. Do not silently alter tokenizer regexes.
- Treat graph capture as a later experiment after the hybrid-state path works.
  Keep the initial graph-capture flag off.

## Vision path

The [upstream Olive Qwen3.5 recipe](https://github.com/microsoft/olive-recipes/tree/main/Qwen-Qwen3.5-4B/builtin)
exports vision, embedding and text as separate graphs, then assembles runtime
configuration. This is the next integration to evaluate if images are required.
Its preprocessing and tokenizer adjustments need verification in the selected
runtime; copying them blindly can change tokenization and invalidate comparisons.

For browser deployment, implement the model-specific state/generation loop and
image pipeline explicitly. A native GenAI configuration alone does not implement
those browser-side behaviors.
