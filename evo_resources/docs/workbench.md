# Evo performance workbench

Start the UI with `.venv\Scripts\evo ui --port 8873` and open http://127.0.0.1:8873.
The current running instance uses port 8873. The CLI default remains 8872.

There are two views:

- **Performance** includes the model list and a simple table of measurements:
  model, prompt, machine, latency, and throughput. Import a local ONNX or GGUF model,
  record or edit a measurement, and filter by model, machine, or prompt.
  Generation records show first-token latency and generated tokens/second;
  embedding records show full embedding latency and embeddings/second. Image
  records show latency and images/second; notes state whether shared text/VAE
  work is included and record resolution, seed, steps, runtime and capture evidence.
  Existing benchmark reports appear automatically. Expand a prompt to see its
  full text and measurement notes. Results from different prompts and machines
  remain separate observations.
- **Wishlist** lists models to convert, with optional notes and To convert,
  Converting, and Converted status. EmbeddingGemma 2 text embeddings are converted.

Compare, Model library, Operators, and Runs are no longer separate UI views.
Conversion, optimization, validation, and benchmarking remain CLI operations.

Models, manually entered performance data, and conversion requests persist in
`artifacts/workbench/models.json`, `performance.json`, and `wishlist.json`.
Benchmark reports stay under `artifacts/benchmarks` and `artifacts/comparison-*`.
The Qwen3.5-4B GDN + MLP candidate includes paired 128- and 512-output-token
measurements. Notes identify the runtime binary and interleaved protocol.
Rejected quality experiments, unverified provider settings, and contaminated
timings are retained on disk but excluded from Performance.
Removing a model from the list retains model files and saved manual measurements.
Older internal optimization ideas are retained in storage but are not shown in
this model-conversion wishlist.

The service is local-only, uses no CDN, and rejects cross-origin mutations.
Desktop and mobile checks cover navigation, filters, saving/editing measurements,
conversion requests, and persistence after reload. The UI does not schedule or
execute benchmark jobs.
