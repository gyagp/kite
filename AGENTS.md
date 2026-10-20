# Kite development

## Windows inference backend

Always use ORT WebGPU with the D3D12 backend in Kite, including Image Studio,
chat, embeddings, and current profiling/performance scripts. Do not select
Vulkan for a model or switch to it for performance. Preserve historical Vulkan
reports as historical evidence; they do not describe the current app backend.

This is the user's explicit instruction from 2026-10-19.

## Product language

Kite's GUI focuses on inference and model management. Do not expose model
conversion, export, quantization, authoring commands, or conversion wishlists.
Keep those capabilities in developer tooling rather than a Model Lab UI.

Present the application and its model tools as Kite. Do not use Evo branding in
UI copy, prompts, tool previews, or help output. Legacy implementation modules,
artifact names, and provenance remain internal compatibility details.

This is the user's UI preference from 2026-10-18.

## Model library

Only add models to My Models when explicitly requested by the user or selected
through import/download. Do not populate the library from the Hugging Face
catalog or treat repository listings as models supported by Kite. Hugging Face
remains available through Discover models, and its metadata may supply download
links for models already chosen. Deleting a chosen model from disk retains its
library entry unless the user explicitly asks to remove the entry too.

This is the user's correction from 2026-10-18.

## New model performance work

When adding support for a new model, create a dedicated ORT optimization workspace
under `E:/workspace/project/ws-ort/ort-prs/<model-name>`. Use a separate Git
worktree/branch and its own build outputs, following the ORT workspace instructions.
Reuse that model's existing workspace when continuing its optimization work.

Profile the real workload and validate both performance and output correctness
before integrating an optimized runtime into Kite. Keep model assets and generated
outputs in Kite's ignored folders. Preserve unrelated ORT worktrees and explicitly
selected custom runtimes.

This is the user's standing preference from 2026-10-18; creating these model
optimization workspaces does not require another confirmation.

Qwen-Image optimization workspace:
`E:/workspace/project/ws-ort/ort-prs/qwen-image`.
