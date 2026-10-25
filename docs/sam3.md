# SAM 3 video tracking

Open **Video Segmentation**, choose **SAM 3 · Video**, and open a video or try
**Jumping together**. Scrub to the starting frame. Include/exclude points and
boxes select up to four objects, each with its own color and temporal memory.
**Track objects** propagates forward from that frame. This integration supports
the SAM 3 interactive video tracker; text/concept prompts and reverse propagation
are not yet exposed.

The sample clip plays on a loop when opened. **Back to selection** returns to the
marked frame and preserves your prompts; **Play clip** previews the motion again.
Tracking results appear above the selection workspace and play automatically on
a loop, with Play/Pause and scrubbing controls. Recent tracks open in the same
player. Playback is muted by default.

Tracking shows the latest masked frame, completed frames, elapsed time, and Stop.
It survives page navigation during a run. Completed runs can be reopened from
Recent tracks. Clips must be MP4/MOV, under 64 MB, up to 4K. Process up to 600
frames per run; frame interval samples every Nth frame and changes playback FPS
accordingly. Output video is silent.

Outputs are automatically saved in **`gitignore/segments/<run-id>/`**:

- `overlay.mp4`: colored masks over the input video.
- `masks.mp4`: colored mask preview (lossy; not an annotation format).
- `masks.zip`: lossless per-frame label PNGs. Background is 0; object IDs are 1–4.
  Where masks overlap, the later selected object's label wins.
- `result.json`: source frame indices/times, object prompts, predicted IoU scores,
  object-presence predictions, pixel counts, runtime identity, and phase timings.

Profile tracking creates four downloadable ORT session traces and an operator or
GPU-shader summary, depending on the selected runtime's timestamp support. The
current main runtime reports CPU operator durations, including WebGPU dispatch
and waits; these are not GPU shader times. Timings separately report setup,
vision encoder, mask decoder, memory encoder, memory attention, decoding and
output. Processing FPS excludes model setup; total request time includes it.

## Install on a new machine

Use the configured native inference Python and existing source-built ORT WebGPU
DLL. No alternate GPU backend is used. Install the small video dependencies:

```powershell
local/.venv/Scripts/python.exe -m pip install Pillow opencv-python-headless imageio-ffmpeg huggingface-hub
local/.venv/Scripts/python.exe scripts/prepare-sam3.py
```

The setup tool downloads only this explicitly selected model. It uses the pinned
[video ONNX graphs](https://huggingface.co/jax-image-tools/sam3-video-onnx/tree/e942689a73d942f83c69b5965881cb3a0f6153ba),
plus the exact learned memory embeddings and pointer-position projection from
[Meta's checkpoint](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7).
The four graphs total approximately 1.88 GB. The checkpoint tensors are fetched
with HTTP byte ranges, without downloading the entire checkpoint. Existing HF
credentials are used when required; Meta's SAM License applies, and is saved
beside the model. Request access at the model page if your account needs it.

The package is stored at `local/artifacts/sam3-video-onnx`. Kite imports it once
when first installed, or use My models → Import. Custom package/runtime selections
are preserved. Deleting files retains the model record; rerun setup to restore
the default package, or import a replacement. No model weights or runtime DLLs
are committed to Git. The shared `webai-community/ai-models` repository does not
currently contain SAM 3; this installer uses the sources above.

## Runtime and qualification

The four graphs execute through native ORT WebGPU **D3D12**. Vision features are
computed once per frame and shared across objects. Each object keeps its initial
conditioning memory, six recent spatial memories, and up to sixteen pointers.
Learned temporal embeddings and projected sine position encodings follow the
Transformers `Sam3TrackerVideoModel` reference orchestration. Graph capture is
disabled. NumPy performs bank assembly; Pillow/OpenCV decode and resize media.
Neither PyTorch nor optical flow executes the tracker.

Local qualification, RTX 5080, 2026-10-24:

- Four real frames: D3D12 masks exactly equal the ORT CPU reference, IoU 1.0 on
  every frame. This checks runtime agreement, not dataset accuracy or independent
  end-to-end PyTorch export equivalence.
- Approximately 3.1 seconds model load; subsequent single-object frames took
  0.61–0.77 seconds in the short profiled run.
- Complete GUI run: 12 frames, two objects, 16.72 seconds including 3.85 seconds
  setup and profiling. Vision encoding 5.76 seconds, memory attention 3.04 seconds,
  mask decoding 1.31 seconds, memory encoding 0.30 seconds. Local observations,
  not a statistically controlled benchmark.
- Profiles establish WebGPU placement of neural operators. Small decoder and
  memory-attention shape operations run on CPU; the vision/memory encoders had
  no CPU operator events in the four-frame trace.
- Real UI: points and boxes, two identities across occlusion, navigation during
  tracking, MP4/PNG/ZIP downloads, four trace downloads, cancellation releasing
  the worker, history reload, and compact layout.
- Playback regression: actual decoded frames advance in both the demo and result
  players; pause/resume, looping, history replay, and preservation of selection
  prompts pass in the browser. The packaged Windows app also verifies advancing
  sample and result playback after real native tracking.

Evidence is ignored under `test-results/sam3-*`. The dedicated ORT worktree is
`E:/workspace/project/ws-ort/ort-prs/sam3`, branch `perf/sam3`. No new ORT source
patch or replacement runtime is required for this integration.
