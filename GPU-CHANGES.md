# Luna: use the GPU when one is available

Based on branch `halcyon-learning` @ 190983e. `GPU-CHANGES.patch` is the same
change as a diff: `git apply GPU-CHANGES.patch` in your checkout.

## Why
On Arch the plain `ollama` and `ggml` packages are CPU-only. On a laptop with
an RTX 3050, `ollama ps` showed every model at "100% CPU". Installing
`ollama-cuda` + `ggml-cuda` moved them to "100% GPU". Luna's code was fine;
it just never told the user.

## What changed
- `src/gpu.rs` (new): if an NVIDIA GPU is present but Ollama / ggml has no
  GPU backend (or `/api/ps` shows loaded models with `size_vram == 0`),
  Luna logs a `GPU:` warning naming the package to install. It stays quiet
  with no NVIDIA GPU or an install layout it doesn't recognise. 6 unit tests.
- `src/main.rs`: runs that check once in the background at startup (not in
  `--daemon`). It goes through tracing, so it can't corrupt the TUI.
- `src/first_run.rs`: `luna --setup` lists the same hints.
- `src/stt/whisper.rs`: if a whisper-cli run fails (e.g. CUDA OOM while
  Ollama fills a 6 GB card), it retries once with `--no-gpu`.
- `src/tts/piper.rs`: Kokoro prefers onnxruntime's CUDA/ROCm provider when
  installed and falls back to the CPU (and to the old constructor on older
  kokoro-onnx).
- `README.md`: new "GPU" section with the install commands.

## Checked
`cargo build` (no warnings) and `cargo test`: 523 passed. The one failure,
`the_live_config_gates_every_capability_tool`, fails without this change too.
It reads the tester's own luna.toml. Not tested end to end: whisper on the GPU
(no whisper model on the test machine) and Kokoro on the GPU (no Kokoro env).
