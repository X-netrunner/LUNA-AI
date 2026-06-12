# Luna — Local AI Assistant

A fast, personal AI assistant built in Rust, running entirely locally on your machine. No cloud, no subscriptions, no data leaving your system.

## Features

- **Local-first** — runs via Ollama, everything stays on your machine
- **ReAct agent loop** — reasons and executes tools in a chain
- **Dual memory** — rolling conversation context + permanent memory that survives restarts
- **Shell history context** — learns your workflow from fish shell history
- **System indexer** — maps your projects, scripts, and configs
- **Voice I/O** — Whisper STT + Kokoro TTS (high quality, CPU-based)
- **Model escalation** — fast model for simple queries, full model for complex tasks
- **Desktop integration** — opens apps, reads/writes files, sends notifications, controls clipboard
- **Web fetch** — fetches and reads web pages for current information
- **Documentation** — Handled by claude or gemini , so that anyone can understand the flow

## Tools

| Tool | Description |
|------|-------------|
| `run_shell` | Run any bash command |
| `edit_file` | Open file in editor (zeditor) |
| `read_file` | Read file contents |
| `write_file` | Write to file |
| `notify` | Desktop notification |
| `system_info` | Battery, CPU, RAM, temp, disk, uptime |
| `clipboard` | Read/write Wayland clipboard |
| `fetch_page` | Fetch webpage text |
| `find_file` | Find files by name |
| `remember` | Save fact to permanent memory |
| `forget` | Remove from permanent memory |
| `list_memories` | Show all permanent memories |
| `index_system` | Scan and map your system |

## Requirements

- Rust 1.75+
- [Ollama](https://ollama.com) with a model pulled (default: `qwen2.5:7b-instruct-q4_K_M`)
- `whisper-cli` (from `whisper.cpp` AUR package) for voice input
- `piper-tts` for basic voice output
- Kokoro ONNX models for high-quality voice output
- `wl-copy`/`wl-paste` for clipboard (Wayland)
- `curl` for web fetch

## Installation

```bash
# Clone the repo
git clone https://github.com/X-netrunner/LUNA.git
cd LUNA

# Build
cargo build --release

# Run
./target/release/luna
```

## Configuration

On first run, Luna creates `~/.config/luna/luna.toml` with defaults.

Key settings:

```toml
[agent]
system_prompt = "You are Luna..."
max_react_iterations = 8

[llm]
model = "qwen2.5:7b-instruct-q4_K_M"
base_url = "http://localhost:11434"
fast_model = "qwen3:0.6b"  # optional, for simple queries

[voice]
mode = "basic"             # basic | off
piper_model = "~/.local/share/luna/kokoro/kokoro-v1.0.onnx"
piper_bin = "af_heart"     # Kokoro voice name

[memory]
context_window = 6
```

## Voice Setup

```bash
# Download Kokoro models
mkdir -p ~/.local/share/luna/kokoro
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -O ~/.local/share/luna/kokoro/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -O ~/.local/share/luna/kokoro/voices-v1.0.bin

# Install Kokoro Python deps
python3.11 -m venv ~/.local/share/luna/rvc_env
~/.local/share/luna/rvc_env/bin/pip install kokoro-onnx soundfile

# Download Whisper model for voice input
mkdir -p ~/.local/share/luna/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin \
  -O ~/.local/share/luna/models/ggml-base.en.bin
```

## Architecture

```
main.rs
├── agent/mod.rs      — main loop, mode routing, escalation
├── llm/
│   ├── ollama.rs     — Ollama HTTP client (streaming + tool calls)
│   ├── react.rs      — ReAct loop with empty retry
│   └── escalation.rs — simple/complex query classifier
├── memory/
│   ├── mod.rs        — rolling context window
│   └── permanent.rs  — persistent fact store
├── tools/
│   ├── mod.rs        — tool registry + executor
│   ├── shell.rs      — bash command runner with sudo injection
│   ├── filesystem.rs — file read/write
│   ├── desktop.rs    — notifications, xdg-open
│   └── web.rs        — DuckDuckGo search
├── tts/
│   └── piper.rs      — Kokoro TTS via Python subprocess
├── stt/
│   └── whisper.rs    — Whisper STT via whisper-cli subprocess
└── audio/
    └── capture.rs    — mic capture with adaptive VAD
```

## Luna Versions

- **v1** — bash, simple keyword matching, ChromaDB RAG
- **v2** — bash, intent routing, model escalation, daemon socket  
- **v3 (this)** — Rust, proper ReAct agent, dual memory, voice I/O

Built by [Netrunner](https://github.com/X-netrunner) — MIT Bengaluru
