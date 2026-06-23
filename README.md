# Luna — Local AI Assistant

A fast, personal AI assistant built in Rust, running entirely locally on your machine. No cloud, no subscriptions, no data leaving your system.

## Features

- **Local-first** — runs via Ollama, everything stays on your machine
- **ReAct agent loop** — reasons and executes tools in a chain, with automatic retry on empty responses
- **Model escalation** — small/fast model for simple chat, full model for tool-heavy or complex tasks
- **Dual memory** — rolling conversation context + permanent memory that survives restarts and `clear`
- **Shell history context** — learns your workflow and app names from fish shell history
- **System indexer** — scans and maps your projects, scripts, and configs into permanent memory
- **Voice I/O** — Whisper STT + Kokoro TTS (high quality, runs on CPU)
- **Voice session mode** — say the wake word once, keep talking without repeating it until you say goodbye or go quiet
- **Proactive monitoring** — background checks for low battery, low disk space, and pending package updates, with real desktop notifications (no LLM call, zero hallucination risk)
- **Desktop integration** — opens apps, edits files, reads/writes the clipboard, sends notifications
- **Web learning** — one tool call searches the web AND fetches the most relevant page, so Luna can learn about a topic and `remember` it permanently
- **Todoist integration** — list, add, and complete tasks in your real Todoist account
- **Sudo passthrough** — runs privileged commands without ever hanging on an interactive prompt

## Tools

| Tool | Description |
|------|-------------|
| `run_shell` | Run any bash command |
| `edit_file` | Open a file in the editor (zeditor) |
| `read_file` / `write_file` | Read/write files |
| `notify` | Desktop notification |
| `system_info` | Battery, CPU, RAM, temp, disk, uptime |
| `clipboard` | Read/write the Wayland clipboard |
| `fetch_page` | Fetch a single webpage's text |
| `learn_topic` | Search + fetch the best result in one call — use this over `fetch_page` for open-ended research |
| `remember` / `forget` / `list_memories` | Manage permanent memory |
| `index_system` | Scan the home directory and save a structured map to permanent memory |
| `todoist_list` / `todoist_add` / `todoist_complete` | Manage Todoist tasks (requires an API token) |

## Requirements

- Rust 1.75+
- [Ollama](https://ollama.com) with a model pulled (default: `qwen2.5:7b-instruct-q4_K_M`)
- `whisper-cli` (from the `whisper.cpp` AUR package) for voice input
- Python 3 + Kokoro ONNX for voice output (see Voice Setup below — no Piper needed)
- `wl-copy` / `wl-paste` for clipboard (Wayland)
- `curl` for web fetch and learning
- (Optional) `pacman-contrib` for proactive update checks via `checkupdates`

## Installation

```bash
git clone https://github.com/X-netrunner/LUNA.git
cd LUNA
cargo build --release
./target/release/luna
```

## Configuration

On first run, Luna creates `~/.config/luna/luna.toml` with defaults. Copy `luna.toml.example` from this repo for a documented starting point — **never commit your real `luna.toml`**, since it can contain your sudo password and Todoist API token.

```toml
[agent]
system_prompt = "You are Luna..."
max_react_iterations = 8
sudo_password = ""          # set here or leave blank and enter at runtime

[llm]
model = "qwen2.5:7b-instruct-q4_K_M"
base_url = "http://localhost:11434"
fast_model = "qwen3:0.6b"   # optional, used for simple/conversational queries

[voice]
mode = "basic"               # basic | off
piper_model = "/home/YOU/.local/share/luna/kokoro/kokoro-v1.0.onnx"
piper_bin = "af_heart"       # Kokoro voice name: af_heart | af_sky | af_nicole | af_sarah

[audio]
input_mode = "both"          # off | push_to_talk | wake_word | both
wake_word = "hey luna"
wake_aliases = ["hey luna", "hello luna", "hay luna"]
vad_silence_ms = 800

[memory]
context_window = 6

[todoist]
api_token = ""                # get one at todoist.com/app/settings/integrations

[proactive]
enabled = true
check_interval_mins = 15
battery_low_threshold = 20
disk_full_threshold = 90
check_updates = true
```

## Voice Setup

Kokoro doesn't need a heavy ML stack — it runs on `onnxruntime`, which Arch already packages, so the venv can reuse your system packages instead of pip-building everything from scratch:

```bash
# Download Kokoro models
mkdir -p ~/.local/share/luna/kokoro
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
  -O ~/.local/share/luna/kokoro/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin \
  -O ~/.local/share/luna/kokoro/voices-v1.0.bin

# Make sure the heavy stuff is installed via pacman, not pip
sudo pacman -S python-numpy python-onnxruntime-cpu python-soundfile

# Lightweight venv that reuses the system packages above
python3 -m venv --system-site-packages ~/.local/share/luna/tts_env
~/.local/share/luna/tts_env/bin/pip install kokoro-onnx soundfile

# Download the Whisper model for voice input
mkdir -p ~/.local/share/luna/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin \
  -O ~/.local/share/luna/models/ggml-base.en.bin
```

Test Kokoro directly before relying on Luna to call it:

```bash
~/.local/share/luna/tts_env/bin/python3 -c "
from kokoro_onnx import Kokoro
import soundfile as sf
k = Kokoro('$HOME/.local/share/luna/kokoro/kokoro-v1.0.onnx', '$HOME/.local/share/luna/kokoro/voices-v1.0.bin')
samples, sr = k.create('Hello, I am Luna.', voice='af_heart', speed=1.0, lang='en-us')
sf.write('/tmp/test.wav', samples, sr)
"
aplay /tmp/test.wav
```

## Todoist Setup

1. Get your API token at `todoist.com/app/settings/integrations`
2. Add it to `luna.toml` under `[todoist] api_token = "..."`
3. Never commit this token — `luna.toml` should always be gitignored

## Architecture

```
main.rs
├── agent/mod.rs       — main loop, hybrid/voice/text routing, voice session mode
├── llm/
│   ├── ollama.rs      — Ollama HTTP client (streaming + tool calls)
│   ├── react.rs       — ReAct loop with empty-response retry
│   └── escalation.rs  — simple/complex query classifier for model routing
├── memory/
│   ├── mod.rs         — rolling context window with tool-artifact filtering
│   └── permanent.rs   — persistent fact store, survives `clear` and restarts
├── tools/
│   ├── mod.rs         — tool registry + executor
│   ├── shell.rs       — bash command runner with sudo injection (never hangs on a prompt)
│   ├── filesystem.rs  — file read/write
│   ├── desktop.rs     — notifications, xdg-open
│   ├── web.rs         — DuckDuckGo search
│   ├── learn.rs        — combined search + fetch for one-shot research
│   ├── todoist.rs      — Todoist Unified API v1 client
│   ├── proactive.rs     — background battery/disk/update monitor
│   ├── todo.rs          — local task list
│   └── remind.rs        — in-process reminders
├── tts/
│   └── piper.rs        — Kokoro TTS via Python subprocess
├── stt/
│   └── whisper.rs      — Whisper STT via whisper-cli subprocess
└── audio/
    └── capture.rs      — mic capture with adaptive, calibrated VAD
```

## Luna Versions

- **v1** — bash, simple keyword matching, ChromaDB RAG
- **v2** — bash, intent routing, model escalation, daemon socket
- **v3 (this one)** — Rust, ReAct agent, dual memory, voice I/O, Todoist, proactive monitoring

Built by [Netrunner](https://github.com/X-netrunner) — MIT Bengaluru
