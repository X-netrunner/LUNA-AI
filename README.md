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
- **Inline wake-word commands** — say "luna what's the time" in one breath; Luna strips the wake word and runs the rest as a command
- **Proactive monitoring** — background checks for low battery, low disk space, and pending package updates, with real desktop notifications (no LLM call, zero hallucination risk)
- **Background daemon** — `luna --daemon` watches for RAM/CPU hogs, learns which apps you use daily, reclaims disk space safely, and can auto-end idle processes you approve by chat
- **Secrets in the OS keyring** — API keys live encrypted in the Secret Service instead of plaintext TOML (`luna --set-key gemini`)
- **Three-tier web search** — Tavily (keyless) first, Gemini as knowledge fallback, DuckDuckGo instant answers last; pages fetched via Firecrawl (handles JavaScript)
- **Source validation** — asks like "is X a scam?" trigger a Reddit-first search to cross-check claims before answering
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
| `web_search` | Three-tier search: Tavily → Gemini → DuckDuckGo instant answers |
| `fetch_page` | Fetch a single webpage's text (Firecrawl keyless, JS-aware) |
| `learn_topic` | Search + fetch the best result in one call — use this over `fetch_page` for open-ended research |
| `process_stats` | Show what the daemon learned about your process usage |
| `allow_autokill` / `deny_autokill` | Grant/revoke idle auto-kill for a process (daemon-managed) |
| `nmap_scan` / `analyze_pcap` / `decode_payload` / `hash_file` / `dns_lookup` | CTF/network toolkit: scanning, pcap analysis, decoding, hashing, DNS/whois |
| `remember` / `forget` / `list_memories` | Manage permanent memory |
| `index_system` | Scan the home directory and save a structured map to permanent memory |
| `todoist_list` / `todoist_add` / `todoist_complete` | Manage Todoist tasks (requires an API token) |

## Requirements

- Rust 1.75+
- [Ollama](https://ollama.com) with a model pulled (default: `qwen2.5:7b-instruct-q4_K_M`)
- `whisper-cli` (from the `whisper.cpp` AUR package) + `ggml-small.en.bin` and `ggml-silero-v6.2.0.bin` models for voice input
- Python 3 + Kokoro ONNX for voice output (see Voice Setup below — no Piper needed)
- `wl-copy` / `wl-paste` for clipboard (Wayland)
- `curl` for web fetch and learning
- (Optional) `pacman-contrib` for proactive update checks via `checkupdates`

## Installation

```bash
git clone https://github.com/X-netrunner/LUNA-AI.git
cd LUNA-AI
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
whisper_model = "/home/YOU/.local/share/luna/models/ggml-small.en.bin"

[audio]
input_mode = "both"          # off | wake_word | both
wake_word = "hey luna"
wake_aliases = ["luna", "hey luna", "hello luna", "hay luna"]
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

[search]
# All optional — works keyless out of the box. Free keys boost quotas:
tavily_api_key = ""           # tavily.com (1000 free searches/month)
gemini_api_key = ""           # aistudio.google.com
# Or keep them OUT of this file entirely (see Secrets below):
# gemini_api_key = "keyring:gemini"
```

## Secrets (OS keyring)

API keys never need to touch disk in plaintext. Store a secret in the
freedesktop Secret Service and reference it by name:

```bash
luna --set-key gemini     # prompts without echo, stores luna/gemini
luna --set-key todoist
luna --get-key gemini     # print for verification
```

```toml
[search]
gemini_api_key = "keyring:gemini"    # resolved at startup, nothing on disk
```

Luna also chmods `luna.toml` to 600 every time it saves it.

## Background Daemon

Run the watchdog standalone (no Ollama, no tty):

```bash
luna --daemon                                   # foreground
cp deploy/luna-daemon.service ~/.config/systemd/user/
systemctl --user enable --now luna-daemon       # as a service
```

Three jobs:

1. **Process watchdog** — flags single processes above RAM/CPU thresholds
   via desktop notification, with the exact pid to end. Never kills anything
   on its own initiative.
2. **Usage learning** — every scan is fed into a per-process profile
   (`~/.local/share/luna/process_stats.json`). Apps seen on ≥5 of the last
   7 days count as *daily use* and are silently ignored from then on.
   Idle non-daily processes become auto-kill candidates: ones you approved
   (`allow auto-kill steam`) get SIGTERMed after ~30 idle minutes;
   everything else only earns an opt-in suggestion once per day. Stateful
   apps (browsers, editors, terminals, chat) are protected no matter what.
3. **Disk hygiene** — measures reclaimable space in the pacman cache,
   `~/.cache`, trash, and journals. In `notify` mode (default) it only
   reports; in `auto` mode it cleans those four safe locations when `/`
   crosses your disk-full threshold.

All thresholds, ignore lists, and safety rails live under `[daemon]` in
`luna.toml` — see `luna.toml.example` for the full documented set.

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

# Download the Whisper model + Silero VAD model for voice input
mkdir -p ~/.local/share/luna/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin \
  -O ~/.local/share/luna/models/ggml-small.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-v6.2.0.bin \
  -O ~/.local/share/luna/models/ggml-silero-v6.2.0.bin
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
2. Add it to `luna.toml` under `[todoist] api_token = "..."` — or better,
   store it with `luna --set-key todoist` and set
   `api_token = "keyring:todoist"`
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
│   ├── desktop.rs     — notifications
│   ├── web.rs         — Tavily → Gemini → DuckDuckGo search chain
│   ├── learn.rs       — combined search + fetch for one-shot research
│   ├── security.rs    — nmap / tshark / encoding / hashing / DNS for CTF work
│   ├── todoist.rs     — Todoist Unified API v1 client
│   └── proactive.rs   — background battery/disk/update monitor
├── daemon/
│   ├── mod.rs         — daemon entry loop + idle-process policy
│   ├── watchdog.rs    — /proc scanner (RAM/CPU/jiffies)
│   ├── tracker.rs     — usage learning, daily-use classification, allowlist
│   └── cleanup.rs     — safe disk hygiene (pacman cache, ~/.cache, trash, journals)
├── tts/
│   └── piper.rs       — Kokoro TTS via Python subprocess
├── stt/
│   └── whisper.rs     — Whisper STT via whisper-cli subprocess
└── audio/
    └── capture.rs     — mic capture with adaptive, calibrated VAD
```

## Luna Versions

- **v1** — bash, simple keyword matching, ChromaDB RAG
- **v2** — bash, intent routing, model escalation, daemon socket
- **v3 (this one)** — Rust, ReAct agent, dual memory, voice I/O, Todoist,
  three-tier web search, keyring secrets, background daemon with usage
  learning and idle auto-kill

Built by [Srijan Satya Bandaru](https://www.linkedin.com/in/srijan-bandaru-nex/) — MIT Bengaluru
