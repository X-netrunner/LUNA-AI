# Luna — Local AI Assistant

A fast, personal AI assistant built in Rust, running entirely locally on your machine. No cloud, no subscriptions, no data leaving your system.

## Features

- **Local-first** — runs via Ollama, everything stays on your machine
- **ReAct agent loop** — reasons and executes tools in a chain, with automatic retry on empty responses
- **Model escalation** — small/fast model for simple chat, full model for tool-heavy or complex tasks
- **Dual memory with semantic recall** — permanent facts are embedded locally (nomic-embed-text via Ollama) and only the ones relevant to your current question get injected; if the embedding model is missing, Luna falls back to the full dump
- **Self-correcting model escalation** — the fast model says "ESCALATE" when a query needs tools, and the full model transparently takes over
- **Real reminders** — "remind me in 20 minutes" fires as a desktop notification even with chat closed (daemon polls `reminders.json` and wakes early for them)
- **Self-learning** — the daemon periodically distills your fish history into workflow facts (top commands, most-visited directories, most-edited files) and re-indexes your projects/scripts/configs into permanent memory, monthly by default; "learn about my system" runs it on demand
- **Self-improving loop (Hermes-style)** — every `nudge_interval` turns Luna reviews the conversation and proactively *remembers* durable facts and *creates skills* from repeatable procedures (`~/.local/share/luna/skills/`); relevant skills are recalled into the prompt by semantic similarity, used skills get refreshed when they go stale, past sessions get titles + summaries, and `search_history` finds old conversations
- **Safety check** — weekly `run_safety_check` runs pacman -Syu, ClamAV, rkhunter, UFW, Lynis and monthly AIDE, with a light scan of hot dirs that escalates to a full-home antivirus scan monthly; the daemon runs it detached and a systemd timer can back it up when the daemon is down
- **Incremental backups** — the weekly `backup` tool snapshots the home directory to `/dev/sda1` at `/mnt/backup/arch-backup/` via hardlink-incremental rsync (only new changes cost space) while the drive is plugged in
- **WhatsApp messaging** — `luna --whatsapp-link` pairs your account once over QR; `whatsapp_send` lets Luna send messages, look up contacts by name, or list the full contact book (names ↔ numbers)
- **Browser automation** — `browser_do` connects to your Project-Vision (SIH) VLM planner and a real Chromium window to execute goals like "add this to cart" or "fill this form"
- **sysmode hardening** — the `sysmode` tool switches system profiles (`secure` / `cyber` / `stealth` / `lockdown`), functional self-tests the honeypot + IDS + decoy stack ("is the honeypot working?", "test if everything is running"), and pulls attack logs (needs the external `sysmode` script + a sudo password)
- **Shell history context** — recent fish commands are injected into every session prompt
- **Background daemon** — `luna --daemon` watches for RAM/CPU hogs, learns which apps you use daily, reclaims disk space safely, and can auto-end idle processes you approve by chat
- **Voice I/O** — Whisper STT + Kokoro TTS (high quality, runs on CPU)
- **Voice session mode** — say the wake word once, keep talking without repeating it until you say goodbye or go quiet
- **Explicit voice mode** — say “luna voice mode” to enter hands-free chatting (no wake word needed); say “luna voice mode off/down” to return, or just go quiet for the configured `voice_mode_idle_mins` and it switches back automatically
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
| `create_skill` / `use_skill` / `list_skills` / `forget_skill` | Save, load, list, and delete reusable skills — Luna creates them automatically from repeatable tasks and improves them when they go stale |
| `search_history` | Search all past conversations (titles, summaries, and turn-by-turn text) |
| `memory_report` | What Luna has permanently learned (workflow, system index, stats) |
| `set_reminder` / `list_reminders` / `cancel_reminder` | Scheduled reminders that fire even when chat is closed |
| `index_system` | Deeply learn the system — maps home projects/scripts/configs AND analyzes fish history (top commands, directories, files) into permanent memory (also runs periodically via daemon); "learn about my system" triggers it |
| `run_safety_check` | Run the weekly safety check (pacman -Syu, ClamAV, rkhunter, UFW, Lynis, AIDE, backup) or report on the last run |
| `backup` | Incremental home-directory backup to /dev/sda1 (/mnt/backup/arch-backup) or a status report |
| `whatsapp_send` | Send WhatsApp messages through your own linked account (pair once with `luna --whatsapp-link`). Actions: `send` (to + text), `lookup` (name → number), `contacts` (list full contact book with names & numbers), `status` (bridge health) |
| `browser_do` | **Run real browser tasks** — spins up the Project-Vision (SIH) VLM planner + a visible Chromium window to execute goals like "add the first PS5 result to cart on amazon.com" or "fill this Google Form". Blocks until done; profile persists so logins only need to happen once. |
| `sysmode` | Switch system hardening profiles (`secure`/`cyber`/`stealth`/`lockdown`), run a functional self-test of the honeypot/IDS/decoy stack ("is the honeypot working?"), or pull attack logs |
| `todoist_list` / `todoist_add` / `todoist_complete` | Manage Todoist tasks (requires an API token) |
| `spotify` | Control Spotify via the Web API: liked songs, playlists, search, transport (requires Premium + one-time OAuth) |

## Requirements

- Rust 1.75+
- [Ollama](https://ollama.com) with a model pulled (default: `qwen2.5:7b-instruct-q4_K_M`)
- `whisper-cli` (from the `whisper.cpp` AUR package) + `ggml-medium.en.bin` and `ggml-silero-v6.2.0.bin` models for voice input
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
fast_model = "qwen2.5:3b"   # non-reasoning small model — instant replies, no hidden thinking
deep_model = "qwen2.5:7b-instruct-q4_K_M"   # optional, for code/reasoning-heavy tasks
embedding_model = "nomic-embed-text"          # local embedding model for semantic memory recall

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
voice_mode_idle_mins = 5    # "luna voice mode" auto-ends after this many min of silence (0 = never)

[memory]
context_window = 6

[todoist]
api_token = ""                # get one at todoist.com/app/settings/integrations

[spotify]
client_id = "keyring:spotify_id"
# PKCE flow needs no client secret — only the client id above.

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

[logging]
level = "debug"              # info | debug | trace — debug shows full tool/thinking output

[daemon]
enabled = true
check_interval_mins = 30
notify_hours = 1              # "I'm alive" desktop notification every N hours (0 = off)

[browser]
enabled = true
srijan_url = "ws://127.0.0.1:8001/ws"
server_dir = "/home/YOU/Projects/sih/Project-Vision/Server[unnati&srijan]"
cdp_port = 9222
headless = false
timeout_secs = 600
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
cargo build --release
install -Dm755 target/release/luna ~/.local/bin/luna        # one-time
luna --daemon                                   # foreground
mkdir -p ~/.config/systemd/user
cp deploy/luna-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now luna-daemon       # as a service
```

If the daemon reports no notifications, confirm the service is actually
running: `systemctl --user status luna-daemon`.

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
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin \
  -O ~/.local/share/luna/models/ggml-medium.en.bin
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

## Spotify Setup (optional — requires Spotify Premium)

Luna can control your Spotify over the Web API: play your Liked Songs,
playlists, artists, albums, search, and transport control. Playback needs a
**Premium** account (play endpoints are 403 on free tiers).

1. Go to `developer.spotify.com/dashboard`, create an app (any name), and add
   `http://127.0.0.1:8888/callback` as a **Redirect URI**.
2. Store the Client ID in the keyring (auth uses PKCE — **no client secret
   needed**, just the id):
   ```bash
   luna --set-key spotify_id
   ```
3. Point `luna.toml` at it:
   ```toml
   [spotify]
   client_id = "keyring:spotify_id"
   ```
4. Authorize once:
   ```bash
   luna --spotify-auth
   ```
   A browser tab opens to approve access; Luna catches the redirect on
   `127.0.0.1:8888`, and the refresh token is stored in the keyring while
   `luna.toml` is updated with `refresh_token = "keyring:spotify_refresh"`.
5. Keep a Spotify client playing on a device signed in to the same account
   (desktop/mobile/web player all work).

Say things like: *"play my liked songs shuffled", "play my gym playlist",
"play bohemian rhapsody", "what's playing on spotify", "next track".*
If no device is active, Luna picks one automatically. If this errors, just
re-run `luna --spotify-auth`.

## Architecture

```
main.rs
├── agent/
│   ├── mod.rs          — main loop, hybrid/voice/text routing, voice session mode
│   └── learning.rs     — self-improvement loop: memory nudge counter, skill recall,
│                         user-profile distill, conversation log, session summaries
├── llm/
│   ├── ollama.rs       — Ollama HTTP client (streaming + tool calls)
│   ├── react.rs        — ReAct loop with empty-response retry + per-turn prompt enrichment
│   └── escalation.rs   — simple/complex query classifier for model routing
├── memory/
│   ├── mod.rs          — rolling context window with tool-artifact filtering
│   ├── permanent.rs    — persistent fact store, survives `clear` and restarts
│   ├── recall.rs       — semantic recall (embedding cache, RAG-lite)
│   ├── skills.rs       — reusable skill store (`.md` files + frontmatter, ~/.local/share/luna/skills/)
│   └── workflow.rs     — fish-history learning + system indexing
├── tools/
│   ├── mod.rs          — tool registry + executor
│   ├── shell.rs        — bash command runner with sudo injection (never hangs on a prompt)
│   ├── filesystem.rs   — file read/write
│   ├── desktop.rs      — notifications
│   ├── web.rs          — Tavily → Gemini → DuckDuckGo search chain
│   ├── learn.rs        — combined search + fetch for one-shot research
│   ├── security.rs     — nmap / tshark / encoding / hashing / DNS for CTF work
│   ├── todoist.rs      — Todoist Unified API v1 client
│   └── proactive.rs    — background battery/disk/update monitor
├── daemon/
│   ├── mod.rs          — daemon entry loop + idle-process policy
│   ├── watchdog.rs     — /proc scanner (RAM/CPU/jiffies)
│   ├── tracker.rs      — usage learning, daily-use classification, allowlist
│   └── cleanup.rs      — safe disk hygiene (pacman cache, ~/.cache, trash, journals) + orphaned-package auto-removal
├── tts/
│   └── mod.rs          — Kokoro TTS via Python subprocess
├── stt/
│   └── whisper.rs      — Whisper STT via whisper-cli subprocess
└── audio/
    └── capture.rs      — mic capture with adaptive, calibrated VAD
```

## Luna Versions

- **v1** — bash, simple keyword matching, ChromaDB RAG
- **v2** — bash, intent routing, model escalation, daemon socket
- **v3 (this one)** — Rust, ReAct agent, dual memory, voice I/O, Todoist,
  three-tier web search, keyring secrets, background daemon with usage
  learning and idle auto-kill

Built by [Srijan Satya Bandaru](https://www.linkedin.com/in/srijan-bandaru-nex/) — MIT Bengaluru
