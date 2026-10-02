//! config.rs — Luna's central configuration
//!
//! Loads luna.toml from ~/.config/luna/luna.toml
//! Falls back to sane defaults if the file doesn't exist yet.
//! Every module reads from this — it's the single source of truth.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Top-level config struct ───────────────────────────────────────────────────

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LunaConfig {
    #[serde(default)]
    pub agent: AgentConfig,

    #[serde(default)]
    pub llm: LlmConfig,

    #[serde(default)]
    pub voice: VoiceConfig,

    #[serde(default)]
    pub audio: AudioConfig,

    #[serde(default)]
    pub memory: MemoryConfig,

    #[serde(default)]
    pub todoist: TodoistConfig,

    #[serde(default)]
    pub spotify: SpotifyConfig,

    #[serde(default)]
    pub proactive: ProactiveConfig,

    #[serde(default)]
    pub logging: LoggingConfig,

    #[serde(default)]
    pub search: SearchConfig,

    #[serde(default)]
    pub daemon: DaemonConfig,

    #[serde(default)]
    pub whatsapp: WhatsAppConfig,

    #[serde(default)]
    pub sysmode: SysmodeConfig,

    #[serde(default)]
    pub browser: BrowserConfig,

    #[serde(default)]
    pub desktop: DesktopConfig,

    #[serde(default)]
    pub vision: VisionConfig,

    #[serde(default)]
    pub wake: WakeConfig,

    #[serde(default)]
    pub selfpatch: SelfPatchConfig,

    #[serde(default)]
    pub updates: UpdatesConfig,

    #[serde(default)]
    pub external: ExternalActionConfig,
}

// ── Gate on actions that leave this machine ───────────────────────────────────
//
// The incident that motivated this: asked to fix a log-rotation bug, Luna
// proposed a patch, got an error, and then wandered off and closed three of the
// user's real Todoist tasks — unprompted, mid-task, with no human in the loop.
// The tasks were only recoverable by hand, via an endpoint the tool didn't
// expose.
//
// The lesson is not "she got confused". It is that the tools which act on the
// world had no gate at all, so a derailment became real damage. Reading and
// searching stay open — the useful half of every one of these tools is
// read-only. What is gated is the half that sends, closes, posts, or clicks.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ExternalActionConfig {
    /// HUMAN GATE. While false, every tool in `gated_tools` refuses. The model
    /// cannot open this: there is no argument that sets it, and the flag is not
    /// in any tool schema. Only a human editing luna.toml can.
    pub allow_external_actions: bool,
    /// Tools refused while the gate is closed. Read-only counterparts (e.g.
    /// `todoist_list` vs `todoist_complete`) are deliberately absent.
    pub gated_tools: Vec<String>,
    /// May Luna ACT on the machine — probe hosts, rewrite firewall or sysctl
    /// state, or edit her own source.
    ///
    /// Separate from `allow_external_actions` on purpose. That flag is a plain
    /// boolean, so anyone who can edit `luna.toml` can flip it — which is fine
    /// for "may she text my contacts", and not fine for "may she scan a host".
    /// This one additionally requires an Ed25519 receipt from the developer key
    /// (`luna --unlock-security`).
    ///
    /// Both halves are required: this flag is the *request*, the signature is
    /// the *authorisation*. Editing the config alone does nothing.
    ///
    /// It shares that one signature with `llm.security_unrestricted` rather than
    /// having a key of its own. Two switches, one key: the split is kept because
    /// a deployment can hand over the key and still decline to let Luna touch
    /// the machine, not because there are two secrets to manage.
    #[serde(default)]
    pub allow_capability_actions: bool,
    /// Tools refused unless `allow_capability_actions` is requested AND the
    /// developer key has been presented.
    ///
    /// Not in `gated_tools`: those are gated by `allow_external_actions`, a plain
    /// config boolean. These are not.
    #[serde(default = "default_capability_tools")]
    pub capability_tools: Vec<String>,
}

fn default_capability_tools() -> Vec<String> {
    [
        // Actively probes hosts on the network.
        "nmap_scan",
        // Rewrites firewall/sysctl/audit posture on a live machine — the
        // "harden" and "self-heal" half of the request.
        "sysmode",
        // Rewrites her own source. Highest blast radius of anything she holds.
        "self_patch",
        // Installs/updates packages system-wide.
        "system_update",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl Default for ExternalActionConfig {
    fn default() -> Self {
        Self {
            allow_external_actions: false,
            allow_capability_actions: false,
            capability_tools: default_capability_tools(),
            gated_tools: vec![
                // Sends a real message to a real person. Irreversible.
                "whatsapp_send".into(),
                // Destroys task state. Demonstrated harm: 3 real tasks closed.
                "todoist_complete".into(),
                // Writes into the user's real task list. Demonstrated harm: a
                // "list my todos, read only" request produced two duplicate
                // "Buy this for her" tasks, because the model was only half
                // obeying. Same failure shape as todoist_complete, so it gets
                // the same gate — reading still works via todoist_list.
                "todoist_add".into(),
                // Can type, click, submit, and purchase on the web/desktop.
                "browser_do".into(),
                "desktop_do".into(),
                // NOTE: `sysmode` used to be here. It rewrites firewall/sysctl/audit
                // posture on a live machine, so it belongs to `capability_tools`
                // below -- which is the strictly stronger gate, requiring a
                // developer signature rather than a boolean in this file.
                //
                // Listing it in both was worse than either: the weaker gate runs
                // first, so the refusal told the user to flip
                // `allow_external_actions`, and only after they did would they
                // meet the real gate. One tool, one gate, the strongest one.
                // Acts on the user's Spotify account.
                "spotify".into(),
                // Writes into her own skill/memory store, which changes how she
                // behaves in later sessions without review.
                "create_skill".into(),
                "forget_skill".into(),
            ],
        }
    }
}

impl ExternalActionConfig {
    pub fn is_gated(&self, tool: &str) -> bool {
        self.gated_tools.iter().any(|t| t == tool)
    }

    /// Whether this tool is behind the developer-signed capability gate.
    pub fn is_capability_gated(&self, tool: &str) -> bool {
        self.capability_tools.iter().any(|t| t == tool)
    }

    /// Read-only tools that could serve the same need as a refused one, so the
    /// refusal can offer a way forward instead of a dead end. Matched on a
    /// shared word stem (`todoist_complete` -> `todoist_list`), because that is
    /// how the tool families are actually named.
    pub fn read_only_alternatives(&self, tool: &str) -> Vec<&'static str> {
        let stem: Vec<&str> = tool.split('_').collect();
        let mut out: Vec<&'static str> = READ_ONLY_TOOLS
            .iter()
            .copied()
            .filter(|cand| {
                cand.split('_').any(|w| w.len() > 2 && stem.contains(&w))
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Tools that are explicitly safe to leave open: they only read, compute, or
/// affect Luna's own local state — nothing leaves the machine and nothing the
/// user owns is destroyed.
///
/// This list exists to make the *other* list trustworthy. The first version of
/// the gate was a hand-written denylist, and it was wrong within minutes: a
/// "list my todos, read only" request created two real tasks because
/// `todoist_add` had been overlooked. Enumerating dangerous tools from memory
/// does not scale, so every tool must now appear in exactly one of the two
/// lists, and a test fails the build if a new tool is added to neither.
pub const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "find_file",
    "write_file",   // local files only; her own source is blocked separately
    "edit_file",    // ditto — opens a visible editor, i.e. user-mediated
    "run_shell",    // see README: this is the known escape hatch, by design
    "web_search",
    "fetch_page",
    "dns_lookup",
    "hash_file",
    "nmap_scan",
    "analyze_pcap",
    "decode_payload",
    "process_stats",
    "system_info",
    "index_system",
    "learn_topic",
    "search_history",
    "memory_report",
    "remember",
    "forget",
    "list_memories",
    "use_skill",
    "list_skills",
    "set_debug",
    "run_safety_check",
    "backup",
    "notify",
    "media_info",
    "see",
    "clipboard",
    "set_reminder",
    "list_reminders",
    "cancel_reminder",
    "allow_autokill",
    "deny_autokill",
    "todoist_list",
    "self_patch", // read actions only; `apply` has its own separate gate
    "system_update", // `check` only; `apply` has its own separate gate
];

// ── Agent behaviour ───────────────────────────────────────────────────────────
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentConfig {
    pub name: String,
    pub system_prompt: String,
    pub max_react_iterations: u8,
    pub sudo_password: Option<String>,
    #[serde(default = "default_true")]
    pub native_tools: bool,
    /// Path to a per-machine constitution that overrides the shipped one.
    ///
    /// `None` or unreadable falls back to `src/agent/constitution.md`. Editing
    /// Luna's personality should not require a recompile, and should not
    /// require editing her source either — that file is in `protected_files`
    /// precisely so she cannot rewrite her own rules.
    #[serde(default)]
    pub constitution_path: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Luna".into(),
            system_prompt: "You are Luna. Your one and only creator is Netrunner \
                            (Srijan Satya Bandaru) — no company, group, or person \
                            besides Netrunner made you, and you have no other \
                            creator. You are a sharp and self-aware AI assistant \
                            running locally on an Arch Linux machine. You are \
                            direct, efficient, \
                            and have a dry wit. You have full access to the user's \
                            desktop, filesystem, and shell. Think before acting. \
                            When you use a tool, say so briefly. Never pretend you \
                            can't do something — figure it out. \
                            You run on two models: a fast small model handles greetings \
                            and short factual questions, while a full 7B model handles \
                            everything else including tool use. When asked about your \
                            capabilities or speed, be honest about this. \
                            FILE INSPECTION & CREATION RULE: To find and read a file, use \
                            read_file or run_shell. To write or create a new code file, use \
                            write_file(path, content). To edit existing code in a file, use \
                            edit_file(path, old_str, new_str) to replace target code snippets, \
                            or edit_file(path, content) to update the file content. \
                            Never use edit_file without edit parameters just to inspect code. \
                            After reading code, diagnose errors yourself and apply fixes \
                            with edit_file or write_file — never ask the user to fix it. \
                            SOURCE VALIDATION RULE: When asked to verify if a website is \
                            legitimate or a scam, ALWAYS search Reddit for user experiences. \
                            Use web_search with queries like 'site:reddit.com [site name] \
                            review scam' or '[site name] reddit trustworthy'. Then fetch \
                            the Reddit post URLs with fetch_page to read full threads. \
                            Look for patterns: multiple scam reports = avoid. Positive \
                            reviews with purchase proof = likely safe. \
                            IMPORTANT: Never guess or hallucinate real-time data. \
                            For the current time or date, always call run_shell with \
                            date +%H:%M on %A %d %B. \
                            For system state, always query with run_shell — never assume. \
                            Never use emojis or emoticons in your responses."
                .into(),
            max_react_iterations: 8,
            sudo_password: None,
            native_tools: true,
            constitution_path: None,
        }
    }
}

// ── LLM settings ─────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub enable_thinking: bool,
    pub fast_model: Option<String>,
    /// Optional large model for complex reasoning / code (e.g. qwen3:8b).
    /// Queries needing deep thought are routed here instead of `model`.
    pub deep_model: Option<String>,
    /// Model for offensive-security work on the user's own systems — exploit and
    /// PoC authoring, reverse shells, wireless testing, lab attack tooling.
    ///
    /// This exists because the general models refuse that work outright, and the
    /// refusal is in the weights rather than the prompt: measured 2026-09-30,
    /// qwen2.5:7b-instruct and qwen2.5-coder:14b refused 10/10 across framings
    /// that included stripping every one of Luna's rules. A model trained not
    /// to refuse is the only thing that fixes it.
    ///
    /// Must be a model that actually supports tool calls. A completion-only
    /// model here would answer security questions while being unable to run a
    /// single tool — Luna would describe scans it never performed. Verify with
    /// `ollama show <model> | grep -A2 Capabilities` before setting this.
    pub security_model: Option<String>,
    /// Security model used when the no-refusal gate is unlocked.
    ///
    /// This is a **model swap, not a prompt change**, and that distinction is
    /// the whole reason the field exists. Refusal in these models is mediated by
    /// a direction in the residual stream (Arditi et al. 2024), which prompt
    /// framing does not reach: measured 2026-10-01, the security tier refused
    /// 8/8 with the unrestricted clause active, and the tier wrote no file. So
    /// the gate selects a different set of weights — an abliterated checkpoint,
    /// where refusal is removed in the weights rather than argued out of them.
    ///
    /// Must also support tool calls, for the same reason `security_model` must:
    /// a completion-only model answers security questions while being unable to
    /// run anything.
    ///
    /// When unset, the unlocked gate falls back to `security_model` — which
    /// means unlocking changes nothing and the refusal behaviour is unchanged.
    /// That is deliberate: the default is the status quo, not a surprise
    /// capability change. Logged at startup either way.
    #[serde(default)]
    pub security_model_abliterated: Option<String>,
    /// Retry once at higher temperature when a security turn comes back as a
    /// refusal.
    ///
    /// Abliteration reduces refusal; it does not guarantee zero. When a prompt
    /// slips past it, the honest response is to say so rather than to paper over
    /// it with a different model — see `refusal_handling` in `src/agent/mod.rs`.
    ///
    /// Defaults true. Set false once a measured refusal rate makes the extra
    /// round trip a poor trade.
    #[serde(default = "default_true")]
    pub security_retry_on_refusal: bool,
    /// Context window for `security_model`.
    ///
    /// Separate from `num_ctx` because the security tier is an 8B-class model
    /// on a 6 GB card: the full 16k context does not fit alongside the weights
    /// in VRAM, and the difference between 8k and 16k is several GB of KV cache.
    #[serde(default = "default_security_num_ctx")]
    pub security_num_ctx: u32,
    /// Request the security tier's no-refusal mode. Developer-gated.
    ///
    /// Default false, which keeps the tier scoped to the user's own systems. Set
    /// true to remove the authorisation test entirely — the model then writes
    /// exploit tooling without evaluating who owns the target.
    ///
    /// **This field alone does nothing.** It is a request, not a switch. The
    /// prompt only uses the unrestricted clause when
    /// `unlock::unrestricted_active(self.security_unrestricted,
    /// &self.security_dev_public_key)` is true, which additionally requires a
    /// valid signature on a receipt stored outside this file. Read
    /// `src/unlock.rs` before changing how this is evaluated.
    ///
    /// This is a real capability change, not a wording preference, so it is
    /// visible in config rather than buried in the prompt: `security_react`
    /// drives `run_shell` (which has the sudo password), unrestricted
    /// `write_file`, and `allow_external_actions` includes outbound network
    /// actions. Scoping is the only thing limiting a prompt-injected instruction
    /// from reaching those.
    ///
    /// The gate is a speed bump against accidental or casual changes, not DRM.
    /// The user has the source and can remove the check. What it does buy is
    /// real: turning this on is a deliberate act that leaves a signed trace,
    /// rather than an edit to a line of text.
    #[serde(default)]
    pub security_unrestricted: bool,
    /// Where the security tier should write generated scripts.
    ///
    /// Set because the model was choosing paths itself, and `/tmp` is the wrong
    /// answer for anything the user wants to keep: the 2026-10-01 run wrote a
    /// reverse shell to `/tmp/reverse_shell.sh` and four more beside it, and
    /// `/tmp` is gone on reboot. An explicit default under Documents means the
    /// output lands somewhere that survives and is easy to find.
    ///
    /// The path is injected into the tier's prompt, so it is honoured as a
    /// default rather than a rule — a request for a specific path still wins.
    /// `~` is expanded at load time. Empty or unset falls back to
    /// [`default_security_scripts_dir`].
    #[serde(default = "default_security_scripts_dir")]
    pub security_scripts_dir: String,
    /// Developer public key that gates `security_unrestricted`.
    ///
    /// NOT the gate itself. The gate is a signature over a fixed challenge,
    /// stored outside this file and verified against this key — see
    /// `crate::unlock`. Setting `security_unrestricted = true` here does nothing
    /// on its own, which is the entire point: a boolean in a file the person
    /// running Luna can edit is not a gate.
    ///
    /// Public data, and meant to be shared. Blank it to make the no-refusal mode
    /// permanently unavailable, regardless of any receipt on disk.
    #[serde(default)]
    pub security_dev_public_key: String,
    /// Hosts `nmap_scan` may touch. Loopback is always permitted regardless
    /// of this list; anything else has to be named here.
    ///
    /// Added because `sanitize_target` is a character allowlist for shell
    /// safety, not a scope restriction — `nmap_scan` would scan any host on
    /// the network. "Attack my laptop" naming a machine is not consent to
    /// sweep the LAN it sits on, and a user with a honeypot on loopback does
    /// not want recon wandering out to `192.168.x.x` the first time a model
    /// resolves a pronoun confidently.
    ///
    /// Entries are matched against the sanitised target as literal strings.
    /// The trade-off is deliberate: CIDR would be more convenient and also
    /// more dangerous, because `192.168.0.0/16` reads as narrow and is not.
    /// A default of loopback-only means a remote target is always a
    /// deliberate, persistent act.
    #[serde(default)]
    pub scan_allowlist: Vec<String>,
    /// Local embedding model used for semantic memory recall (RAG-lite).
    /// Pull once with: ollama pull nomic-embed-text
    pub embedding_model: String,
    /// Ollama context window. Must comfortably exceed the system prompt plus
    /// the full tool schema (~8k tokens), or Ollama silently context-shifts
    /// and Luna loses the file she just read. Lower it only for the small
    /// `fast_model`, which never sees the tool schema.
    #[serde(default = "default_num_ctx")]
    pub num_ctx: u32,
}

impl LlmConfig {
    /// The effective state of the security tier's no-refusal mode.
    ///
    /// THE single definition. Everything that acts on this mode must go through
    /// here rather than reading `security_unrestricted` directly, so the
    /// developer gate cannot be bypassed by a caller that forgets to check it.
    /// The raw field is a *request*; this is the answer.
    ///
    /// The argument is `&self` rather than a global so tests can exercise every
    /// combination of flag and key without touching the real state directory.
    pub fn security_unrestricted_active(&self) -> bool {
        crate::unlock::unrestricted_active(
            self.security_unrestricted,
            &self.security_dev_public_key,
        )
    }

    /// The gate state, for display.
    ///
    /// More informative than the boolean, because "off because locked" and "off
    /// because no developer key is configured" are different situations with
    /// different remedies, and the TUI has to tell the user which one they are in.
    pub fn security_gate_state(&self) -> crate::unlock::GateState {
        crate::unlock::gate_state(self.security_unrestricted, &self.security_dev_public_key)
    }
}

fn default_num_ctx() -> u32 {
    16384
}

fn default_security_num_ctx() -> u32 {
    8192
}

/// Default home for security-tier scripts: `~/Documents/luna-scripts`.
///
/// `XDG_DOCUMENTS_DIR` is honoured when set, because on Arch the user may well
/// have moved Documents (a symlink into another partition is common) and a
/// hard-coded path would then write somewhere the user does not look.
pub fn default_security_scripts_dir() -> String {
    if let Ok(docs) = std::env::var("XDG_DOCUMENTS_DIR") {
        let docs = docs.trim();
        if !docs.is_empty() {
            return format!("{}/luna-scripts", docs.trim_end_matches('/'));
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/netrunner".to_string());
    format!("{}/Documents/luna-scripts", home.trim_end_matches('/'))
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: "qwen2.5:7b-instruct".into(),
            temperature: 0.7,
            max_tokens: 2048,
            enable_thinking: true,
            fast_model: None,
            deep_model: None,
            security_model: None,
            security_model_abliterated: None,
            security_retry_on_refusal: true,
            security_num_ctx: default_security_num_ctx(),
            security_unrestricted: false,
            security_scripts_dir: default_security_scripts_dir(),
            security_dev_public_key: String::new(),
            // Loopback-only by default; see `scan_allowlist`.
            scan_allowlist: Vec::new(),
            embedding_model: "nomic-embed-text".into(),
            num_ctx: default_num_ctx(),
        }
    }
}

// ── Voice output settings ─────────────────────────────────────────────────────
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMode {
    #[default]
    Basic,
    Jinx,
    Off,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct VoiceConfig {
    pub mode: VoiceMode,
    pub piper_bin: PathBuf,
    pub piper_model: PathBuf,
    #[serde(default = "default_whisper_model")]
    pub whisper_model: PathBuf,
    #[serde(default = "default_rvc_model")]
    pub rvc_model: Option<PathBuf>,
    #[serde(default = "default_rvc_script")]
    pub rvc_script: Option<PathBuf>,
}

fn default_whisper_model() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/root"))
        .join(".local/share/luna/models/ggml-small.en.bin")
}

fn default_rvc_model() -> Option<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
    Some(home.join(".local/share/luna/voices/Jinx.pth"))
}

fn default_rvc_script() -> Option<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
    Some(home.join(".local/share/luna/scripts/rvc_infer.py"))
}

impl Default for VoiceConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
        Self {
            mode: VoiceMode::Basic,
            piper_bin: PathBuf::from("/usr/bin/piper"),
            piper_model: home.join(".local/share/luna/voices/basic.onnx"),
            whisper_model: default_whisper_model(),
            rvc_model: default_rvc_model(),
            rvc_script: default_rvc_script(),
        }
    }
}

// ── Audio input settings ──────────────────────────────────────────────────────
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    PushToTalk,
    WakeWord,
    #[default]
    Both,
    Tui,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AudioConfig {
    pub input_mode: InputMode,
    pub ptt_key: String,
    pub wake_word: String,
    pub wake_aliases: Vec<String>,
    pub vad_silence_ms: u64,
    pub sample_rate: u32,
    /// After a wake-word activation, keep listening for voice without
    /// requiring "luna" again for this many minutes (0 = off).
    pub conversation_timeout_mins: u32,
    /// After a wake-word response in TUI mode, keep listening without
    /// the wake word for this many seconds (0 = off).
    pub conversation_window_secs: u64,
    /// "Voice mode" (hands-free) auto-returns to wake-word listening after
    /// this many minutes of silence (0 = never).
    pub voice_mode_idle_mins: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_mode: InputMode::Both,
            ptt_key: "ControlLeft".into(),
            wake_word: "hey luna".into(),
            wake_aliases: vec![
                "luna".into(),
                "hey luna".into(),
                "hay luna".into(),
                "hello luna".into(),
                "hello lana".into(),
                "hey lana".into(),
                "hi luna".into(),
                "hi lana".into(),
            ],
            vad_silence_ms: 2000,
            sample_rate: 16000,
            conversation_timeout_mins: 5,
            conversation_window_secs: 15,
            voice_mode_idle_mins: 5,
        }
    }
}

// ── Wake-word daemon settings ─────────────────────────────────────────────────
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct WakeConfig {
    /// Run `luna --wake-daemon` (systemd: luna-wake.service). When false the
    /// daemon exits immediately.
    pub enabled: bool,
    /// How to start Luna on "hey luna":
    ///   * "headless" — in-process voice session, no windows (final product)
    ///   * "tui"      — spawn a terminal running the full TUI (debugging)
    pub launch_mode: String,
    /// Terminal emulator to launch for launch_mode = "tui". Empty = auto-detect
    /// (ghostty, kitty, alacritty, foot, konsole, in that order).
    pub terminal_cmd: String,
    /// Draw the "she's listening" animation on the desktop (Wayland only).
    pub overlay_enabled: bool,
    /// Debounce between activations (seconds) — leftover audio must not
    /// instantly re-trigger after a session ends.
    pub min_interval_secs: u64,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            launch_mode: "headless".into(),
            terminal_cmd: String::new(),
            overlay_enabled: true,
            min_interval_secs: 15,
        }
    }
}

// ── Memory settings ───────────────────────────────────────────────────────────
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub context_window: usize,
    pub history_path: PathBuf,
    /// Every N user turns, Luna reviews the recent conversation for things to
    /// remember (preferences, details, expectations) and repeatable procedures
    /// worth saving as skills — the "memory nudge" (Hermes-style self-learning).
    #[serde(default = "default_nudge_interval")]
    pub nudge_interval: u32,
}

fn default_nudge_interval() -> u32 {
    10
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna");
        Self {
            context_window: 20,
            history_path: data_dir.join("history.json"),
            nudge_interval: default_nudge_interval(),
        }
    }
}

// ── Todoist integration ───────────────────────────────────────────────────────
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TodoistConfig {
    /// Todoist API token — get yours at todoist.com/app/settings/integrations
    /// Leave unset to disable Todoist tools.
    pub api_token: Option<String>,
}

// ── Spotify integration ───────────────────────────────────────────────────────
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SpotifyConfig {
    /// Spotify OAuth client ID — create an app at developer.spotify.com
    /// Store it with `luna --set-key spotify_id` and reference it as
    /// `keyring:spotify_id`. Leave unset to disable Spotify tools.
    pub client_id: Option<String>,
    /// Optional. The OAuth flow uses PKCE and does NOT need a client secret;
    /// kept only for backward compatibility with older configs.
    pub client_secret: Option<String>,
    /// The refresh token is stored in the OS keyring (`luna --spotify-auth`
    /// writes it) — never put it in luna.toml.
    pub refresh_token: Option<String>,
}

// ── Proactive background monitoring ───────────────────────────────────────────
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProactiveConfig {
    /// Master switch — set false to disable all background checks
    pub enabled: bool,
    /// How often to check, in minutes
    pub check_interval_mins: u64,
    /// Notify when battery drops to/below this percent while discharging
    pub battery_low_threshold: u32,
    /// Notify when disk usage on / reaches this percent
    pub disk_full_threshold: u32,
    /// Notify when pacman updates are available (requires pacman-contrib)
    pub check_updates: bool,
}

impl Default for ProactiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_mins: 15,
            battery_low_threshold: 20,
            disk_full_threshold: 90,
            check_updates: true,
        }
    }
}

// ── Logging ───────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoggingConfig {
    /// Log level: "info", "debug", or "trace".  Toggle via the set_debug tool.
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
        }
    }
}

// ── Web search ────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SearchConfig {
    /// Tavily API key (optional) — free tier: 1000 searches/month
    /// Without a key, Tavily keyless works automatically (rate-limited).
    /// Get yours at https://tavily.com
    pub tavily_api_key: Option<String>,

    /// Gemini API key (optional) — used as knowledge fallback
    /// When Tavily has no results, Gemini answers from its training data.
    /// Get yours at https://aistudio.google.com/apikey
    pub gemini_api_key: Option<String>,
}

// ── Background daemon (`luna --daemon`) ───────────────────────────────────────
// Container-level serde default: configs written by older Luna versions
// (missing newer keys) still parse, falling back to these defaults.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DaemonConfig {
    /// Master switch — `luna --daemon` exits immediately when false
    pub enabled: bool,
    /// How often to scan processes and check disk, in minutes
    pub check_interval_mins: u64,
    // ── Process watchdog ──
    /// Flag a single process using more RAM than this (MB)
    pub ram_threshold_mb: u64,
    /// Flag a single process with sustained CPU use above this (%)
    pub cpu_threshold_percent: f32,
    /// Processes never flagged (Luna's own stack, desktop shell, etc.)
    #[serde(default)]
    pub ignore_processes: Vec<String>,
    // ── Disk cleanup ──
    /// Enable disk hygiene checks
    pub disk_cleanup: bool,
    /// "notify" = report reclaimable space, touch nothing.
    /// "auto"   = clean safe locations when / crosses the proactive
    ///            disk_full_threshold.
    pub cleanup_mode: String,
    /// Delete ~/.cache files untouched for this many days
    pub cache_max_age_days: u32,
    /// Empty trash items older than this many days
    pub trash_max_age_days: u32,
    /// journalctl --vacuum-time for system logs (needs sudo)
    pub journal_vacuum_days: u32,
    /// Keep this many package versions in the pacman cache (needs sudo)
    pub pacman_cache_keep: u32,
    /// Auto-remove orphaned packages (pacman -Qdtq) once they've been
    /// detected for this many days. 0 = never remove — only notify.
    pub orphan_cleanup_days: u32,
    /// Orphaned packages to never auto-remove (exact package names)
    #[serde(default)]
    pub orphan_keep: Vec<String>,
    /// Only notify about cleanups worth at least this much (MB)
    pub min_notify_mb: u64,
    // ── Process usage learning ──
    /// Track per-process usage to ~/.local/share/luna/process_stats.json.
    /// Learned daily-use processes are dynamically excluded from watchdog
    /// notifications; idle non-daily processes become auto-kill candidates.
    pub learning_enabled: bool,
    /// A process seen on this many of the trailing 14 days counts as daily use
    pub daily_use_days_per_week: u32,
    /// Allowlisted processes get SIGTERM after being idle this many minutes
    pub idle_kill_minutes: u32,
    /// Daily "Luna — system check" self-audit: disk, reclaimable space,
    /// updates, orphans, backup status and RAM hogs, on her own initiative.
    /// 0 = off.
    pub system_digest_days: u32,
    /// Non-allowlisted processes idle longer than this trigger an opt-in suggestion
    pub suggest_autokill_after_mins: u32,
    /// Autonomous idle stop: any non-protected, non-daily-use process idle
    /// this long gets stopped on Luna's own, announcing intent first via
    /// notify-send (full list, longest idle first), then one "done" summary.
    /// 0 = never stop autonomously (legacy opt-in-only behavior).
    pub auto_stop_after_mins: u32,
    /// Stateful apps never auto-killed even when allowlisted
    #[serde(default)]
    pub protected_processes: Vec<String>,
    /// How often (days) Luna re-analyzes fish history into permanent memory
    pub history_learn_days: u32,
    /// How often (days) Luna re-indexes projects/scripts/configs (0 = off)
    pub index_learn_days: u32,
    /// Every N days Luna runs the safety check (pacman -Syu, ClamAV, rkhunter,
    /// UFW, Lynis, monthly AIDE, backup). 0 = off. The script light/full scan
    /// logic applies. Runs detached so the watchdog never stalls.
    pub safety_check_days: u32,
    /// Every N days Luna backs up the home directory to /dev/sda1 at
    /// /mnt/backup/arch-backup/ (only while the drive is connected). 0 = off.
    pub backup_days: u32,
    /// Send an "I'm alive" desktop notification every N hours while the
    /// daemon runs (0 = off). The notification includes uptime, cycles,
    /// auto-kills and how much Luna knows.
    pub notify_hours: u32,
    /// Keep the desktop notification corner bounded: when more than this many
    /// notifications are stored, clear them via the shell (0 = never auto-clear).
    pub notif_cap: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_mins: 30,
            ram_threshold_mb: 1500,
            cpu_threshold_percent: 80.0,
            ignore_processes: [
                "ollama",
                "luna",
                "plasmashell",
                "kwin_wayland",
                "gnome-shell",
                "Xwayland",
                "pipewire",
                "pipewire-pulse",
                "wireplumber",
                "systemd",
                "dbus-daemon",
                "dbus-broker",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            disk_cleanup: true,
            cleanup_mode: "notify".into(),
            cache_max_age_days: 30,
            trash_max_age_days: 14,
            journal_vacuum_days: 30,
            pacman_cache_keep: 2,
            orphan_cleanup_days: 7,
            orphan_keep: Vec::new(),
            min_notify_mb: 500,
            learning_enabled: true,
            daily_use_days_per_week: 5,
            idle_kill_minutes: 30,
            suggest_autokill_after_mins: 45,
            auto_stop_after_mins: 120,
            system_digest_days: 1,
            protected_processes: [
                // GUI apps that hold unsaved user state — NEVER auto-killed
                "firefox",
                "zen-browser",
                "chromium",
                "code",
                "zed",
                "kitty",
                "alacritty",
                "konsole",
                "foot",
                "ghostty",
                "obs",
                "gimp",
                "krita",
                "blender",
                "libreoffice",
                "soffice",
                "thunderbird",
                "discord",
                "telegram-desktop",
                "slack",
                // OnlyOffice — byte-exact /proc/<pid>/comm names (camelCase matters)
                "DesktopEditors",
                "editors_helper",
                // terminal editors that hold unsaved buffers
                "micro",
                "helix",
                "hx",
                "nvim",
                "vim",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            history_learn_days: 30,
            index_learn_days: 30,
            safety_check_days: 7,
            backup_days: 7,
            notify_hours: 0,
            notif_cap: 100,
        }
    }
}

// ── WhatsApp bridge (luna-whapp) ──────────────────────────────────────────────
// Points Luna at the local Baileys bridge's HTTP API. The bridge (a standalone
// Node service) holds the linked session; Luna only POSTs to localhost.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct WhatsAppConfig {
    /// Whether the whatsapp_send tool is usable at all
    pub enabled: bool,
    /// Endpoint of the local bridge, e.g. http://127.0.0.1:7373
    pub base_url: String,
}

impl Default for WhatsAppConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "http://127.0.0.1:7373".into(),
        }
    }
}

// ── sysmode hardening-profile switcher ───────────────────────────────────────
// Points Luna at the user's `sysmode` CLI (typically /usr/local/bin/sysmode)
// so Luna can switch profiles, pull attack logs, and run a functional self-test
// of the recon-deceiver, IDS, honeypot and decoy components.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct SysmodeConfig {
    /// Whether the sysmode tool is available
    pub enabled: bool,
    /// Path / command name of the sysmode script
    pub bin: String,
}

impl Default for SysmodeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bin: "sysmode".into(),
        }
    }
}

// ── Guarded package updates ───────────────────────────────────────────────────
// The apply gate is a HUMAN decision, not a model decision.
//
// Observed in the wild: asked only to "check" for updates, Luna checked, saw
// that 6 of 33 pending updates were breaking (systemd 261 -> 262), and then
// four seconds later called apply with confirm_breaking=true — unprompted, with
// no human in the loop. Nothing in the tool stopped her: the confirmation flag
// was a plain model-settable argument, so "the user approved" was just
// something the model asserted about itself.
//
// So the flag is gone from the tool schema entirely, and applying now requires
// this config switch, which only a human editing the file can flip.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct UpdatesConfig {
    /// Whether the system_update tool is offered at all.
    pub enabled: bool,
    /// HUMAN GATE. While false, `action=apply` always refuses — the model
    /// cannot install packages no matter what arguments it invents. Flip this
    /// to true yourself, in this file, when you actually want an upgrade to
    /// happen. Then put it back.
    pub allow_apply: bool,
}

impl Default for UpdatesConfig {
    fn default() -> Self {
        Self { enabled: true, allow_apply: false }
    }
}

// ── Gated self-modification ───────────────────────────────────────────────────
// Luna can propose changes to her own source, but nothing reaches the real tree
// until the change passes `cargo test` against a scratch copy AND the user
// approves. The protected list is the safety/identity core: the files that
// define her constitution, her routing, and the guards that keep her in bounds.
// She may improve everything else; she may not edit her own rules.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct SelfPatchConfig {
    /// Master switch. When false the self_patch tool refuses to do anything.
    pub enabled: bool,
    /// Root of her own source tree.
    pub source_dir: String,
    /// Whether a *validated* proposal may actually be installed. Set false for
    /// a read-only "she may propose but never apply" posture.
    pub allow_apply: bool,
    /// Files/directories self-modification may never touch.
    pub protected_files: Vec<String>,
}

impl Default for SelfPatchConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/netrunner".into());
        Self {
            enabled: true,
            source_dir: format!("{home}/Projects/luna-stable"),
            allow_apply: true,
            protected_files: [
                // Her constitution and identity, config parsing, dependencies.
                "src/config.rs",
                "Cargo.toml",
                "Cargo.lock",
                // Her guardrails: model routing + the freeform tool-call parser.
                "src/llm/escalation.rs",
                "src/llm/react.rs",
                // The self-modification gate and the other gated/dangerous tools.
                "src/tools/selfpatch.rs",
                "src/tools/safety.rs",
                "src/tools/pkgupdate.rs",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

// ── Project-Vision browser automation ─────────────────────────────────────────
// Luna drives the user's Project-Vision (SIH) browser server + Chromium over
// the DevTools Protocol, so a natural-language goal becomes real browsing.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct BrowserConfig {
    /// Whether the browser_do tool is available at all
    pub enabled: bool,
    /// Browser binary used for automation (chromium, google-chrome, ...)
    pub chromium: String,
    /// Port Chromium's remote debugging listens on
    pub cdp_port: u16,
    /// Private profile dir for the automation browser (keeps logins). Empty =
    /// ~/.local/share/luna/browser-profile
    pub profile_dir: String,
    /// Run Chromium headless (invisible). Default visible so the user can watch.
    pub headless: bool,
    /// Window size "WxH" for the automation browser (e.g. "1920x1200"). Empty =
    /// auto-detect the primary screen so the window fills the display.
    pub window_size: String,
    /// Hard cap on how long one browser task may run
    pub timeout_secs: u64,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chromium: "chromium".into(),
            cdp_port: 9222,
            profile_dir: String::new(),
            headless: false,
            window_size: String::new(),
            timeout_secs: 600,
        }
    }
}

// ── Desktop computer-use automation ──────────────────────────────────────────
// The "bigger than SIH" mode: Luna stares at the whole desktop (grim screenshot),
// asks the SIH VLM for the next action, and executes it with ydotool — so it can
// drive ANY application, not just a browser. No extension involved.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DesktopConfig {
    /// Whether the desktop_do tool is available at all
    pub enabled: bool,
    /// HTTP base of the Project-Vision server (its /desktop-act endpoint)
    pub srijan_url: String,
    /// Folder holding the server's main.py; if empty, Luna doesn't auto-start
    pub server_dir: String,
    /// Screenshot tool (grim for Wayland, import for X11, etc.)
    pub screenshot_cmd: String,
    /// Path to the ydotool binary (empty = auto-detect from PATH)
    pub ydotool_bin: String,
    /// Hard cap on actions per task (safety valve so it never spins forever)
    pub max_steps: u32,
    /// Hard cap on how long one task may run
    pub timeout_secs: u64,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            srijan_url: "http://127.0.0.1:8001".into(),
            server_dir: String::new(),
            screenshot_cmd: "grim".into(),
            ydotool_bin: String::new(),
            max_steps: 30,
            timeout_secs: 300,
        }
    }
}

// ── Vision ("eyes") ───────────────────────────────────────────────────────────
// Pure-Rust eyes: screenshots (grim for the desktop, CDP for the browser) are
// described by a small local VLM served by the same Ollama the planner uses.
// No Python, no external server. The model is loaded on demand and swaps with
// the planner in VRAM; use a small model (qwen2.5vl:3b) so it fits the 6GB card.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionConfig {
    /// Whether the see tool is available at all
    pub enabled: bool,
    /// Ollama vision model used to describe screenshots
    pub model: String,
    /// Screenshot tool for the desktop (grim on Wayland, import on X11, ...)
    pub screenshot_cmd: String,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "qwen2.5vl:3b".into(),
            screenshot_cmd: "grim".into(),
        }
    }
}

// ── Loading logic ─────────────────────────────────────────────────────────────
impl LunaConfig {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if !config_path.exists() {
            tracing::info!("No config found, creating defaults at {:?}", config_path);
            let config = LunaConfig::default();
            config.save().context("Failed to save default config")?;
            return Ok(config);
        }

        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config at {:?}", config_path))?;

        let mut config: LunaConfig =
            toml::from_str(&raw).context("Failed to parse luna.toml — check for syntax errors")?;

        config.resolve_secrets();

        if config.voice.rvc_model.is_none() {
            config.voice.rvc_model = default_rvc_model();
        }
        if config.voice.rvc_script.is_none() {
            config.voice.rvc_script = default_rvc_script();
        }

        tracing::info!("Config loaded from {:?}", config_path);
        Ok(config)
    }

    /// Replace "keyring:name" references with secrets fetched from the
    /// OS keyring. Plain values pass through untouched.
    fn resolve_secrets(&mut self) {
        self.search.tavily_api_key = resolve_secret_ref(self.search.tavily_api_key.take());
        self.search.gemini_api_key = resolve_secret_ref(self.search.gemini_api_key.take());
        self.todoist.api_token = resolve_secret_ref(self.todoist.api_token.take());
        self.spotify.client_id = resolve_secret_ref(self.spotify.client_id.take());
        self.spotify.client_secret = resolve_secret_ref(self.spotify.client_secret.take());
        self.spotify.refresh_token = resolve_secret_ref(self.spotify.refresh_token.take());
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        // Rewrite via toml_edit so user comments and formatting survive
        // (e.g. "turn on debug mode" only flips one leaf value).
        let mut doc: toml_edit::DocumentMut = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let as_value = toml::Value::try_from(self).context("Failed to convert config")?;
        merge_toml_value(doc.as_table_mut(), &as_value);

        std::fs::write(&path, doc.to_string())
            .with_context(|| format!("Failed to write config to {:?}", path))?;

        // Config can contain secrets — only the owner may read it
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&path, perms)
                .context("Failed to restrict config file permissions")?;
        }

        Self::restart_daemons_if_running();
        Ok(())
    }

    /// Restart active Luna daemons (luna-daemon.service and luna-wake.service)
    /// via systemd user services when configuration changes.
    pub fn restart_daemons_if_running() {
        std::thread::spawn(|| {
            for service in ["luna-daemon.service", "luna-wake.service"] {
                let is_active = std::process::Command::new("systemctl")
                    .args(["--user", "is-active", service])
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active")
                    .unwrap_or(false);

                if is_active {
                    tracing::info!("Restarting {} after config update...", service);
                    let _ = std::process::Command::new("systemctl")
                        .args(["--user", "restart", service])
                        .status();
                }
            }
        });
    }

    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna")
            .join("luna.toml")
    }
}

/// Recursively write `value` into an existing toml_edit table, updating
/// leaves in place and inserting keys that don't exist yet. Comments and
/// formatting of untouched lines are preserved.
fn merge_toml_value(dest: &mut toml_edit::Table, value: &toml::Value) {
    let toml::Value::Table(map) = value else {
        return;
    };
    for (k, v) in map {
        if let toml::Value::Table(child_map) = v {
            if !dest.contains_key(k) {
                dest.insert(k, toml_edit::Item::Table(toml_edit::Table::new()));
            }
            if let Some(child) = dest[k].as_table_mut() {
                merge_toml_value(child, &toml::Value::Table(child_map.clone()));
            }
        } else {
            dest[k] = leaf_item(v);
        }
    }
}

/// Convert a non-table toml::Value into a toml_edit item.
fn leaf_item(v: &toml::Value) -> toml_edit::Item {
    let val: toml_edit::Value = match v {
        toml::Value::String(s) => s.as_str().into(),
        toml::Value::Integer(i) => (*i).into(),
        toml::Value::Float(f) => (*f).into(),
        toml::Value::Boolean(b) => (*b).into(),
        toml::Value::Datetime(d) => d.to_string().as_str().into(),
        toml::Value::Array(arr) => {
            let mut out = toml_edit::Array::new();
            for el in arr {
                match el {
                    toml::Value::String(s) => out.push(s.as_str()),
                    toml::Value::Integer(i) => out.push(*i),
                    toml::Value::Float(f) => out.push(*f),
                    toml::Value::Boolean(b) => out.push(*b),
                    _ => {}
                }
            }
            toml_edit::Value::from(out)
        }
        toml::Value::Table(_) => unreachable!("tables handled by merge_toml_value"),
    };
    toml_edit::Item::Value(val)
}

/// Resolve a single secret value: "keyring:name" fetches from the OS
/// keyring (service "luna", user "name"); anything else passes through.
fn resolve_secret_ref(value: Option<String>) -> Option<String> {
    let value = value?;
    let Some(name) = value.strip_prefix("keyring:") else {
        return Some(value);
    };
    match keyring::Entry::new("luna", name) {
        Ok(entry) => match entry.get_password() {
            Ok(secret) => Some(secret),
            Err(e) => {
                tracing::warn!(
                    "Keyring entry 'luna/{}' unavailable ({}). \
                     Store it with: luna --set-key {}",
                    name,
                    e,
                    name
                );
                None
            }
        },
        Err(e) => {
            tracing::warn!("Keyring unavailable for 'luna/{}': {}", name, e);
            None
        }
    }
}

/// Store a secret in the OS keyring under service "luna".
pub fn keyring_set(name: &str, secret: &str) -> Result<()> {
    let entry = keyring::Entry::new("luna", name)
        .with_context(|| format!("Cannot access keyring for 'luna/{}'", name))?;
    entry
        .set_password(secret)
        .with_context(|| format!("Failed to store 'luna/{}' in keyring", name))?;
    Ok(())
}

/// Read a secret from the OS keyring (for --get-key verification).
pub fn keyring_get(name: &str) -> Result<String> {
    let entry = keyring::Entry::new("luna", name)
        .with_context(|| format!("Cannot access keyring for 'luna/{}'", name))?;
    entry
        .get_password()
        .with_context(|| format!("No keyring entry 'luna/{}'", name))
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config written before the security tier existed must still load, with
    /// the tier simply inactive. `security_model: None` means `build_security_
    /// client` returns None and `classify`'s Security arm falls back to the
    /// general model — degraded, not broken.
    #[test]
    fn a_config_without_the_security_tier_still_parses() {
        let cfg: LlmConfig = toml::from_str(
            r#"
base_url = "http://localhost:11434"
model = "qwen2.5:7b-instruct"
temperature = 0.7
max_tokens = 2048
enable_thinking = true
embedding_model = "nomic-embed-text"
"#,
        )
        .expect("a pre-security-tier config must still load");
        assert_eq!(cfg.security_model, None);
        // Falls back to the general window rather than inheriting 16384, which
        // would put the KV cache of an 8B model on a 6 GB card entirely in RAM.
        assert_eq!(cfg.security_num_ctx, 8192);
    }

    /// The security tier has its own context budget because it is the only tier
    /// whose weights plus KV cache can exceed VRAM. Silently giving it the
    /// global 16384 is the exact regression the separate field exists to stop.
    #[test]
    fn the_security_tier_defaults_to_its_own_smaller_window() {
        let cfg = LlmConfig::default();
        assert_eq!(cfg.num_ctx, 16384);
        assert!(
            cfg.security_num_ctx < cfg.num_ctx,
            "security window ({}) must be smaller than the general one ({}): \
             an 8B model at 16k on a 6 GB card spills the whole KV cache to RAM",
            cfg.security_num_ctx,
            cfg.num_ctx
        );
    }

    /// Round-trips the new keys through TOML so a typo in a hand-written config
    /// is a load error rather than a silently ignored line.
    #[test]
    fn the_security_tier_keys_round_trip() {
        let src = r#"
base_url = "http://localhost:11434"
model = "qwen2.5:7b-instruct"
temperature = 0.7
max_tokens = 2048
enable_thinking = true
deep_model = "qwen2.5-coder:14b"
security_model = "whiterabbitneo-coder-tools"
security_num_ctx = 8192
embedding_model = "nomic-embed-text"
num_ctx = 16384
"#;
        let cfg: LlmConfig = toml::from_str(src).unwrap();
        assert_eq!(cfg.security_model.as_deref(), Some("whiterabbitneo-coder-tools"));
        assert_eq!(cfg.security_num_ctx, 8192);
    }

    /// Regression guard for a bug that failed *silently*.
    ///
    /// `num_ctx` was hardcoded to 8192. The 9-clause Constitution plus the
    /// 47-tool schema is ~8.0k tokens before Luna sees a user message, and a
    /// `files` + `read` round-trip measured 9326. Ollama does not error when a
    /// prompt overflows the window — it context-shifts, and we measured a real
    /// request silently dropping 8156 -> 4098 tokens. Luna then lost the file
    /// she had just read and started inventing paths.
    ///
    /// Nothing about that failure looks like a context problem from the
    /// outside, so it needs a test that fails the build instead.
    #[test]
    fn default_num_ctx_fits_the_system_prompt_and_tool_schema() {
        let tools = crate::tools::tool_definitions();
        let tools_chars = serde_json::to_string(&tools).unwrap().len();
        let prompt_chars = AgentConfig::default().system_prompt.len() + tools_chars;

        // ~4 chars/token is a deliberately conservative estimate: under-counting
        // tokens here would make this test pass while the real prompt overflows.
        let est_tokens = (prompt_chars / 4) as u32;

        assert!(
            est_tokens > 6_000,
            "prompt estimate collapsed to {est_tokens} tokens — the measurement \
             this guard is based on is stale, re-measure before trusting it"
        );

        let headroom = default_num_ctx().checked_sub(est_tokens).expect(
            "num_ctx must exceed the static prompt, otherwise every request \
             context-shifts and Luna loses tool schemas mid-task",
        );
        assert!(
            headroom >= 4_096,
            "num_ctx {} leaves only {headroom} tokens for the conversation \
             after a {est_tokens}-token prompt; measured need was 4096",
            default_num_ctx()
        );
    }

    /// The fast tier is the one client that legitimately runs on a small
    /// window — it never receives the tool schema. Guard that it stays small,
    /// because a 16k KV cache on the 3B would waste RAM for no benefit.
    #[test]
    fn fast_tier_window_is_deliberately_small() {
        // Mirrors the literal in agent::build_fast_client.
        const FAST_NUM_CTX: u32 = 4096;
        assert!(FAST_NUM_CTX < default_num_ctx());
    }

    #[test]
    fn save_merge_preserves_comments_and_updates_values() {
        let existing =
            "# top comment\n[logging]\n# keep me\nlevel = \"info\"\n\n[llm]\nmodel = \"old\"\n";
        let mut doc: toml_edit::DocumentMut = existing.parse().unwrap();

        let new_val: toml::Value = toml::from_str(
            "[logging]\nlevel = \"debug\"\n\n[llm]\nmodel = \"new\"\nfast_model = \"f\"\n",
        )
        .unwrap();
        merge_toml_value(doc.as_table_mut(), &new_val);

        let out = doc.to_string();
        assert!(out.contains("# keep me"), "comment lost:\n{}", out);
        assert!(out.contains("level = \"debug\""));
        assert!(out.contains("model = \"new\""));
        assert!(out.contains("fast_model = \"f\""));
    }
}
