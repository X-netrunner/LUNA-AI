//! stdio.rs — `luna --stdio [--history FILE]`: Luna as a child process of another program (Halcyon's Luna window).
//!
//! One JSON object per line in each direction, on stdout and stdin only (logs stay on stderr).
//!
//! In:   {"type":"turn","text":"..."}   one question; the reply comes back as one `reply` event
//!       {"type":"clear"}               forget the conversation held in the history file
//!       {"type":"quit"}                exit (end of input does the same)
//!       {"type":"settings_get"}        the TUI's settings, as data: categories of toggles / values / the two signed gates, the keys
//!       {"type":"setting_set","key":"external.allow_halcyon_actions","value":true}    one setting (the key names are the TUI's)
//!       {"type":"models"}              what Ollama has, and which model fills which role
//!       {"type":"model_set","name":"qwen3:8b","role":"model"}   role: model | fast | deep | security (the TUI's Primary Model, and /model)
//!       {"type":"key_set","name":"tavily","value":"..."}         an API key: stored in the OS keyring, luna.toml gets only `keyring:NAME`
//! Out:  {"type":"ready","model":"...","messages":N}
//!       {"type":"activity","kind":"step"|"tool_start"|"tool_end","name":"...","summary":"...","ok":true,"secs":1.2}
//!       {"type":"thinking","delta":"..."}                 the model's reasoning, as it streams
//!       {"type":"reply","text":"...","thinking":"..."|null,"model":"...","secs":3.4}
//!       {"type":"error","error":"..."}   {"type":"cleared"}
//!       {"type":"settings","categories":[...],"keys":[...]}   {"type":"saved","key":"...","value":...}   {"type":"models",...}
//!       {"type":"key_saved","name":"...","where":"keyring"|"file"}   (the value is never sent back)
//!
//! The turn goes through `agent::run_routed_turn`, the same fast / deep / security / full routing the TUI uses, so the
//! answer comes from the same tier it would in a terminal. `--history` points the rolling conversation at a file of the
//! caller's choosing: a caller that keeps one file per conversation can resume any of them by starting Luna with that file.
//! Tools run exactly as they do in the terminal; nothing here adds or removes a permission.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::config::LunaConfig;
use crate::memory::Memory;

/// API keys the GUI can take: (name in the keyring, luna.toml setting that refers to it, label, where to get one).
const KEYS: &[(&str, &str, &str, &str)] = &[
    ("tavily", "search.tavily_api_key", "Tavily (web search)", "Free at tavily.com: sign in, copy the key from the dashboard."),
    ("gemini", "search.gemini_api_key", "Gemini (knowledge fallback)", "Free at aistudio.google.com: Get API key."),
    ("todoist", "todoist.api_token", "Todoist", "todoist.com: Settings > Integrations > Developer > API token."),
    ("spotify_id", "spotify.client_id", "Spotify client id", "developer.spotify.com: create an app, copy its client id, then run `luna --spotify-auth` once."),
];

const ROLES: &[(&str, &str)] = &[("model", "llm.model"), ("fast", "llm.fast_model"), ("deep", "llm.deep_model"), ("security", "llm.security_model"),
    ("deep_fallback", "llm.deep_model_unrestricted"), ("security_nr", "llm.security_model_abliterated")];

fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn activity_event(ev: &crate::activity::Event) -> Value {
    use crate::activity::Event;
    match ev {
        Event::Iteration { n } => json!({"type": "activity", "kind": "step", "name": "", "summary": "", "n": n}),
        Event::ToolStart { name, summary } => {
            json!({"type": "activity", "kind": "tool_start", "name": name, "summary": summary})
        }
        Event::ToolEnd { name, ok, elapsed } => json!({
            "type": "activity", "kind": "tool_end", "name": name, "ok": ok,
            "secs": (elapsed.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        }),
        Event::Thinking { delta } => json!({"type": "thinking", "delta": delta}),
    }
}

/// What one input line asks for. Anything that is not a recognised object is ignored, so a stray line never starts a turn.
#[derive(Debug, PartialEq)]
enum Request {
    Turn(String),
    Clear,
    Quit,
    SettingsGet,
    SetSetting(String, Value),
    Models,
    SetModel(String, String),
    SetKey(String, String),
    Speak(String),
    VoiceStatus,
    Ignore,
}

fn parse_request(line: &str) -> Request {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Request::Ignore;
    };
    match v.get("type").and_then(Value::as_str) {
        Some("turn") => match v.get("text").and_then(Value::as_str).map(str::trim) {
            Some(t) if !t.is_empty() => Request::Turn(t.to_string()),
            _ => Request::Ignore,
        },
        Some("clear") => Request::Clear,
        Some("quit") => Request::Quit,
        Some("settings_get") => Request::SettingsGet,
        Some("setting_set") => match (v.get("key").and_then(Value::as_str), v.get("value")) {
            (Some(k), Some(val)) => Request::SetSetting(k.to_string(), val.clone()),
            _ => Request::Ignore,
        },
        Some("models") => Request::Models,
        Some("model_set") => match v.get("name").and_then(Value::as_str).map(str::trim) {
            Some(n) if !n.is_empty() => Request::SetModel(n.to_string(), v.get("role").and_then(Value::as_str).unwrap_or("model").to_string()),
            _ => Request::Ignore,
        },
        Some("key_set") => match (v.get("name").and_then(Value::as_str), v.get("value").and_then(Value::as_str)) {
            (Some(n), Some(val)) => Request::SetKey(n.to_string(), val.to_string()),
            _ => Request::Ignore,
        },
        Some("speak") => match v.get("text").and_then(Value::as_str).map(str::trim) {
            Some(t) if !t.is_empty() => Request::Speak(t.chars().take(4000).collect()),
            _ => Request::Ignore,
        },
        Some("voice_status") => Request::VoiceStatus,
        _ => Request::Ignore,
    }
}

/// What the configured voice still needs on this machine (empty = it can speak). Looked at, never downloaded.
pub fn voice_missing(config: &LunaConfig) -> Vec<String> {
    let mut out = Vec::new();
    if config.voice.mode == crate::config::VoiceMode::Off {
        return out;
    }
    let model = config.voice.piper_model.to_string_lossy().into_owned();
    if !config.voice.piper_model.exists() {
        out.push("the Kokoro voice model (kokoro-v1.0.onnx)".to_string());
    }
    let voices = std::path::PathBuf::from(model.replace("kokoro-v1.0.onnx", "voices-v1.0.bin"));
    if voices == config.voice.piper_model || !voices.exists() {
        out.push("the voice file (voices-v1.0.bin)".to_string());
    }
    let home = dirs::home_dir().unwrap_or_default();
    let have_env = home.join(".local/share/luna/tts_env/bin/python3").exists() || home.join(".local/share/luna/rvc_env/bin/python3").exists();
    if !have_env {
        let ok = std::process::Command::new("python3")
            .args(["-c", "import kokoro_onnx, soundfile"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            out.push("the Python packages kokoro-onnx and soundfile".to_string());
        }
    }
    out
}


// ── settings, models, keys ───────────────────────────────────────────────────

fn key_in_store(name: &str) -> bool {
    crate::config::keyring_get(name).map(|v| !v.trim().is_empty()).unwrap_or(false)
}

/// The TUI's settings as data. Same categories, labels and key names as the TUI's menu; the two signed switches report their real
/// gate state (they cannot be flipped from here or from the TUI: they need the developer key).
pub fn settings_json(config: &LunaConfig) -> Value {
    use crate::tui::app::{SettingType, SettingsMenuState};
    let sm = SettingsMenuState::load(config.clone());
    let cats: Vec<Value> = sm
        .categories
        .iter()
        .map(|c| {
            let items: Vec<Value> = c
                .items
                .iter()
                .map(|i| {
                    let signed = matches!(i.key, "llm.security_unrestricted" | "external.allow_capability_actions");
                    match &i.setting_type {
                        SettingType::Toggle(b) => json!({"key": i.key, "label": i.label, "desc": i.description, "kind": "toggle", "value": b}),
                        SettingType::Value(v) if signed => json!({"key": i.key, "label": i.label, "desc": i.description, "kind": "gate", "value": v}),
                        SettingType::Value(v) => {
                            let opts = if i.key == "voice.mode" { json!(["basic", "jinx", "off"]) } else { Value::Null };
                            json!({"key": i.key, "label": i.label, "desc": i.description, "kind": "value", "value": v.to_lowercase(), "options": opts})
                        }
                    }
                })
                .collect();
            json!({"name": c.name, "items": items})
        })
        .collect();
    let keys: Vec<Value> = KEYS
        .iter()
        .map(|(name, _, label, help)| json!({"name": name, "label": label, "help": help, "set": key_in_store(name)}))
        .collect();
    json!({"type": "settings", "categories": cats, "keys": keys,
           "models": {"model": config.llm.model, "fast": config.llm.fast_model, "deep": config.llm.deep_model, "security": config.llm.security_model,
                      "deep_fallback": config.llm.deep_model_unrestricted, "security_nr": config.llm.security_model_abliterated},
           "auto": config.llm.auto_route})
}

/// Set one TUI setting by its key. Toggles are set to the wanted value (not flipped); the two signed gates are refused with the
/// reason. Only that one value is written to luna.toml (see `LunaConfig::set_leaf`), and the in-memory config follows.
pub fn apply_setting(config: &mut LunaConfig, key: &str, value: &Value) -> anyhow::Result<()> {
    use crate::tui::app::{SettingType, SettingsMenuState};
    if matches!(key, "llm.security_unrestricted" | "external.allow_capability_actions") {
        anyhow::bail!("This switch needs the developer key and cannot be changed from here: run `luna --unlock-security` (or `--lock-security`).");
    }
    let mut sm = SettingsMenuState::load(config.clone());
    let found = sm.categories.iter().enumerate().find_map(|(ci, c)| c.items.iter().position(|i| i.key == key).map(|ii| (ci, ii)));
    let (ci, ii) = found.ok_or_else(|| anyhow::anyhow!("no such setting: {key}"))?;
    match (&sm.categories[ci].items[ii].setting_type.clone(), value) {
        (SettingType::Toggle(cur), Value::Bool(want)) => {
            if cur != want {
                sm.cat_index = ci;
                sm.item_index = ii;
                sm.toggle_selected();
            }
            crate::config::LunaConfig::set_leaf(key, toml_edit::Value::from(*want))?;
        }
        (SettingType::Value(_), Value::String(v)) if key == "voice.mode" => {
            let v = v.trim().to_lowercase();
            if !["basic", "jinx", "off"].contains(&v.as_str()) {
                anyhow::bail!("voice.mode is basic, jinx or off");
            }
            sm.config.voice.mode = match v.as_str() {
                "jinx" => crate::config::VoiceMode::Jinx,
                "off" => crate::config::VoiceMode::Off,
                _ => crate::config::VoiceMode::Basic,
            };
            crate::config::LunaConfig::set_leaf(key, toml_edit::Value::from(v))?;
        }
        (SettingType::Value(_), Value::String(v)) if key == "llm.model" => {
            let v = v.trim();
            if v.is_empty() || v.len() > 200 {
                anyhow::bail!("not a model name");
            }
            sm.config.llm.model = v.to_string();
            crate::config::LunaConfig::set_leaf(key, toml_edit::Value::from(v))?;
        }
        _ => anyhow::bail!("{key} cannot be set to that value from here"),
    }
    *config = sm.config;
    Ok(())
}

/// The models Ollama has: (name, size in bytes). `None` when Ollama does not answer.
async fn installed_models(config: &LunaConfig) -> Option<Vec<(String, u64)>> {
    let url = format!("{}/api/tags", config.llm.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build().ok()?;
    let v: Value = client.get(url).send().await.ok()?.json().await.ok()?;
    Some(
        v["models"]
            .as_array()?
            .iter()
            .filter_map(|m| Some((m["name"].as_str()?.to_string(), m["size"].as_u64().unwrap_or(0))))
            .collect(),
    )
}

async fn models_json(config: &LunaConfig) -> Value {
    let inst = installed_models(config).await;
    json!({"type": "models", "reachable": inst.is_some(),
           "installed": inst.unwrap_or_default().into_iter().map(|(n, b)| json!({"name": n, "gb": (b as f64 / 1e8).round() / 10.0})).collect::<Vec<_>>(),
           "roles": {"model": config.llm.model, "fast": config.llm.fast_model, "deep": config.llm.deep_model, "security": config.llm.security_model,
                      "deep_fallback": config.llm.deep_model_unrestricted, "security_nr": config.llm.security_model_abliterated},
           "auto": config.llm.auto_route})
}

async fn set_model(config: &mut LunaConfig, name: &str, role: &str) -> anyhow::Result<()> {
    let leaf = ROLES.iter().find(|(r, _)| *r == role).map(|(_, l)| *l).ok_or_else(|| anyhow::anyhow!("role is model, fast, deep, security, deep_fallback or security_nr"))?;
    match installed_models(config).await {
        None => anyhow::bail!("Ollama is not answering at {}, so the model name cannot be checked. Start it first.", config.llm.base_url),
        Some(list) if !list.iter().any(|(n, _)| n == name || n.strip_suffix(":latest") == Some(name)) => {
            let near: Vec<&str> = list.iter().map(|(n, _)| n.as_str()).filter(|n| n.contains(name) || name.contains(*n)).take(4).collect();
            anyhow::bail!("Ollama does not have '{name}'.{}", if near.is_empty() { " Pull it first: ollama pull NAME".to_string() } else { format!(" Did you mean: {}?", near.join(", ")) });
        }
        Some(_) => {}
    }
    match role {
        "model" => config.llm.model = name.to_string(),
        "fast" => config.llm.fast_model = Some(name.to_string()),
        "deep" => config.llm.deep_model = Some(name.to_string()),
        "deep_fallback" => config.llm.deep_model_unrestricted = Some(name.to_string()),
        "security_nr" => config.llm.security_model_abliterated = Some(name.to_string()),
        _ => config.llm.security_model = Some(name.to_string()),
    }
    crate::config::LunaConfig::set_leaf(leaf, toml_edit::Value::from(name))
}

/// Store an API key: the secret goes to the OS keyring (or Luna's secrets file when there is no keyring), luna.toml only gets the
/// reference `keyring:NAME`. Returns where it went. The value is never logged or sent back.
fn set_key(name: &str, value: &str) -> anyhow::Result<&'static str> {
    let (_, leaf, _, _) = KEYS.iter().find(|(n, ..)| *n == name).ok_or_else(|| anyhow::anyhow!("unknown key '{name}'"))?;
    let v = value.trim();
    if v.is_empty() || v.len() > 512 || v.chars().any(|c| c.is_control()) {
        anyhow::bail!("that does not look like a key (empty, too long, or it holds a line break)");
    }
    crate::config::keyring_set(name, v)?;
    crate::config::LunaConfig::set_leaf(leaf, toml_edit::Value::from(format!("keyring:{name}")))?;
    let in_keyring = keyring::Entry::new("luna", name).and_then(|e| e.get_password()).is_ok();
    Ok(if in_keyring { "keyring" } else { "file" })
}

pub async fn run(config: &LunaConfig) -> Result<()> {
    let mut config = config.clone();
    let config = &mut config;
    let mut memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
    let _activity = crate::activity::subscribe(|ev| emit(activity_event(ev)));
    emit(json!({"type": "ready", "model": config.llm.model, "messages": memory.len()}));

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        match parse_request(&line) {
            Request::Quit => break,
            Request::Ignore => {}
            Request::Clear => {
                memory.clear()?;
                crate::llm::escalation::reset_security_latch();
                crate::recon::reset_latch();
                emit(json!({"type": "cleared"}));
            }
            Request::SettingsGet => emit(settings_json(config)),
            Request::SetSetting(key, value) => match apply_setting(config, &key, &value) {
                Ok(()) => emit(json!({"type": "saved", "key": key, "value": value})),
                Err(e) => emit(json!({"type": "error", "error": e.to_string()})),
            },
            Request::Models => emit(models_json(config).await),
            Request::SetModel(name, role) => match set_model(config, &name, &role).await {
                Ok(()) => emit(json!({"type": "saved", "key": format!("model.{role}"), "value": name})),
                Err(e) => emit(json!({"type": "error", "error": e.to_string()})),
            },
            Request::SetKey(name, value) => match set_key(&name, &value) {
                Ok(w) => emit(json!({"type": "key_saved", "name": name, "where": w})),
                Err(e) => emit(json!({"type": "error", "error": e.to_string()})),
            },
            Request::VoiceStatus => {
                let missing = voice_missing(config);
                emit(json!({"type": "voice", "mode": format!("{:?}", config.voice.mode).to_lowercase(), "missing": missing, "ok": config.voice.mode != crate::config::VoiceMode::Off && missing.is_empty()}));
            }
            Request::Speak(text) => {
                let missing = voice_missing(config);
                if config.voice.mode == crate::config::VoiceMode::Off {
                    emit(json!({"type": "error", "error": "Voice is off. Turn it on under Voice & Audio in Luna's settings."}));
                } else if !missing.is_empty() {
                    emit(json!({"type": "error", "error": format!("Luna cannot speak yet: she needs {}. Run scripts/luna-voice-setup.sh in the Halcyon folder.", missing.join(", "))}));
                } else {
                    match crate::tts::speak(&text, &config.voice.mode, config).await {
                        Ok(()) => emit(json!({"type": "spoke"})),
                        Err(e) => emit(json!({"type": "error", "error": format!("Speaking failed: {e}")})),
                    }
                }
            }
            Request::Turn(text) => {
                let started = Instant::now();
                match crate::agent::run_routed_turn(&text, &mut memory, &*config).await {
                    Ok(outcome) => {
                        crate::agent::learning::append_turn("user", &text);
                        crate::agent::learning::append_turn("assistant", &outcome.text);
                        emit(json!({
                            "type": "reply", "text": outcome.text, "thinking": outcome.thinking,
                            "model": outcome.model, "secs": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                        }));
                    }
                    Err(e) => emit(json!({"type": "error", "error": e.to_string()})),
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_needs_text() {
        assert_eq!(parse_request(r#"{"type":"turn","text":"  hi  "}"#), Request::Turn("hi".into()));
        assert_eq!(parse_request(r#"{"type":"turn","text":"   "}"#), Request::Ignore);
        assert_eq!(parse_request(r#"{"type":"turn"}"#), Request::Ignore);
    }

    #[test]
    fn control_lines_and_noise() {
        assert_eq!(parse_request(r#"{"type":"clear"}"#), Request::Clear);
        assert_eq!(parse_request(r#"{"type":"quit"}"#), Request::Quit);
        assert_eq!(parse_request("not json"), Request::Ignore);
        assert_eq!(parse_request(r#"{"type":"rm -rf"}"#), Request::Ignore);
    }

    #[test]
    fn settings_requests_parse() {
        assert_eq!(parse_request(r#"{"type":"settings_get"}"#), Request::SettingsGet);
        assert_eq!(
            parse_request(r#"{"type":"setting_set","key":"llm.enable_thinking","value":true}"#),
            Request::SetSetting("llm.enable_thinking".into(), Value::Bool(true))
        );
        assert_eq!(parse_request(r#"{"type":"setting_set","key":"x"}"#), Request::Ignore);
        assert_eq!(parse_request(r#"{"type":"models"}"#), Request::Models);
        assert_eq!(parse_request(r#"{"type":"model_set","name":" qwen3:8b "}"#), Request::SetModel("qwen3:8b".into(), "model".into()));
        assert_eq!(parse_request(r#"{"type":"model_set","name":""}"#), Request::Ignore);
        assert_eq!(parse_request(r#"{"type":"key_set","name":"tavily","value":"k"}"#), Request::SetKey("tavily".into(), "k".into()));
    }

    #[test]
    fn the_settings_are_the_tuis_and_include_the_halcyon_switch() {
        let v = settings_json(&LunaConfig::default());
        let all = v.to_string();
        for k in ["llm.model", "external.allow_halcyon_actions", "external.allow_capability_actions", "voice.mode", "tavily"] {
            assert!(all.contains(k), "{k} missing: {all}");
        }
        // the two signed gates are shown as gates, never as toggles
        let gates: Vec<&Value> = v["categories"].as_array().unwrap().iter().flat_map(|c| c["items"].as_array().unwrap()).filter(|i| i["kind"] == "gate").collect();
        assert_eq!(gates.len(), 2, "{all}");
        // no secret value can be in the output: keys only say whether they are set
        assert!(!all.contains("api_key\":\""), "{all}");
    }

    #[test]
    fn the_signed_gates_cannot_be_set_from_here() {
        let mut c = LunaConfig::default();
        for k in ["llm.security_unrestricted", "external.allow_capability_actions"] {
            let e = apply_setting(&mut c, k, &Value::Bool(true)).unwrap_err().to_string();
            assert!(e.contains("developer key"), "{e}");
        }
        assert!(!c.external.allow_capability_actions);
        assert!(apply_setting(&mut c, "no.such.key", &Value::Bool(true)).is_err());
        assert!(apply_setting(&mut c, "llm.enable_thinking", &Value::String("yes".into())).is_err());
    }

    #[test]
    fn a_key_must_look_like_one() {
        assert!(set_key("tavily", "").is_err());
        assert!(set_key("tavily", "abc\ndef").is_err());
        assert!(set_key("nonsense", "abc").unwrap_err().to_string().contains("unknown key"));
        assert!(set_key("tavily", &"x".repeat(600)).is_err());
    }

    #[test]
    fn activity_events_carry_their_kind() {
        let e = crate::activity::Event::ToolStart { name: "run_shell".into(), summary: "ls".into() };
        assert_eq!(activity_event(&e)["kind"], "tool_start");
        let t = crate::activity::Event::Thinking { delta: "hm".into() };
        assert_eq!(activity_event(&t)["type"], "thinking");
    }
}
