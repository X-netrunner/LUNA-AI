use crate::tools::shell::run_command;
use crate::config::LunaConfig;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

pub async fn notify(title: &str, body: &str, sudo_pass: Option<&str>) -> Result<()> {
    run_command(
        &format!(
            "notify-send '{}' '{}'",
            title.replace('\'', "'\\''"),
            body.replace('\'', "'\\''")
        ),
        sudo_pass,
    )
    .await?;
    Ok(())
}

/// True when `cmd` resolves to a real executable on PATH (`command -v`).
async fn command_exists(cmd: &str) -> bool {
    run_stdout(&format!("command -v {cmd} 2>/dev/null"))
        .await
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

// ── Desktop computer-use agent ────────────────────────────────────────────────
// The "bigger than SIH" mode: Luna stares at the WHOLE desktop via grim
// screenshots, asks the SIH VLM (Qwen2.5-VL) what to do next, and drives the
// screen with ydotool — so it can control ANY application, not just a browser.
//
// Loop:  grim screenshot (base64) → POST /desktop-act → VLM picks next action
//        (click x/y | type | press | scroll | launch | wait | done)
//        → execute via ydotool → repeat until done or max_steps.

const YDOTOOL_SOCKET: &str = "/run/user/1000/.ydotool_socket";
const SCREENSHOT_TMP: &str = "/tmp/luna-desktop-shot.png";
const SIH_HEALTH_PATH: &str = "/health";

/// Resolve "open firefox" from a word the VLM (or user) used. Returns the
/// command to launch, or None when it's not a known app name.
fn app_command(word: &str) -> Option<&'static str> {
    let w = word.to_lowercase();
    let w = w.trim().trim_matches('"').trim_matches('\'');
    let known: &[(&str, &str)] = &[
        // browsers
        ("firefox", "firefox"),
        ("chrome", "google-chrome-stable"),
        ("google chrome", "google-chrome-stable"),
        ("chromium", "chromium"),
        ("brave", "brave-browser"),
        ("zen", "zen-browser"),
        // editors / dev
        ("code", "code"),
        ("vscode", "code"),
        ("visual studio code", "code"),
        ("codium", "codium"),
        // terminals
        ("terminal", "kitty"),
        ("kitty", "kitty"),
        ("konsole", "konsole"),
        ("alacritty", "alacritty"),
        // comms
        ("whatsapp", "whatsdesk"),
        ("telegram", "telegram-desktop"),
        ("discord", "discord"),
        ("slack", "slack"),
        // media / misc
        ("spotify", "spotify"),
        ("music", "spotify"),
        ("vlc", "vlc"),
        ("files", "nautilus"),
        ("file manager", "nautilus"),
        ("nautilus", "nautilus"),
        ("settings", "gnome-control-center"),
        ("calculator", "gnome-calculator"),
        ("calendar", "gnome-calendar"),
        ("obsidian", "obsidian"),
        ("steam", "steam"),
    ];
    known
        .iter()
        .find(|(k, _)| *k == w)
        .map(|(_, cmd)| *cmd)
}

/// Start ydotoold if it isn't already running (needs the `input` group + uinput).
async fn ensure_ydotoold(config: &LunaConfig) -> Result<()> {
    let socket = std::path::Path::new(YDOTOOL_SOCKET);
    if socket.exists() {
        return Ok(());
    }
    // Try the systemd user unit first (fast, honors the socket dir).
    let _ = run_command("systemctl --user start ydotool.service", None).await;
    // Give it a beat, then check again.
    tokio::time::sleep(Duration::from_millis(800)).await;
    if socket.exists() {
        return Ok(());
    }
    // Fall back to a plain background daemon.
    let bin = ydotool_bin(config);
    let daemon = if bin.ends_with("ydotool") {
        bin.replace("ydotool", "ydotoold")
    } else {
        "ydotoold".to_string()
    };
    let _ = run_command(&format!("nohup {daemon} >/tmp/ydotoold.log 2>&1 &"), None).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    if !socket.exists() {
        bail!(
            "ydotoold isn't running (socket {YDOTOOL_SOCKET} missing). Start it with \
             `systemctl --user enable --now ydotool` (you're in the input group, /dev/uinput \
             exists) and try again."
        );
    }
    Ok(())
}

fn ydotool_bin(config: &LunaConfig) -> String {
    let bin = config.desktop.ydotool_bin.trim();
    if !bin.is_empty() {
        return bin.trim_end_matches('/').to_string();
    }
    let path = std::env::var("PATH").unwrap_or_default();
    for dir in path.split(':') {
        let candidate = format!("{dir}/ydotool");
        if std::path::Path::new(&candidate).exists() {
            return candidate;
        }
    }
    "ydotool".to_string()
}

/// Make sure the SIH server is answering; auto-start it from server_dir if set.
async fn ensure_sih(config: &LunaConfig) -> Result<()> {
    let base = config.desktop.srijan_url.trim_end_matches('/').to_string();
    let url = format!("{base}{SIH_HEALTH_PATH}");
    if reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .is_ok()
    {
        return Ok(());
    }

    let dir = PathBuf::from(&config.desktop.server_dir);
    if !dir.join("main.py").exists() {
        bail!(
            "The Project-Vision VLM server isn't running and I don't know where to start it. \
             Set [desktop] server_dir in luna.toml to the folder holding its main.py \
             (e.g. ~/Projects/sih/Project-Vision/Server[unnati&srijan]), or start it yourself."
        );
    }

    tracing::info!("starting Project-Vision server from {}", dir.display());
    let log_path = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("luna")
        .join("project-vision-server.log");
    std::fs::create_dir_all(log_path.parent().expect("log has parent"))?;
    let log = std::fs::File::create(&log_path)?;
    let mut cmd = std::process::Command::new("python3");
    cmd.arg("main.py")
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    cmd.spawn().context("spawn python main.py")?;

    // The VLM takes a while to load under 4-bit quant; poll up to ~120s.
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_secs(2)).await;
        if reqwest::Client::new()
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .is_ok()
        {
            tracing::info!("Project-Vision server is up");
            return Ok(());
        }
    }
    bail!("started the Project-Vision server but it never became healthy — check its log");
}

async fn capture_screenshot(config: &LunaConfig) -> Result<Vec<u8>> {
    let tool = if config.desktop.screenshot_cmd.trim().is_empty() {
        "grim".to_string()
    } else {
        config.desktop.screenshot_cmd.trim().to_string()
    };
    let out = run_command(&format!("{tool} {SCREENSHOT_TMP}"), None).await?;
    if !std::path::Path::new(SCREENSHOT_TMP).exists() {
        bail!(
            "screenshot failed ({}): {}",
            tool,
            out.stderr.trim().chars().take(200).collect::<String>()
        );
    }
    std::fs::read(SCREENSHOT_TMP).context("read screenshot PNG")
}

use base64::Engine;
fn encode_b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Ask the SIH VLM for the next action given the current screenshot + goal.
async fn ask_vlm(config: &LunaConfig, goal: &str, image_b64: &str, history: &[Value]) -> Result<Value> {
    let base = config.desktop.srijan_url.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{base}/desktop-act"))
        .json(&json!({ "image": image_b64, "goal": goal, "history": history }))
        .timeout(Duration::from_secs(90))
        .send()
        .await
        .context("POST /desktop-act — is the SIH server up?")?;
    let status = res.status();
    let v: Value = res.json().await.unwrap_or(Value::Null);
    if status.is_success() && v["ok"].as_bool() == Some(true) {
        Ok(v["action"].clone())
    } else {
        let err = v["error"].as_str().unwrap_or("unknown error");
        bail!("desktop-act failed ({status}): {err}");
    }
}

async fn run_stdout(cmd: &str) -> Result<String> {
    let out = run_command(cmd, None).await?;
    Ok(out.stdout)
}

/// Execute one ydotool-powered action. Returns a human-readable note.
async fn execute_action(config: &LunaConfig, action: Value) -> Result<String> {
    let yd = ydotool_bin(config);
    let kind = action["action"].as_str().unwrap_or("wait");
    match kind {
        "launch" => {
            let app = action["app"].as_str().unwrap_or("file-manager");
            match app_command(app) {
                // Fast path: a knowable binary with a real command. Verify it
                // exists before spawning so a stale map entry ("code", "kitty")
                // never starts a shell that errors instantly.
                Some(cmd) if command_exists(cmd).await => {
                    run_command(&format!("setsid nohup {cmd} >/dev/null 2>&1 &"), None).await?;
                    Ok(format!("launched {app} ({cmd})"))
                }
                // Universal path (Hyprland): Super opens the app menu, then
                // type the app's name and hit Enter. Works for ANY app even
                // when we have no mapping or the binary has a different name.
                _ => {
                    let yd = ydotool_bin(config);
                    // Super opens the launcher (rofi/wofi/any on Hyprland).
                    run_stdout(&format!("{yd} key 125:1 125:0")).await?;
                    tokio::time::sleep(Duration::from_millis(700)).await;
                    let name = app.trim().trim_matches('"').to_string();
                    run_stdout(&format!("{yd} type --escape=1 {}", shell_quote(&name))).await?;
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    run_stdout(&format!("{yd} key 28:1 28:0")).await?;
                    Ok(format!("opened {app} via Super-menu"))
                }
            }
        }
        "click" | "double_click" | "right_click" => {
            let x = action["x"].as_i64().unwrap_or(0);
            let y = action["y"].as_i64().unwrap_or(0);
            run_command(&format!("{yd} mousemove -a -x {x} -y {y}; {yd} click 0xC0"), None).await?;
            if kind == "double_click" {
                run_command(&format!("{yd} click -r 2 0xC0"), None).await?;
            } else if kind == "right_click" {
                run_command(&format!("{yd} mousemove -a -x {x} -y {y}; {yd} click 0xC1"), None).await?;
            }
            Ok(format!("{kind} at ({x},{y})"))
        }
        "type" => {
            let text = action["text"].as_str().unwrap_or("");
            if text.is_empty() {
                bail!("VLM returned an empty type text");
            }
            run_stdout(&format!("{yd} type --escape=1 {}", shell_quote(text))).await?;
            // Typing alone never navigates: URLs and searches need Enter to
            // submit. Guarded so free-text (notes, names) isn't auto-submitted.
            let looks_navigable = text.contains("://")
                || text.starts_with("www.")
                || text.contains(".com")
                || text.contains(".io")
                || text.contains(".org")
                || text.contains(".github");
            if looks_navigable {
                tokio::time::sleep(Duration::from_millis(300)).await;
                run_stdout(&format!("{yd} key 28:1 28:0")).await?;
            }
            Ok(format!("typed {len} chars{n}", len = text.chars().count(), n = if looks_navigable { " + Enter" } else { "" }))
        }
        "press" => {
            let key = action["key"].as_str().unwrap_or("enter");
            press_key(config, key).await?;
            Ok(format!("pressed {key}"))
        }
        "scroll" => {
            let dir = action["direction"].as_str().unwrap_or("down");
            let amt = action["amount"].as_i64().unwrap_or(300);
            let wheel = if dir == "up" { amt } else { -amt };
            run_stdout(&format!("{yd} mousemove --wheel -y {wheel}")).await?;
            Ok(format!("scrolled {dir} ({amt})"))
        }
        "wait" => {
            let ms = action["ms"].as_i64().unwrap_or(1000).clamp(100, 15000) as u64;
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(format!("waited {ms}ms"))
        }
        _ => bail!("unknown desktop action: {kind}"),
    }
}

async fn press_key(config: &LunaConfig, key: &str) -> Result<()> {
    let yd = ydotool_bin(config);
    // keycode map for common keys (Linux input-event-codes.h)
    let single: &[(&str, u16)] = &[
        ("enter", 28),
        ("return", 28),
        ("tab", 15),
        ("escape", 1),
        ("esc", 1),
        ("backspace", 14),
        ("space", 57),
        ("up", 103),
        ("down", 108),
        ("left", 105),
        ("right", 106),
        ("super", 125),
        ("home", 102),
        ("end", 107),
        ("pageup", 104),
        ("pagedown", 109),
        ("delete", 111),
        ("insert", 110),
    ];
    for (name, code) in single {
        if key == *name {
            run_stdout(&format!("{yd} key {code}:1 {code}:0")).await?;
            return Ok(());
        }
    }
    // Modifier combos like ctrl+c / alt+tab / super+d
    let (mod_name, rest) = key
        .split_once('+')
        .ok_or_else(|| anyhow!("unrecognized key: {key}"))?;
    let mod_code: u16 = match mod_name {
        "ctrl" | "control" => 29,
        "alt" => 56,
        "shift" => 42,
        "super" | "meta" | "win" => 125,
        _ => bail!("unrecognized modifier: {mod_name}"),
    };
    let base_key = single.iter().find(|(n, _)| *n == rest).map(|(_, c)| *c);
    let plain_code: u16 = match rest {
        s if s.len() == 1 && s.as_bytes()[0].is_ascii_lowercase() => {
            (s.as_bytes()[0] - b'a' + 30u8) as u16
        }
        s if s.len() == 1 && s.as_bytes()[0].is_ascii_digit() => {
            (s.as_bytes()[0] - b'0' + 2u8) as u16
        }
        _ => base_key.ok_or_else(|| anyhow!("unrecognized key in combo: {rest}"))?,
    };
    run_stdout(&format!(
        "{yd} key {mod_code}:1 {plain_code}:1 {plain_code}:0 {mod_code}:0"
    ))
    .await?;
    Ok(())
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "'\\''"))
}

/// Run one desktop-computer-use task to completion.
pub async fn run(task: &str, config: &LunaConfig) -> Result<String> {
    if !config.desktop.enabled {
        bail!("Desktop automation is disabled in config ([desktop] enabled = false).");
    }
    let task = task.trim();
    if task.is_empty() {
        bail!("The desktop task was empty. Ask the user what they want done on the computer.");
    }
    ensure_ydotoold(config).await.context("ydotoold check")?;
    ensure_sih(config).await.context("SIH server check")?;

    let cap = config.desktop.max_steps.max(1) as usize;
    let mut history: Vec<Value> = Vec::new();
    let mut steps: Vec<String> = Vec::new();
    let mut repeats = 0u32;
    let mut last_action: Option<String> = None;
    let started = std::time::Instant::now();
    let timeout = Duration::from_secs(config.desktop.timeout_secs.max(1));

    for step_i in 0..cap {
        if started.elapsed() > timeout {
            return Ok(format!(
                "Desktop task stopped: exceeded the {}s timeout after {} actions.\nActions: {}",
                config.desktop.timeout_secs,
                steps.len(),
                summarize(&history),
            ));
        }

        let png = capture_screenshot(config).await?;
        let b64 = encode_b64(&png);

        let action = match ask_vlm(config, task, &b64, &history).await {
            Ok(a) => a,
            Err(_) => {
                // The VLM might be mid-load; one retry with a short wait, then bail.
                tokio::time::sleep(Duration::from_secs(2)).await;
                ask_vlm(config, task, &b64, &history).await?
            }
        };

        let kind = action["action"].as_str().unwrap_or("wait").to_string();
        if kind == "done" {
            let summary = action["summary"].as_str().unwrap_or("task complete");
            return Ok(format!(
                "Done. Desktop task finished in {} actions.\n\n{summary}\n\nActions taken:\n{}",
                steps.len(),
                summarize(&history),
            ));
        }

        // Loop guard: the VLM occasionally fixates on the same action (typing
        // a URL into the wrong focus, clicking the same dead spot). If it
        // repeats the identical non-wait action 3x, stop and hand control back
        // instead of burning the whole step budget on a stuck loop.
        let fingerprint = action.to_string();
        let repeat_of_same = kind != "wait" && last_action.as_deref() == Some(fingerprint.as_str());
        repeats = if repeat_of_same { repeats + 1 } else { 0 };
        last_action = Some(fingerprint);
        if repeats >= 3 {
            return Ok(format!(
                "Desktop task stopped: the vision model repeated the same action \
                 ({kind}) 3 times with no visible progress. It may be stuck trying \
                 to interact with something it can't reach.\nActions so far:\n{}",
                summarize(&history),
            ));
        }

        let note = execute_action(config, action.clone())
            .await
            .with_context(|| format!("executing desktop action {kind}"))
            .map_err(|e| anyhow!("{e:#} — I can retry or a human can take over."))?;
        steps.push(format!("{}. {}", step_i + 1, note));
        history.push(json!({
            "action": kind,
            "note": note,
            "args": action,
        }));
        // Wait briefly so the UI can react to the action before the next look.
        tokio::time::sleep(Duration::from_millis(900)).await;
    }

    Ok(format!(
        "Desktop task reached the {cap}-action cap without a 'done'. Current progress:\n{}",
        summarize(&history),
    ))
}

fn summarize(history: &[Value]) -> String {
    if history.is_empty() {
        return "(no actions taken yet)".to_string();
    }
    history
        .iter()
        .filter_map(|h| h["note"].as_str())
        .enumerate()
        .map(|(i, n)| format!("{}. {n}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_command_handles_nown_apps() {
        assert_eq!(app_command("firefox"), Some("firefox"));
        assert_eq!(app_command("Firefox"), Some("firefox"));
        assert_eq!(app_command("code"), Some("code"));
        assert_eq!(app_command("\"spotify\""), Some("spotify"));
        assert_eq!(app_command("file manager"), Some("nautilus"));
        assert_eq!(app_command("hovercraft"), None);
        assert_eq!(app_command(""), None);
    }

    #[test]
    fn shell_quote_escapes_quotes() {
        assert_eq!(shell_quote("hello world"), "'hello world'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn summarize_prints_numbered_notes() {
        let h = json!({ "note": "clicked at (5,5)" });
        let h2 = json!({ "note": "pressed enter" });
        let s = summarize(&[h, h2]);
        assert!(s.contains("1. clicked at (5,5)"));
        assert!(s.contains("2. pressed enter"));
        assert!(summarize(&[]).contains("no actions"));
    }

    /// Live end-to-end: full grim → VLM → ydotool loop against the real SIH
    /// server on 127.0.0.1:8001. Moves the real mouse and launches real apps —
    /// run only when a human is watching the screen.
    #[tokio::test]
    #[ignore = "live desktop automation; grabs the mouse and opens real apps"]
    async fn live_desktop_loop() {
        use crate::config::{DesktopConfig, LunaConfig};
        let mut cfg = LunaConfig::default();
        cfg.desktop = DesktopConfig {
            enabled: true,
            srijan_url: "http://127.0.0.1:8001".into(),
            server_dir: "/home/netrunner/Projects/sih/Project-Vision/Server[unnati&srijan]"
                .into(),
            screenshot_cmd: "grim".into(),
            ydotool_bin: "ydotool".into(),
            max_steps: 25,
            timeout_secs: 300,
        };
        let out = crate::tools::desktop::run(
            "Open firefox and go to the Luna project on GitHub",
            &cfg,
        )
        .await
        .expect("live desktop loop should complete");
        println!("{out}");
    }
}