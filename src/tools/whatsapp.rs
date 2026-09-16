//! tools/whatsapp.rs — sends WhatsApp messages through the local bridge.
//!
//! The heavy lifting lives in the standalone Node service `whapp/bridge.mjs`
//! (a Baileys client). It links the user's account once via QR (`luna
//! --whatsapp-link`), holds the live socket, and exposes a tiny HTTP API on
//! 127.0.0.1. Luna only POSTs to that localhost API — no account state lives
//! in this process.
//!
//! Runtime data: ~/.local/share/whapp/{bridge.mjs,node_modules,session,config.json}

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;

const DEFAULT_PORT: u16 = 7373;

pub fn data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("whapp")
}

/// Human-readable reason the bridge can't be used, or None if it can.
pub fn missing_reason() -> Option<String> {
    let dir = data_dir();
    if !dir.join("bridge.mjs").exists() {
        return Some("the WhatsApp bridge isn't installed".into());
    }
    if !dir.join("node_modules").exists() {
        return Some("the WhatsApp bridge dependencies are missing. Run `luna --whatsapp-link` to install them.".into());
    }
    None
}

fn read_config() -> Result<Value> {
    let cfg = data_dir().join("config.json");
    let raw =
        std::fs::read_to_string(&cfg).context("read ~/.local/share/whapp/config.json")?;
    let v: Value = serde_json::from_str(&raw).context("parse whapp config.json")?;
    if v["token"].as_str().is_none() {
        return Err(anyhow!("no token in {}", cfg.display()));
    }
    Ok(v)
}

fn read_token() -> Result<String> {
    read_config()?
        .get("token")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("no token in whapp config.json"))
}

fn my_phone() -> Option<String> {
    read_config()
        .ok()?
        .get("my_phone")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn base_url(explicit: Option<&str>) -> String {
    explicit
        .unwrap_or("")
        .trim_end_matches('/')
        .to_string()
}
fn resolved_base(explicit: Option<&str>) -> String {
    if base_url(explicit).is_empty() {
        format!("http://127.0.0.1:{DEFAULT_PORT}")
    } else {
        base_url(explicit)
    }
}

/// Turn whatever the user said into a number the bridge can use:
///   "myself" / "me"      → the phone number of the linked account
///   "15551234567"/"+919..." → used as-is
///   "mom", "alice", ...   → looked up in the WhatsApp contacts via the bridge
async fn resolve_recipient(
    to: &str,
    base: &str,
    token: &str,
) -> Result<String> {
    let t = to.trim().to_string();
    let low = t.to_lowercase();

    if low == "myself" || low == "me" || low == "my number" || low == "my phone" {
        if let Some(p) = my_phone() {
            return Ok(p);
        }
        bail!(
            "I can resolve 'myself' once the bridge has seen your number — it records it right \
             after linking. Ask again in a moment, or give me the full number once."
        );
    }

    // A bare number (optionally prefixed with +) passes straight through.
    if t.chars().all(|c| c.is_ascii_digit()) || t.starts_with('+') {
        return Ok(t);
    }

    // Anything else is a name: look it up in the bridge's contact list.
    let client = reqwest::Client::new();
    let res = client
        .get(format!("{base}/resolve"))
        .bearer_auth(token)
        .query(&[("name", &t)])
        .timeout(std::time::Duration::from_secs(8))
        .send()
        .await
        .context("WhatsApp bridge is offline — is luna-whapp running?")?;
    let status = res.status();
    let v: Value = res.json().await.unwrap_or(Value::Null);
    if status == reqwest::StatusCode::NOT_FOUND {
        let err = v["error"].as_str().unwrap_or("contact not found");
        bail!(
            "{err} — either say a number, or have them message you once so I can find them in contacts."
        );
    }
    if let Some(phone) = v["matches"]
        .as_array()
        .and_then(|m| m.first())
        .and_then(|m| m["phone"].as_str())
    {
        return Ok(phone.to_string());
    }
    bail!(
        "no WhatsApp contact matching \"{t}\". Use the full number once, or have them message you so I can save them."
    );
}

/// Look up a contact by name and report number(s) without sending anything.
pub async fn lookup(name: &str, explicit_base: Option<&str>) -> Result<String> {
    let base = resolved_base(explicit_base);
    let token = read_token()?;
    let client = reqwest::Client::new();
    let res = client
        .get(format!("{base}/resolve"))
        .bearer_auth(&token)
        .query(&[("name", name)])
        .timeout(std::time::Duration::from_secs(8))
        .send()
        .await
        .context("WhatsApp bridge is offline — is luna-whapp running?")?;
    let status = res.status();
    let v: Value = res.json().await.unwrap_or(Value::Null);
    if let Some(matches) = v["matches"].as_array() {
        if !matches.is_empty() {
            let lines: Vec<String> = matches
                .iter()
                .map(|m| {
                    let n = m["name"].as_str().unwrap_or(name);
                    let p = m["phone"].as_str().unwrap_or("?");
                    format!("- {n}: +{p}")
                })
                .collect();
            return Ok(format!("Contacts matching \"{name}\":\n{}", lines.join("\n")));
        }
    }
    let err = v["error"].as_str().unwrap_or("no contact found");
    let _ = status;
    Ok(format!(
        "{err} — still no contact named \"{name}\". Have them message you once (or re-link to \
         backfill the contact book), then I'll know their number without being told it."
    ))
}

/// Keep the shipped bridge script current. Luna ships its own copy of
/// `whapp/bridge.mjs`; if the installed copy under ~/.local/share/whapp is an
/// older revision, refresh it and restart the systemd unit so new endpoints
/// take effect without a manual `luna --whatsapp-link`. Cheap (one file read),
/// safe to call before any bridge call.
pub fn sync_bridge() {
    let dir = data_dir();
    if !dir.join("bridge.mjs").exists() {
        return; // never installed
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("whapp");
    let mut changed = false;
    for name in ["bridge.mjs", "package.json"] {
        let src = repo.join(name);
        let dst = dir.join(name);
        if !src.exists() {
            continue;
        }
        let shipped = match std::fs::read(&src) {
            Ok(b) => b,
            Err(_) => continue,
        };
        if std::fs::read(&dst).map(|b| b == shipped).unwrap_or(false) {
            continue;
        }
        if std::fs::write(&dst, shipped).is_ok() {
            changed = true;
        }
    }
    if changed {
        tracing::info!("refreshed the WhatsApp bridge script (new endpoints live)");
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "restart", "luna-whapp"])
            .status();
    }
}

fn render_contacts(v: &Value) -> String {
    let list = match v["contacts"].as_array() {
        Some(l) => l,
        None => {
            return "The bridge returned a response I couldn't parse. Try again in a moment."
                .to_string();
        }
    };
    if list.is_empty() {
        return "No WhatsApp contacts are indexed yet — WhatsApp only pushes your contact book \
                on a device link. Re-link once (`luna --whatsapp-link`) to backfill it, or have \
                the person message you once."
            .to_string();
    }
    let mut lines: Vec<String> = list
        .iter()
        .map(|c| {
            let name = c["name"].as_str().unwrap_or("(unnamed)");
            match c["phone"].as_str() {
                Some(p) => format!("- {name}: +{p}"),
                None => format!("- {name}"),
            }
        })
        .collect();
    lines.sort();
    lines.insert(
        0,
        format!("WhatsApp contact book ({} entries):", list.len()),
    );
    lines.join("\n")
}

/// List the indexed WhatsApp contact book (name ↔ phone).
///
/// Prefers the live bridge (`GET /contacts`); if the bridge is down or not
/// linked, falls back to the on-disk contact book the bridge itself maintains.
pub async fn contacts(query: Option<&str>, explicit_base: Option<&str>) -> Result<String> {
    sync_bridge();
    if let Some(reason) = missing_reason() {
        bail!("WhatsApp bridge isn't installed ({reason}). Run `luna --whatsapp-link` to set it up.");
    }
    let q = query.map(str::trim).filter(|s| !s.is_empty());

    let live: Option<String> = (async || {
        let token = read_token().ok()?;
        let base = resolved_base(explicit_base);
        let client = reqwest::Client::new();
        let mut req = client
            .get(format!("{base}/contacts"))
            .bearer_auth(&token)
            .timeout(std::time::Duration::from_secs(8));
        if let Some(q) = q {
            req = req.query(&[("q", q)]);
        }
        let res = req.send().await.ok()?;
        let v = res.json::<Value>().await.ok()?;
        if v["ok"].as_bool() == Some(true) {
            Some(render_contacts(&v))
        } else {
            None
        }
    })()
    .await;

    if let Some(rendered) = live {
        return Ok(rendered);
    }

    // Bridge offline or not linked — read the contact book the bridge maintains.
    let path = data_dir().join("contacts.json");
    match std::fs::read_to_string(&path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(v) if v.is_array() => Ok(render_contacts(&json!({ "contacts": v }))),
            _ => Ok("The bridge is offline and I couldn't read the saved contact book. \
                     Start the bridge (luna-whapp.service) and try again."
                .to_string()),
        },
        Err(_) => Ok("The WhatsApp bridge is offline and no saved contact book exists yet. \
                      Make sure luna-whapp.service is running, then try again."
            .to_string()),
    }
}

/// Send a WhatsApp message through the local bridge.
pub async fn send(to: &str, text: &str, explicit_base: Option<&str>) -> Result<String> {
    if let Some(reason) = missing_reason() {
        bail!("Can't send: {reason}. Run `luna --whatsapp-link` once to pair your WhatsApp.");
    }
    let token = read_token()?;
    let base = resolved_base(explicit_base);
    let recipient = resolve_recipient(to, &base, &token).await?;
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{base}/send"))
        .bearer_auth(&token)
        .json(&json!({ "to": recipient, "text": text }))
        .timeout(std::time::Duration::from_secs(45))
        .send()
        .await
        .context("WhatsApp bridge is offline — is luna-whapp running?")?;
    let status_code = res.status();
    let v: Value = res.json().await.unwrap_or(Value::Null);
    match v["ok"].as_bool() {
        Some(true) => Ok(format!("Message sent to {recipient} on WhatsApp ({text_len} chars).", text_len = text.trim().chars().count())),
        _ => {
            let err = v["error"].as_str().unwrap_or("unknown bridge error");
            if status_code == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                bail!("{err}");
            }
            bail!("WhatsApp send failed: {err}");
        }
    }
}

/// Live status of the bridge and its connection.
pub async fn status(explicit_base: Option<&str>) -> Result<String> {
    if let Some(reason) = missing_reason() {
        return Ok(format!(
            "WhatsApp bridge not installed ({reason}). Run `luna --whatsapp-link` once to pair your WhatsApp."
        ));
    }
    let client = reqwest::Client::new();
    let base = resolved_base(explicit_base);
    match client
        .get(format!("{base}/health"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(res) => {
            let v: Value = res.json().await.unwrap_or(Value::Null);
            let connected = v["connected"].as_bool().unwrap_or(false);
            Ok(if connected {
                let who = my_phone()
                    .map(|p| format!(" (your number: +{p})"))
                    .unwrap_or_default();
                format!("WhatsApp bridge is up and linked{who}.")
            } else {
                format!("WhatsApp bridge is running but not linked — run `luna --whatsapp-link` and scan the QR.")
            })
        }
        Err(_) => Ok(format!(
            "WhatsApp bridge is not running. Start it (it auto-starts with luna-whapp.service) or run `luna --whatsapp-link`."
        )),
    }
}

/// One-time pairing: ensure the bridge is installed, print the QR code, and
/// once linked, leave the always-on service running. Blocks until linked.
pub async fn install_and_link() -> Result<()> {
    tokio::task::block_in_place(install_files);
    install_node_deps().await?;

    let dir = data_dir();
    let cfg_path = dir.join("config.json");
    if !cfg_path.exists() {
        // materialize config.json (token generation is the bridge's job)
        read_token()?;
    }

    println!("\n⚡ Luna WhatsApp bridge");
    println!("data dir: {}", dir.display());
    println!("The next step shows a QR code. In WhatsApp on your phone:");
    println!("  Settings → Linked devices → Link a device → scan it.\n");

    let status = tokio::process::Command::new("node")
        .arg(dir.join("bridge.mjs"))
        .arg("link")
        .current_dir(&dir)
        .status()
        .await
        .context("failed to run the bridge (is node installed?)")?;

    if !status.success() {
        bail!("QR linking did not complete cleanly (exit {status}). Try `luna --whatsapp-link` again.");
    }

    // Leave the always-on bridge running so Luna can send later.
    crate::tools::whatsapp::enable_service()
        .context("linked your WhatsApp, but failed to start the background bridge")?;

    println!("\nWhatsApp linked and the bridge is running in the background.");
    Ok(())
}

/// Copy the whapp/ assets from the repo install into the data dir.
fn install_files() {
    let dir = data_dir();
    if dir.join("bridge.mjs").exists() {
        return; // already installed
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("whapp");
    if !repo.join("bridge.mjs").exists() {
        return; // nothing to copy from (running from a different checkout) — caller errors
    }
    let _ = std::fs::create_dir_all(&dir);
    for f in ["package.json", "bridge.mjs"] {
        let _ = std::fs::copy(repo.join(f), dir.join(f));
    }
}

async fn install_node_deps() -> Result<()> {
    let dir = data_dir();
    if !dir.join("bridge.mjs").exists() {
        bail!(
            "could not find the WhatsApp bridge sources in {}. Rebuild Luna from the repo checkout so whapp/ ships alongside it.",
            dir.display()
        );
    }
    if dir.join("node_modules").exists() {
        return Ok(());
    }
    println!("Installing WhatsApp bridge dependencies (one-time, needs network)...");
    let status = tokio::process::Command::new("npm")
        .args(["install", "--omit=dev", "--no-audit", "--no-fund"])
        .current_dir(&dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("failed to run npm install in the whapp dir")?;
    if !status.success() {
        bail!("npm install failed (exit {status}); the bridge can't run without its packages.");
    }
    Ok(())
}

/// Enable and start the per-user systemd unit that keeps the bridge alive.
pub fn enable_service() -> Result<()> {
    let unit = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("systemd")
        .join("user")
        .join("luna-whapp.service");
    if !unit.exists() {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("deploy/luna-whapp.service");
        std::fs::create_dir_all(unit.parent().expect("unit has parent dir"))?;
        std::fs::copy(src, &unit)?;
    }
    let ok = std::process::Command::new("systemctl")
        .args(["--user", "enable", "--now", "luna-whapp"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        anyhow::bail!("could not start luna-whapp.service via systemd");
    }
    Ok(())
}