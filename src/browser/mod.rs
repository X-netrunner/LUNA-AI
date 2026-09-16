//! browser/mod.rs — Luna drives the Project-Vision (SIH) browser-automation
//! server and a real Chromium instance so the user gets "do this in the
//! browser" task automation.
//!
//! Architecture:
//!   - Luna speaks the SAME WebSocket protocol the Project-Vision Chrome
//!     extension speaks (`ws://127.0.0.1:8001/ws`): send the user's goal
//!     (`USER_PROMPT`), hand over a screenshot, then for every `AGENT_ACTION`
//!     the server sends back (its planner + VLM deciding what to click/type),
//!     execute it in Chromium via the DevTools Protocol and return a fresh
//!     screenshot. The server keeps planning until the task is done.
//!   - Chromium is launched with `--remote-debugging-port` and a dedicated
//!     profile (~/.local/share/luna/browser-profile) so logins persist between
//!     automation sessions. Screenshots are raw PNGs — the redaction step the
//!     extension performed is unneeded because every hop stays on this machine.
//!
//! The browser stays open after the task so the user can see the result.

use anyhow::{anyhow, bail, Context, Result};
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;

// ── SIH server location ────────────────────────────────────────────────────────

/// Split a config `srijan_url` like `ws://127.0.0.1:8001/ws` into its HTTP base
/// (`http://127.0.0.1:8001`) and the original WS URL.
fn split_srijan(url: &str) -> (String, String) {
    let ws_url = url.trim_end_matches('/').to_string();
    let host = ws_url.strip_prefix("ws://").unwrap_or(&ws_url);
    let host = host.strip_suffix("/ws").unwrap_or(host).trim_end_matches('/');
    let http = format!("http://{host}");
    (http, ws_url)
}

async fn http_get(url: &str, timeout: Duration) -> Result<reqwest::Response> {
    reqwest::Client::new()
        .get(url)
        .timeout(timeout)
        .send()
        .await
        .with_context(|| format!("GET {url}"))
}

/// Make sure the Project-Vision server is up. If it isn't and `server_dir` is
/// configured, launch `python main.py` there and wait for it to answer.
async fn ensure_server(config: &crate::config::BrowserConfig) -> Result<()> {
    let (base, _) = split_srijan(&config.srijan_url);
    if http_get(&format!("{base}/health"), Duration::from_secs(2)).await.is_ok() {
        return Ok(());
    }

    let dir = PathBuf::from(&config.server_dir);
    if !dir.join("main.py").exists() {
        bail!(
            "The Project-Vision browser server isn't running and I don't know where to start it. \
             Set [browser] server_dir in luna.toml to the folder holding its main.py \
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
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log.try_clone()?))
        .stderr(std::process::Stdio::from(log));
    match cmd.spawn() {
        Ok(_) => {}
        Err(e) => bail!("failed to launch the Project-Vision server: {e}"),
    }

    for _ in 0..30 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if http_get(&format!("{base}/health"), Duration::from_secs(2)).await.is_ok() {
            return Ok(());
        }
    }
    bail!(
        "The Project-Vision server started but never answered on {base}. Check \
         ~/.local/share/luna/project-vision-server.log for errors."
    )
}

// ── Chromium / CDP driver ──────────────────────────────────────────────────────

fn cdp_base(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

async fn list_targets(port: u16) -> Result<Value> {
    let res = http_get(&format!("{}/json/list", cdp_base(port)), Duration::from_secs(3)).await?;
    if !res.status().is_success() {
        bail!("CDP /json/list returned {}", res.status());
    }
    res.json::<Value>()
        .await
        .context("parse CDP target list")
}

/// Launch Chromium with remote debugging enabled. Uses a dedicated profile so
/// the instance is always controllable and keeps logins between sessions.
fn launch_chromium(config: &crate::config::BrowserConfig) -> Result<()> {
    let profile = if config.profile_dir.trim().is_empty() {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("luna")
            .join("browser-profile")
    } else {
        PathBuf::from(&config.profile_dir)
    };
    std::fs::create_dir_all(&profile)?;

    let log_path = profile.join("chromium.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let mut cmd = std::process::Command::new(&config.chromium);
    cmd.arg(format!("--user-data-dir={}", profile.display()))
        .arg(format!("--remote-debugging-port={}", config.cdp_port))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("--disable-background-networking")
        .arg("--disable-component-update")
        .arg("--password-store=basic")
        .arg("--window-size=1366,900")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log.try_clone()?))
        .stderr(std::process::Stdio::from(log));
    if config.headless {
        cmd.arg("--headless=new");
    }
    match cmd.spawn() {
        Ok(_) => Ok(()),
        Err(e) => bail!("failed to launch {}: {e}", config.chromium),
    }
}

/// Make sure a debuggable Chromium is running; returns a usable page target.
async fn ensure_chromium(config: &crate::config::BrowserConfig) -> Result<Value> {
    let mut targets = list_targets(config.cdp_port).await;
    if targets.is_err() {
        tracing::info!("no debuggable Chromium on port {}; launching one", config.cdp_port);
        launch_chromium(config)?;
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if let Ok(t) = list_targets(config.cdp_port).await {
                targets = Ok(t);
                break;
            }
        }
    }
    let targets = targets.context("Chromium never exposed a debugging endpoint")?;
    if let Some(target) = pick_page_target(&targets) {
        return Ok(target.clone());
    }
    // No usable page yet — create one ourselves.
    new_blank_target(config.cdp_port).await.context("no usable Chrome tab and can't create one")
}

/// Pick a page target to drive: prefer the last real page (non-blank, non
/// chrome://); otherwise return the newest page target, else None.
fn pick_page_target(targets: &Value) -> Option<&Value> {
    let arr = targets.as_array()?;
    let mut pages: Vec<&Value> = arr
        .iter()
        .filter(|t| t["type"] == "page" && t["webSocketDebuggerUrl"].is_string())
        .collect();
    if pages.is_empty() {
        return None;
    }
    if let Some(pos) = pages.iter().position(|t| {
        matches!(
            t["url"].as_str(),
            Some(u) if !u.is_empty() && !u.starts_with("chrome://") && u != "about:blank"
        )
    }) {
        let last = pages.remove(pos);
        pages.push(last);
    }
    pages.last().copied()
}

/// Create a brand-new blank tab via CDP and return its target descriptor.
async fn new_blank_target(port: u16) -> Result<Value> {
    reqwest::Client::new()
        .put(format!("{}/json/new?about:blank", cdp_base(port)))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("CDP /json/new")?
        .json::<Value>()
        .await
        .context("parse /json/new")
}

type CdpCall = (u64, Value, oneshot::Sender<Value>);

/// A single DevTools session to one page target. Commands are issued with
/// monotonically increasing ids and matched to responses by a reader task.
struct Cdp {
    cmd: mpsc::Sender<CdpCall>,
    id: u64,
    /// stable integer tab id reported to the Project-Vision server (1-based)
    tab_id: u64,
    target_id: String,
}

impl Cdp {
    async fn connect(ws_url: &str, tab_index: u64, target_id: String) -> Result<Self> {
        let (ws, _) = tokio_tungstenite::connect_async(ws_url)
            .await
            .with_context(|| format!("CDP connect to {ws_url}"))?;
        let (mut write, mut read) = ws.split();
        let (cmd, mut rx) = mpsc::channel::<CdpCall>(64);

        tokio::spawn(async move {
            let mut pending: HashMap<u64, oneshot::Sender<Value>> = HashMap::new();
            loop {
                tokio::select! {
                    out = rx.recv() => match out {
                        Some((id, msg, reply)) => {
                            pending.insert(id, reply);
                            if write
                                .send(Message::text(serde_json::to_string(&msg).unwrap_or_default()))
                                .await.is_err()
                            { break; }
                        }
                        None => break,
                    },
                    frame = read.next() => match frame {
                        Some(Ok(Message::Text(t))) => {
                            if let Ok(v) = serde_json::from_str::<Value>(&t) {
                                if let Some(id) = v.get("id").and_then(Value::as_u64) {
                                    if let Some(reply) = pending.remove(&id) {
                                        let _ = reply.send(v);
                                    }
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                        _ => {}
                    },
                }
            }
            for (_, reply) in pending {
                let _ = reply.send(Value::Null);
            }
        });

        Ok(Self { cmd, id: 0, tab_id: tab_index, target_id })
    }

    async fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        self.id += 1;
        let id = self.id;
        let msg = json!({ "id": id, "method": method, "params": params });
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send((id, msg, tx))
            .await
            .context("CDP command channel closed")?;
        let resp = tokio::time::timeout(Duration::from_secs(30), rx)
            .await
            .context("CDP command timed out")?
            .context("CDP channel dropped")?;
        if let Some(err) = resp.get("error") {
            bail!("CDP {method} failed: {}", err);
        }
        Ok(resp.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn eval(&mut self, expression: &str) -> Result<Value> {
        let r = self
            .call(
                "Runtime.evaluate",
                json!({ "expression": expression, "returnByValue": true, "awaitPromise": true }),
            )
            .await?;
        if r.get("exceptionDetails").is_some() {
            bail!("page script threw while evaluating");
        }
        Ok(r.get("result").and_then(|x| x.get("value")).cloned().unwrap_or(Value::Null))
    }

    async fn screenshot(&mut self) -> Result<String> {
        let r = self
            .call(
                "Page.captureScreenshot",
                json!({ "format": "png", "fromSurface": true }),
            )
            .await?;
        Ok(r.get("data").and_then(Value::as_str).unwrap_or("").to_string())
    }

    async fn navigate(&mut self, url: &str) -> Result<()> {
        self.call("Page.navigate", json!({ "url": url })).await?;
        self.wait_ready().await
    }

    async fn wait_ready(&mut self) -> Result<()> {
        for _ in 0..40 {
            let state = self
                .eval("document.readyState")
                .await
                .unwrap_or_else(|_| json!("unknown"));
            if state == "complete" {
                return Ok(());
            }
            // Some SPAs never reach 'complete'; a second beat of stability is enough.
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        Ok(())
    }

    async fn activate(&mut self) -> Result<()> {
        let target_id = self.target_id.clone();
        self.call("Target.activateTarget", json!({ "targetId": target_id })).await?;
        Ok(())
    }
}

// ── DOM action execution (mirrors the extension's content script) ─────────────

fn dom_click_expr(x: i64, y: i64) -> String {
    format!(
        r#"(() => {{
  const el = document.elementFromPoint({x}, {y});
  if (!el) return {{ ok: false, error: "no element at point" }};
  const clickable =
    el.closest('button,a,input,textarea,select,label,[role="button"],[role="radio"],[role="checkbox"],[role="option"],[role="menuitem"],[role="switch"],[onclick],[tabindex]') ||
    (el instanceof HTMLElement ? el : null);
  if (!clickable) return {{ ok: false, error: "element not clickable" }};
  const ev = {{ bubbles: true, cancelable: true, view: window, clientX: {x}, clientY: {y} }};
  clickable.dispatchEvent(new PointerEvent("pointerdown", ev));
  clickable.dispatchEvent(new MouseEvent("mousedown", ev));
  clickable.focus();
  clickable.dispatchEvent(new PointerEvent("pointerup", ev));
  clickable.dispatchEvent(new MouseEvent("mouseup", ev));
  clickable.click();
  return {{ ok: true }};
}})()"#
    )
}

fn dom_type_expr(text: &str) -> String {
    let t = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
  const isInput = (e) =>
    e instanceof HTMLInputElement || e instanceof HTMLTextAreaElement ||
    (e instanceof HTMLElement && e.isContentEditable);
  let el = document.activeElement;
  if (!(el && isInput(el))) {{
    el = document.querySelector(
      'input[type="search"],input[type="text"],textarea[name="q"],textarea[aria-label*="Search" i],input[name="q"],input[aria-label*="Search" i],textarea,input:not([type="hidden"])'
    );
  }}
  if (!el || !isInput(el)) return {{ ok: false, error: "no supported input found" }};
  el.focus();
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {{
    const d = Object.getOwnPropertyDescriptor(el.constructor.prototype, "value");
    const setter = d && d.set ? d.set : null;
    if (setter) setter.call(el, {t}); else el.value = {t};
    el.dispatchEvent(new Event("input", {{ bubbles: true, composed: true }}));
    el.dispatchEvent(new Event("change", {{ bubbles: true, composed: true }}));
    return {{ ok: true }};
  }}
  el.textContent = {t};
  el.dispatchEvent(new InputEvent("input", {{ bubbles: true, inputType: "insertText", data: {t} }}));
  return {{ ok: true }};
}})()"#
    )
}

fn dom_press_expr(key: &str) -> String {
    let k = serde_json::to_string(key).unwrap_or_else(|_| "\"Enter\"".into());
    format!(
        r#"(() => {{
  const K = {k};
  const t = document.activeElement instanceof HTMLElement ? document.activeElement : document.body;
  t.dispatchEvent(new KeyboardEvent("keydown", {{ key: K, bubbles: true, cancelable: true }}));
  t.dispatchEvent(new KeyboardEvent("keyup", {{ key: K, bubbles: true, cancelable: true }}));
  if (K === "Enter" && document.activeElement) {{
    const form = document.activeElement.closest ? document.activeElement.closest("form") : null;
    if (form) {{ if (typeof form.requestSubmit === "function") form.requestSubmit(); else form.submit(); }}
  }}
  return {{ ok: true }};
}})()"#
    )
}

fn dom_scroll_expr(direction: &str, amount: i64) -> String {
    let delta = if direction == "down" { amount } else { -amount };
    format!(
        r#"(() => {{ window.scrollBy({{ top: {delta}, behavior: "auto" }}); return {{ ok: true }}; }})()"#
    )
}

/// Execute one server action inside the browser. Mirrors the extension's
/// content script + background action handling.
async fn execute_action(
    browser: &mut Browser,
    action: &Value,
    step_index: u64,
    tab_id: u64,
) -> Result<ActionResult> {
    let name = action["action"].as_str().unwrap_or("").to_string();
    let fail = |error: String| ActionResult {
        success: false,
        action: name.clone(),
        step_index,
        tab_id,
        error: Some(error),
    };

    match name.as_str() {
        "click" => {
            let x = action["x"].as_i64().unwrap_or(0);
            let y = action["y"].as_i64().unwrap_or(0);
            let r = browser.current().await?.eval(&dom_click_expr(x, y)).await?;
            Ok(result_from(r, name, step_index, tab_id))
        }
        "type" => {
            let text = action["text"].as_str().unwrap_or("");
            let r = browser.current().await?.eval(&dom_type_expr(text)).await?;
            Ok(result_from(r, name, step_index, tab_id))
        }
        "press" => {
            let key = action["key"].as_str().unwrap_or("Enter");
            let r = browser.current().await?.eval(&dom_press_expr(key)).await?;
            Ok(result_from(r, name, step_index, tab_id))
        }
        "scroll" => {
            let direction = action["direction"].as_str().unwrap_or("down");
            let amount = action["amount"].as_i64().unwrap_or(500);
            let r = browser.current().await?.eval(&dom_scroll_expr(direction, amount)).await?;
            Ok(result_from(r, name, step_index, tab_id))
        }
        "navigate" => {
            let url = action["url"].as_str().unwrap_or("");
            if url.is_empty() {
                return Ok(fail("navigate needs a url".into()));
            }
            if let Err(e) = browser.current().await?.navigate(url).await {
                return Ok(fail(format!("navigation failed: {e:#}")));
            }
            Ok(ActionResult { success: true, action: name, step_index, tab_id, error: None })
        }
        "search" => {
            let query = action["query"].as_str().unwrap_or("");
            if query.is_empty() {
                return Ok(fail("search needs a query".into()));
            }
            let url = format!("https://duckduckgo.com/?q={}", percent_encode(query));
            if let Err(e) = browser.current().await?.navigate(&url).await {
                return Ok(fail(format!("search failed: {e:#}")));
            }
            Ok(ActionResult { success: true, action: name, step_index, tab_id, error: None })
        }
        "open_tab" => {
            let url = action["url"].as_str().unwrap_or("about:blank");
            match browser.open_tab(url).await {
                Ok(new_idx) => Ok(ActionResult {
                    success: true,
                    action: name,
                    step_index,
                    tab_id: new_idx,
                    error: None,
                }),
                Err(e) => Ok(fail(format!("open_tab failed: {e:#}"))),
            }
        }
        "switch_tab" => {
            let requested = action["tab_id"].as_u64().unwrap_or(tab_id);
            match browser.switch_tab(requested).await {
                Ok(_) => Ok(ActionResult { success: true, action: name, step_index, tab_id: requested, error: None }),
                Err(_) => Ok(fail("no such tab id to switch to".into())),
            }
        }
        "close_tab" => {
            let requested = action["tab_id"].as_u64().unwrap_or(tab_id);
            match browser.close_tab(requested).await {
                Ok(_) => Ok(ActionResult { success: true, action: name, step_index, tab_id: requested, error: None }),
                Err(_) => Ok(fail("no such tab id to close".into())),
            }
        }
        _ => Ok(fail(format!("unsupported action '{name}'"))),
    }
}

fn result_from(v: Value, action: String, step_index: u64, tab_id: u64) -> ActionResult {
    match v.get("ok").and_then(Value::as_bool) {
        Some(true) => ActionResult { success: true, action, step_index, tab_id, error: None },
        _ => ActionResult {
            success: false,
            action,
            step_index,
            tab_id,
            error: Some(v.get("error").and_then(Value::as_str).unwrap_or("page script failed").to_string()),
        },
    }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'!' | b'*' | b'\'' | b'(' | b')' | b';' | b':' | b'@' | b'&' | b'=' | b'+'
            | b'$' | b',' | b'/' | b'?' | b'#' | b'[' | b']' | b'%' => {
                out.push_str(&format!("%{b:02X}"));
            }
            _ if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[derive(Serialize)]
struct ActionResult {
    success: bool,
    action: String,
    step_index: u64,
    tab_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ── Browser container: multi-tab CDP state shared across one task ─────────────

struct Browser {
    sessions: Vec<Cdp>,
    current: usize,
    port: u16,
}

impl Browser {
    async fn attach(config: &crate::config::BrowserConfig, target: Value) -> Result<Self> {
        let ws_url = target["webSocketDebuggerUrl"]
            .as_str()
            .ok_or_else(|| anyhow!("target has no webSocketDebuggerUrl"))?
            .to_string();
        let target_id = target["id"].as_str().unwrap_or("").to_string();
        let mut session = Cdp::connect(&ws_url, 1, target_id).await?;
        session.call("Page.enable", json!({})).await?;
        session.call("Runtime.enable", json!({})).await?;
        Ok(Self { sessions: vec![session], current: 0, port: config.cdp_port })
    }

    async fn current(&mut self) -> Result<&mut Cdp> {
        self.sessions
            .get_mut(self.current)
            .ok_or_else(|| anyhow!("no active browser tab"))
    }

    async fn open_tab(&mut self, url: &str) -> Result<u64> {
        let res = reqwest::Client::new()
            .put(format!(
                "{}/json/new?{}",
                cdp_base(self.port),
                percent_encode(url)
            ))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("CDP /json/new")?
            .json::<Value>()
            .await
            .context("parse /json/new")?;
        let ws_url = res["webSocketDebuggerUrl"]
            .as_str()
            .ok_or_else(|| anyhow!("new tab has no debugger url"))?
            .to_string();
        let target_id = res["id"].as_str().unwrap_or("").to_string();
        let idx = (self.sessions.len() + 1) as u64;
        let mut session = Cdp::connect(&ws_url, idx, target_id).await?;
        session.call("Page.enable", json!({})).await?;
        session.call("Runtime.enable", json!({})).await?;
        if url != "about:blank" {
            session.navigate(url).await?;
        }
        self.sessions.push(session);
        self.current = self.sessions.len() - 1;
        Ok(idx)
    }

    async fn switch_tab(&mut self, tab_id: u64) -> Result<()> {
        let idx = (tab_id as usize).checked_sub(1).ok_or_else(|| anyhow!("tab id must be >= 1"))?;
        if idx >= self.sessions.len() {
            bail!("tab id {tab_id} out of range");
        }
        self.sessions[idx].activate().await?;
        self.current = idx;
        Ok(())
    }

    async fn close_tab(&mut self, tab_id: u64) -> Result<()> {
        let idx = (tab_id as usize).checked_sub(1).ok_or_else(|| anyhow!("tab id must be >= 1"))?;
        if idx >= self.sessions.len() {
            bail!("tab id {tab_id} out of range");
        }
        let target_id = self.sessions[idx].target_id.clone();
        let _ = reqwest::Client::new()
            .put(format!("{}/json/close/{target_id}", cdp_base(self.port)))
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        self.sessions.remove(idx);
        // renumber remaining tabs
        for (i, s) in self.sessions.iter_mut().enumerate() {
            s.tab_id = (i + 1) as u64;
        }
        self.current = self.current.min(self.sessions.len().saturating_sub(1));
        Ok(())
    }
}

// ── Project-Vision client loop ─────────────────────────────────────────────────

/// Run one browser-automation task to completion and summarize what happened.
pub async fn run(task: &str, config: &crate::config::LunaConfig) -> Result<String> {
    let bc = &config.browser;
    let (_, ws_url) = split_srijan(&bc.srijan_url);

    let steps = tokio::time::timeout(
        Duration::from_secs(bc.timeout_secs.max(1)),
        run_inner(task, bc, &ws_url),
    )
    .await
    .map_err(|_| anyhow!("browser task exceeded the {}s timeout", bc.timeout_secs))??;

    if steps.is_empty() {
        return Ok("The browser automation server finished without executing any steps.".into());
    }
    let history = steps.join("\n");
    Ok(format!(
        "Done. The browser automation ({bc} steps) executed:\n{history}\n\nThe browser window is \
         still open so you can see the final state; tell me if you'd like it closed or if the \
         result isn't what you wanted.",
        bc = steps.len()
    ))
}

async fn run_inner(
    task: &str,
    bc: &crate::config::BrowserConfig,
    ws_url: &str,
) -> Result<Vec<String>> {
    ensure_server(bc).await.context("couldn't reach the Project-Vision server")?;
    let target = ensure_chromium(bc).await?;
    let mut browser = Browser::attach(bc, target).await?;

    let (mut ws, _) = tokio_tungstenite::connect_async(ws_url)
        .await
        .with_context(|| format!("connect to Project-Vision server at {ws_url}"))?;

    let request_id = uuid::Uuid::new_v4().to_string();
    let start_tab = browser.sessions.first().map(|s| s.tab_id).unwrap_or(1);
    let dpr = browser
        .current()
        .await?
        .eval("window.devicePixelRatio")
        .await
        .and_then(|v| v.as_f64().ok_or_else(|| anyhow!("no dpr")))
        .unwrap_or(1.0);

    // Phase 1: announce the goal, then hand over the current view.
    ws.send(Message::text(
        json!({ "type": "USER_PROMPT", "request_id": request_id, "prompt": task }).to_string(),
    ))
    .await
    .context("send USER_PROMPT")?;

    let initial_shot = browser.current().await?.screenshot().await?;
    if initial_shot.is_empty() {
        bail!("couldn't capture the initial browser screenshot");
    }
    ws.send(Message::text(
        json!({
            "type": "RAW_SCREENSHOT",
            "request_id": request_id,
            "tab_id": start_tab,
            "step_index": 0,
            "image": initial_shot,
            "action_result": null,
            "device_pixel_ratio": dpr,
        })
        .to_string(),
    ))
    .await
    .context("send initial screenshot")?;

    // Phase 2: drive the action loop until the server goes quiet.
    let mut history: Vec<String> = Vec::new();
    let mut last_was_last = false;

    loop {
        let frame = tokio::time::timeout(Duration::from_secs(20), ws.next()).await;
        match frame {
            Ok(Some(Ok(Message::Text(text)))) => {
                let msg: Value =
                    serde_json::from_str(&text).context("parse server message")?;
                match msg["type"].as_str() {
                    Some("AGENT_ACTION") => {
                        last_was_last = false;
                        let action = msg
                            .get("action")
                            .cloned()
                            .unwrap_or(Value::Null);
                        let step_index = msg["step_index"].as_u64().unwrap_or(0);
                        let server_tab = msg["tab_id"].as_u64().unwrap_or(start_tab);
                        let is_last = msg["is_last_step"].as_bool().unwrap_or(false);
                        let action_name = action["action"].as_str().unwrap_or("?").to_string();

                        let step = match execute_action(&mut browser, &action, step_index, server_tab).await {
                            Ok(a) => {
                                history.push(describe_action(&a, &action));
                                a
                            }
                            Err(e) => {
                                tracing::warn!("execute_action error: {e:#}");
                                ActionResult {
                                    success: false,
                                    action: action_name,
                                    step_index,
                                    tab_id: server_tab,
                                    error: Some(format!("{e:#}")),
                                }
                            }
                        };

                        let request_id2 = request_id.clone();
                        let action_id = msg["action_id"].as_str().unwrap_or("").to_string();

                        if !is_last {
                            let shot = match browser.current().await {
                                Ok(s) => s.screenshot().await.unwrap_or_default(),
                                Err(_) => String::new(),
                            };
                            let tab_now = browser.sessions.first().map(|s| s.tab_id).unwrap_or(step.tab_id);
                            ws.send(Message::text(
                                json!({
                                    "type": "RAW_SCREENSHOT",
                                    "request_id": request_id2,
                                    "tab_id": tab_now,
                                    "step_index": step_index,
                                    "image": shot,
                                    "action_result": step,
                                    "device_pixel_ratio": dpr,
                                })
                                .to_string(),
                            ))
                            .await
                            .context("send post-action screenshot")?;
                        } else {
                            last_was_last = true;
                            ws.send(Message::text(
                                json!({
                                    "type": "ACTION_RESULT",
                                    "request_id": request_id2,
                                    "action_id": action_id,
                                    "result": step,
                                })
                                .to_string(),
                            ))
                            .await
                            .context("send final ACTION_RESULT")?;
                        }
                    }
                    Some("ERROR") => {
                        let err = msg["error"].as_str().unwrap_or("server error");
                        bail!("Project-Vision server error: {err}");
                    }
                    Some("QUEUE_STATUS") => {
                        let status = msg["status"].as_str().unwrap_or("?");
                        tracing::info!("server queue: {status}");
                    }
                    _ => {
                        tracing::debug!("ignoring server message: {}", msg["type"]);
                    }
                }
            }
            Ok(Some(Ok(_))) => {}
            Ok(Some(Err(e))) => bail!("server websocket error: {e}"),
            Ok(None) => break,
            Err(_elapsed) => {
                // Silence after the last step means the task is complete.
                if last_was_last {
                    break;
                }
                bail!("no response from the Project-Vision server for 20s");
            }
        }
    }

    Ok(history)
}

fn describe_action(r: &ActionResult, action: &Value) -> String {
    let base = match r.action.as_str() {
        "click" => "click".to_string(),
        "type" => format!("type: {}", action["text"].as_str().unwrap_or("")),
        "press" => format!("press {}", action["key"].as_str().unwrap_or("Enter")),
        "scroll" => format!("scroll {}", action["direction"].as_str().unwrap_or("")),
        "navigate" => format!("open {}", action["url"].as_str().unwrap_or("")),
        "search" => format!("search for \"{}\"", action["query"].as_str().unwrap_or("")),
        "open_tab" => format!("open tab {}", action["url"].as_str().unwrap_or("")),
        "switch_tab" => "switch tab".to_string(),
        "close_tab" => "close tab".to_string(),
        other => other.to_string(),
    };
    if r.success {
        format!("  ✔ {base}")
    } else {
        format!(
            "  ✘ {base} — {}",
            r.error.as_deref().unwrap_or("failed")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_srijan_isolation() {
        let (http, ws) = split_srijan("ws://127.0.0.1:8001/ws");
        assert_eq!(http, "http://127.0.0.1:8001");
        assert_eq!(ws, "ws://127.0.0.1:8001/ws");

        let (http, ws) = split_srijan("ws://localhost:9000");
        assert_eq!(http, "http://localhost:9000");
        assert_eq!(ws, "ws://localhost:9000");
    }

    #[test]
    fn percent_encode_is_rfc3986() {
        assert_eq!(percent_encode("ps4 on amazon.com"), "ps4%20on%20amazon.com");
        assert_eq!(
            percent_encode("add to cart & buy 'now'"),
            "add%20to%20cart%20%26%20buy%20%27now%27"
        );
        assert_eq!(percent_encode("café"), "caf%C3%A9");
    }

    #[test]
    fn pick_page_target_prefers_real_pages() {
        let targets = json!([
            { "type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://a" },
            { "type": "page", "url": "https://example.com/", "webSocketDebuggerUrl": "ws://b" },
            { "type": "other", "url": "anything", "webSocketDebuggerUrl": "ws://c" }
        ]);
        let picked = pick_page_target(&targets).unwrap();
        assert_eq!(picked["url"], "https://example.com/");

        let none = json!([{ "type": "browser", "webSocketDebuggerUrl": "ws://x" }]);
        assert!(pick_page_target(&none).is_none());
    }
}