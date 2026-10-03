//! CDP client using chromiumoxide — native async DevTools Protocol driver.

use anyhow::{anyhow, Context, Result};
use chromiumoxide::page::ScreenshotParams;
use chromiumoxide::{Browser, BrowserConfig, browser::HeadlessMode};
use chromiumoxide_cdp::cdp::browser_protocol::page::NavigateParams;
use futures::StreamExt;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

/// Persistent Chromium instance with a dedicated profile.
pub struct CdpBrowser {
    browser: Browser,
    /// The page owned by this session (created by `new_page` on reuse, or the
    /// initial tab on a fresh launch). `current_page` returns this handle
    /// directly instead of scanning all tabs — a scan can latch onto a tab
    /// that pre-dates this CDP connection, and navigation on such stale tabs
    /// never completes (chromiumoxide `CdpError::Timeout`).
    page: Option<chromiumoxide::Page>,
    _handler_task: tokio::task::JoinHandle<()>,
}

/// The dedicated profile directory used for automation — mirrors
/// [`CdpBrowser::new`] so `ensure_browser` can target the same instance.
fn profile_path(config: &crate::config::BrowserConfig) -> PathBuf {
    if config.profile_dir.trim().is_empty() {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("luna")
            .join("browser-profile")
    } else {
        PathBuf::from(&config.profile_dir)
    }
}

/// True when some live process was launched with `--user-data-dir=<profile>`
/// (i.e. one of our automation Chromiums is still running).
fn process_uses_profile(profile: &PathBuf) -> bool {
    let needle = format!("--user-data-dir={}", profile.display());
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return false;
    };
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let Ok(raw) = std::fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        if raw.windows(needle.len()).any(|w| w == needle.as_bytes()) {
            return true;
        }
    }
    false
}

/// Hard-kill any leftover automation Chromium (by profile dir and by CDP port)
/// and purge the stale singleton lock files it leaves behind.
async fn kill_leftovers(config: &crate::config::BrowserConfig) {
    let profile = profile_path(config);

    // 1. Kill processes bound to our CDP port.
    let _ = tokio::process::Command::new("fuser")
        .arg("-k")
        .arg(format!("{}/tcp", config.cdp_port))
        .output()
        .await;
    // 2. Kill by profile flag too — catches zombies that no longer hold the port.
    //    The `--` separator is required: procps-ng would otherwise parse the
    //    `--user-data-dir=...` pattern as its own (invalid) option.
    let _ = tokio::process::Command::new("pkill")
        .arg("-KILL")
        .arg("-f")
        .arg("--")
        .arg(format!("--user-data-dir={}", profile.display()))
        .output()
        .await;
    tokio::time::sleep(Duration::from_millis(600)).await;

    // 3. Remove stale singleton locks. Chromium writes these on start and only
    //    clears them on a clean exit; after a SIGKILL they outlive the process
    //    and make the next launch report "profile is in use by another process".
    for name in ["SingletonLock", "SingletonSocket", "SingletonCookie"] {
        let _ = std::fs::remove_file(profile.join(name));
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
}

/// Parse a "WxH" config value (e.g. "1920x1200") into (width, height).
fn parse_window_size(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (w, h) = s.split_once(['x', 'X'])?;
    let w: u32 = w.trim().parse().ok()?;
    let h: u32 = h.trim().parse().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

/// Best-effort primary-screen resolution so the automation window fills the
/// display. Tries Hyprland first (its JSON exposes the focused monitor), then
/// X11. Returns None when no tool answers.
async fn detect_screen_size() -> Option<(u32, u32)> {
    // Hyprland: `hyprctl monitors -j` -> [{ width, height, focused, ... }]
    if let Ok(out) = tokio::process::Command::new("hyprctl")
        .args(["monitors", "-j"])
        .output()
        .await
    {
        if out.status.success() {
            if let Ok(list) = serde_json::from_slice::<Vec<serde_json::Value>>(&out.stdout) {
                let mon = list
                    .iter()
                    .find(|m| m.get("focused").and_then(|v| v.as_bool()).unwrap_or(false))
                    .or_else(|| list.first());
                if let Some(mon) = mon {
                    let w = mon.get("width").and_then(|v| v.as_u64())?;
                    let h = mon.get("height").and_then(|v| v.as_u64())?;
                    if w > 0 && h > 0 {
                        return Some((w as u32, h as u32));
                    }
                }
            }
        }
    }

    // X11 fallback: first "connected" line containing a WxH token.
    if let Ok(out) = tokio::process::Command::new("xrandr")
        .arg("--current")
        .output()
        .await
    {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if !line.contains("connected") {
                continue;
            }
            for tok in line.split_whitespace() {
                if let Some((w, h)) = tok.split_once('x') {
                    if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                        if w > 0 && h > 0 {
                            return Some((w, h));
                        }
                    }
                }
            }
        }
    }

    None
}

/// What the page copy says, split by the remedy each implies.
///
/// Checked bot-first: a bot wall is sometimes dressed in login wording ("log in
/// to buy"), and telling the user to log in when the site is refusing
/// automation sends them to do something that will not work.
///
/// Every `bot` marker is a *phrase*, never the single distinctive word. Bare
/// "not a robot" would fire on a product page for the book or film of that
/// name, and the response to a false positive here is "stop using this site" —
/// so the marker set is biased toward the longer wording that only a real
/// challenge produces.
#[derive(serde::Deserialize)]
struct WallMarkers {
    #[serde(default)]
    login: Vec<String>,
    #[serde(default)]
    bot: Vec<String>,
}

impl CdpBrowser {
    /// Launch or connect to a Chromium instance with a persistent profile.
    pub async fn new(config: &crate::config::BrowserConfig) -> Result<Self> {
        let profile = profile_path(config);
        std::fs::create_dir_all(&profile)?;

        // Size the window to the primary display (or an explicit WxH override)
        // instead of a hardcoded 1366x900 that looks "weird" on other screens.
        let (w, h) = match parse_window_size(&config.window_size) {
            Some(size) => size,
            None => detect_screen_size().await.unwrap_or((1366, 900)),
        };

        // Build chromiumoxide config
        let mut builder = BrowserConfig::builder()
            .port(config.cdp_port)
            .user_data_dir(&profile)
            // chromiumoxide defaults to an emulated 800x600 viewport via
            // Emulation.setDeviceMetricsOverride, which letterboxes the page
            // in the top-left of a full-size window ("weird resolution").
            // Disable emulation so pages use the real window/display size.
            .viewport(None)
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--disable-background-networking")
            .arg("--disable-component-update")
            .arg("--password-store=basic")
            .arg(format!("--window-size={w},{h}"))
            .launch_timeout(Duration::from_secs(60));

        if config.headless {
            builder = builder.headless_mode(HeadlessMode::New);
        } else {
            // Don't let Chromium open its own initial "new tab" — luna owns the
            // only tab in the window. Without this, Chromium's initial tab plus
            // our new_page() produced two tabs on every fresh launch.
            builder = builder.with_head().arg("--no-startup-window");
        }

        let browser_config = builder
            .build()
            .map_err(|e| anyhow!("build chromiumoxide config: {}", e))?;

        let (browser, mut handler) = Browser::launch(browser_config)
            .await
            .context("launch chromiumoxide browser")?;

        // Run the CDP event loop in background by polling handler.next()
        let handler_task = tokio::spawn(async move {
            while let Some(h) = handler.next().await {
                if h.is_err() {
                    break;
                }
            }
        });

        // With --no-startup-window visible launches have no initial tabs, so we
        // create exactly one. Headless keeps the old fallback: reuse Chromium's
        // initial tab if it surfaces, otherwise create one.
        let page = if config.headless {
            let pages = browser.pages().await.context("get pages")?;
            let mut chosen = None;
            for p in pages {
                if let Ok(Some(url)) = p.url().await {
                    if url != "about:blank"
                        && !url.is_empty()
                        && !url.starts_with("chrome://")
                        && !url.starts_with("chrome-extension://")
                    {
                        chosen = Some(p);
                        break;
                    }
                }
            }
            match chosen {
                Some(p) => Some(p),
                None => browser
                    .new_page("about:blank")
                    .await
                    .context("create initial page")?
                    .into(),
            }
        } else {
            browser
                .new_page("about:blank")
                .await
                .context("create initial page")?
                .into()
        };

        Ok(Self {
            browser,
            page,
            _handler_task: handler_task,
        })
    }

    /// Leave the Automation Chromium running after this handle goes away.
    ///
    /// chromiumoxide's launched `Browser` kills its child on drop
    /// (`kill_on_drop`), which closes the window at the end of a task.
    /// Forgetting the handle keeps the process alive so the user can inspect
    /// the final state; the next task picks it up again via
    /// [`ensure_browser`]'s connect path.
    pub fn detach(self) {
        std::mem::forget(self);
    }

    /// Get the current active page.
    ///
    /// Prefers the page owned by this session (created via `new_page` on the
    /// reuse path, or the initial tab on a fresh launch). Falls back to a scan
    /// of all tabs only when no handle is held — the scan is unsafe on a
    /// reused connection because tabs that pre-dated the connection register
    /// as pages yet never complete navigation.
    pub async fn current_page(&self) -> Result<chromiumoxide::Page> {
        if let Some(page) = &self.page {
            return Ok(page.clone());
        }
        let pages = self.browser.pages().await.context("get pages")?;
        // Prefer the first non-about:blank page
        for page in &pages {
            if let Ok(Some(url)) = page.url().await {
                if !url.is_empty() && url != "about:blank" && !url.starts_with("chrome://") {
                    return Ok(page.clone());
                }
            }
        }
        // Fallback to first page
        pages
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no page available"))
    }

    /// Navigate to a URL and wait for load
    pub async fn navigate(&self, url: &str) -> Result<()> {
        let page = self.current_page().await?;
        page.goto(NavigateParams::new(url)).await.context("navigate")?;
        page.wait_for_navigation().await.context("wait for navigation")?;
        Ok(())
    }

    /// Evaluate JS, surviving a navigation that happened underneath us.
    ///
    /// A click that follows a link destroys the execution context, so the next
    /// evaluate fails for a reason that has nothing to do with the expression.
    /// Waiting for the new document and trying again turns a crash into the
    /// answer, and only on that specific error — a genuine JS error is still
    /// returned so it can be reported rather than retried into a timeout.
    pub async fn eval_resilient(&self, expression: &str) -> Result<Value> {
        match self.eval(expression).await {
            Err(e) if Self::is_context_lost(&e) => {
                self.wait_until_ready(Duration::from_millis(6000)).await?;
                self.eval(expression).await
            }
            other => other,
        }
    }

    /// Wait for a document to finish loading, bounded.
    ///
    /// `readyState` is the cheapest honest signal there is: it is the page's own
    /// statement about whether it can be queried yet, rather than a sleep
    /// guessed to be long enough.
    pub async fn wait_until_ready(&self, budget: Duration) -> Result<()> {
        let started = std::time::Instant::now();
        loop {
            let state = self
                .eval("document.readyState")
                .await
                .ok()
                .and_then(|v| v.as_str().map(|s| s.to_string()));
            match state.as_deref() {
                Some("interactive") | Some("complete") => return Ok(()),
                _ if started.elapsed() >= budget => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(200)).await,
            }
        }
    }
    /// Did this error come from the page navigating out from under us?
    ///
    /// Clicking a product link tears down the JS execution context. The next
    /// `Runtime.evaluate` then fails with "Cannot find context with specified id",
    /// which is not a failure of the thing being asked — it is the previous step
    /// having worked. Measured 2026-10-03: `pick_best` clicked through to a product
    /// page and the following `click: Add to Cart` died on exactly this, so a
    /// working step was reported as a crash.
    pub fn is_context_lost(err: &anyhow::Error) -> bool {
        let m = err.to_string();
        [
            "Cannot find context",
            "Execution context was destroyed",
            "No execution context",
            "Cannot find context with specified id",
            "Inspected target navigated or closed",
        ]
        .iter()
        .any(|k| m.contains(k))
    }

/// The URL the page is actually on right now.
    ///
    /// Needed because a navigate can fail to arrive: a login wall, a redirect,
    /// or a stale tab, and every one of those still leaves a page to report
    /// success from. Anything that claims to have gone somewhere has to be able
    /// to check where it went.
    pub async fn current_url(&self) -> Result<String> {
        let page = self.current_page().await?;
        page.url()
            .await
            .context("read current url")?
            .ok_or_else(|| anyhow!("page has no url"))
    }

    /// Evaluate JavaScript in the page context
    pub async fn eval(&self, expression: &str) -> Result<Value> {
        let page = self.current_page().await?;
        let result = page
            .evaluate(expression)
            .await
            .context("evaluate js")?;
        Ok(result.object().value.clone().unwrap_or(Value::Null))
    }

    /// Evaluate JS, surviving a navigation that happened underneath us.
    ///
    /// A click that follows a link destroys the execution context, so the next
    /// evaluate fails for a reason that has nothing to do with the expression.
    /// Detect a login wall that blocks a purchase — a sign-in page or a strong
    /// "you must log in to continue" prompt. Returns a short human reason.
    ///
    /// Only STRONG signals count (sign-in URL paths, explicit login-required
    /// copy, anti-bot challenge URLs), so a site's harmless "Sign in" header
    /// link never trips it.
    ///
    /// Why this page cannot be used as the one that was asked for.
    ///
    /// Two kinds of wall, which need different remedies and are easy to
    /// confuse. A login wall needs the user to log in; a bot wall needs the
    /// automation to *stop*, because retrying is what extends the block. On
    /// 2026-10-03 Amazon served a bot interstitial and the run carried on
    /// reading it as a product page.
    pub async fn page_wall_reason(&self) -> Option<String> {
        let page = self.current_page().await.ok()?;

        // 1. The browser landed on a sign-in page.
        if let Ok(Some(url)) = page.url().await {
            let lower = url.to_lowercase();
            let login_paths = [
                "/login", "/login?", "/accounts/login", "/account/login", "/signin",
                "/sign-in", "/sign_in", "/ap/signin", "/identity/login",
            ];
            if login_paths.iter().any(|p| lower.contains(p)) {
                return Some(format!(
                    "the browser is on a sign-in page ({url}) and cannot continue \
                     without logging into an account"
                ));
            }

            // 2. The URL says this is an anti-bot challenge. Distinct from a
            //    login wall and worth separating from one: signing in does not
            //    clear it, and reloading does not either.
            for marker in [
                "validatecaptcha",
                "/captcha",
                "/errors/validate",
                "challenge-form",
                "px-captcha",
                "unusual traffic",
                "are you a robot",
                "/bot-detect",
            ] {
                if lower.contains(marker) {
                    return Some(format!(
                        "the site served an anti-bot challenge ({url}) — it is blocking \
                         automated access, not asking for a login. Stop navigating \
                         this site; further requests are what deepen the block"
                    ));
                }
            }
        }

        // 3. Copy that means the same thing without saying it in the URL.
        let expr = r#"(() => {
            const login = [
                "log in to continue", "login to continue", "please sign in",
                "please login", "sign in to continue", "you must be logged in",
                "you need to log in", "you need to login", "login required",
                "sign in or register to continue", "continue with login",
                "log in to buy", "login to buy", "login to place your order",
                "login to add", "please sign in to continue"
            ];
            const bot = [
                "make sure you're not a robot",
                "enter the characters you see below",
                "type the characters you see",
                "sorry, we just need to make sure",
                "automated access to amazon",
                "discuss automated access",
                "unusual traffic from your computer",
                "verify you are human",
                "checking your browser before accessing",
                "enable javascript and cookies to continue"
            ];
            const t = ((document.body && document.body.innerText) || "").toLowerCase();
            return {
                login: login.filter(m => t.includes(m)),
                bot: bot.filter(m => t.includes(m))
            };
        })()"#;
        if let Ok(res) = page.evaluate(expr).await {
            if let Ok(found) = res.into_value::<WallMarkers>() {
                if !found.bot.is_empty() {
                    return Some(format!(
                        "the page is an anti-bot check — it shows \"{}\" and the site \
                         is refusing automated access. Stop; repeated requests make \
                         the block worse",
                        found.bot[0]
                    ));
                }
                if !found.login.is_empty() {
                    return Some(format!(
                        "the page requires an account — it shows \"{}\" and the \
                         purchase cannot continue without logging in",
                        found.login[0]
                    ));
                }
            }
        }
        None
    }

    /// Current page's URL + title — fed back into the tool result so the model
    /// reports what is ACTUALLY on screen instead of guessing.
    pub async fn current_url_title(&self) -> (Option<String>, Option<String>) {
        let page = match self.current_page().await {
            Ok(p) => p,
            Err(_) => return (None, None),
        };
        let url = page.url().await.ok().flatten();
        let title = page.get_title().await.ok().flatten();
        (url, title)
    }

    /// Capture the current page as PNG bytes — Luna's eyes for the browser.
    pub async fn screenshot_png(&self) -> Result<Vec<u8>> {
        let page = self.current_page().await?;
        let png = page
            .screenshot(ScreenshotParams::default())
            .await
            .context("page screenshot")?;
        Ok(png)
    }
}

/// Ensure our automation Chromium is running, return a connected CdpBrowser.
///
/// Reuses an already-running automation instance (same profile + port) so
/// repeated tasks don't spawn a second window; otherwise it kills any leftovers
/// and launches a fresh one.
pub async fn ensure_browser(config: &crate::config::BrowserConfig) -> Result<CdpBrowser> {
    // Reuse: is one of our automation Chromiums (same profile dir) still alive?
    let alive_profile = process_uses_profile(&profile_path(config));
    tracing::debug!("ensure_browser: process_uses_profile={alive_profile}");
    if alive_profile {
        let cdp_url = format!("http://127.0.0.1:{}", config.cdp_port);
        match Browser::connect(&cdp_url).await {
            Ok((mut browser, mut handler)) => {
                tokio::spawn(async move {
                    while let Some(h) = handler.next().await {
                        if h.is_err() {
                            break;
                        }
                    }
                });
                // A freshly connected client does NOT get `targetCreated`
                // events replayed for tabs that already exist, so the
                // handler's page registry is empty and `pages()` returns [].
                // Actively fetch current targets to populate it.
                if browser.fetch_targets().await.is_ok() {
                    // Close leftovers from the previous task so the window
                    // stays clean and current_page() never latches onto a
                    // stale, pre-connection tab (navigation on such tabs never
                    // completes — CdpError::Timeout). pages() races target
                    // init after connect, so poll briefly until the stale tabs
                    // are visible before closing them.
                    for _ in 0..15 {
                        match browser.pages().await {
                            Ok(pages) if !pages.is_empty() => {
                                for page in pages {
                                    let _ = page.close().await;
                                }
                                break;
                            }
                            Ok(_) => {
                                tokio::time::sleep(Duration::from_millis(200)).await;
                            }
                            Err(_) => break,
                        }
                    }
                    // Open a fresh page — the same proven path as
                    // CdpBrowser::new. new_page() returns only once the page
                    // is fully initialized, so steps won't hit a timeout. We
                    // hold its handle so current_page() uses OUR page, never
                    // a stale tab.
                    match browser.new_page("about:blank").await {
                        Ok(page) => {
                            tracing::info!(
                                "reusing existing automation Chromium on port {}",
                                config.cdp_port
                            );
                            return Ok(CdpBrowser {
                                browser,
                                page: Some(page),
                                _handler_task: tokio::spawn(async {}),
                            });
                        }
                        Err(e) => {
                            tracing::debug!(
                                "ensure_browser: new_page on reused browser failed: {e}, \
                                 relaunching"
                            );
                        }
                    }
                } else {
                    tracing::debug!("ensure_browser: fetch_targets failed, relaunching");
                }
            }
            Err(e) => {
                tracing::debug!("ensure_browser: connect failed: {e}, relaunching");
            }
        }
    }

    // No healthy instance — clear stale processes/locks, then launch fresh.
    kill_leftovers(config).await;
    tracing::info!("launching isolated Chromium on port {} for automation", config.cdp_port);
    CdpBrowser::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error text from the 2026-10-03 run, which killed a working
    /// `click: Add to Cart` immediately after `pick_best` had done its job.
    #[test]
    fn a_navigation_is_recognised_as_a_navigation_not_a_failure() {
        let lost = anyhow::anyhow!(
            "evaluate js: Error -32000: Cannot find context with specified id"
        );
        assert!(CdpBrowser::is_context_lost(&lost));
        for text in [
            "Execution context was destroyed",
            "No execution context",
            "Inspected target navigated or closed",
        ] {
            assert!(CdpBrowser::is_context_lost(&anyhow::anyhow!(text)), "{text}");
        }
    }

    /// The retry must be narrow. Retrying a genuine JS error would turn a real
    /// failure into a timeout and hide it.
    #[test]
    fn a_real_javascript_error_is_not_retried() {
        for text in [
            "evaluate js: Error -32000: Cannot read properties of null",
            "SyntaxError: Unexpected token",
            "navigate: net::ERR_NAME_NOT_RESOLVED",
        ] {
            assert!(!CdpBrowser::is_context_lost(&anyhow::anyhow!(text)), "{text}");
        }
    }
    use super::*;
    // Tests require a running Chromium — skipped by default
}