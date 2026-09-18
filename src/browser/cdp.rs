//! CDP client using chromiumoxide — native async DevTools Protocol driver.

use anyhow::{anyhow, Context, Result};
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

impl CdpBrowser {
    /// Launch or connect to a Chromium instance with a persistent profile.
    pub async fn new(config: &crate::config::BrowserConfig) -> Result<Self> {
        let profile = profile_path(config);
        std::fs::create_dir_all(&profile)?;

        // Build chromiumoxide config
        let mut builder = BrowserConfig::builder()
            .port(config.cdp_port)
            .user_data_dir(&profile)
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--disable-background-networking")
            .arg("--disable-component-update")
            .arg("--password-store=basic")
            .arg("--window-size=1366,900")
            .launch_timeout(Duration::from_secs(60));

        if config.headless {
            builder = builder.headless_mode(HeadlessMode::New);
        } else {
            builder = builder.with_head();
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

        // Ensure we have at least one page
        let pages = browser.pages().await.context("get pages")?;
        let page = match pages.into_iter().next() {
            Some(p) => Some(p),
            None => {
                browser
                    .new_page("about:blank")
                    .await
                    .context("create initial page")?
                    .into()
            }
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

    /// Evaluate JavaScript in the page context
    pub async fn eval(&self, expression: &str) -> Result<Value> {
        let page = self.current_page().await?;
        let result = page
            .evaluate(expression)
            .await
            .context("evaluate js")?;
        Ok(result.object().value.clone().unwrap_or(Value::Null))
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
    // Tests require a running Chromium — skipped by default
}