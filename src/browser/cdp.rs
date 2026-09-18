//! CDP client using chromiumoxide — native async DevTools Protocol driver.

use anyhow::{anyhow, Context, Result};
use chromiumoxide::{
    Browser, BrowserConfig,
    browser::HeadlessMode,
    js::Evaluation,
    page::ScreenshotParams,
};
use chromiumoxide_cdp::cdp::browser_protocol::page::NavigateParams;
use futures::StreamExt;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

/// Persistent Chromium instance with a dedicated profile.
pub struct CdpBrowser {
    browser: Browser,
    _handler_task: tokio::task::JoinHandle<()>,
}

impl CdpBrowser {
    /// Launch or connect to a Chromium instance with a persistent profile.
    pub async fn new(config: &crate::config::BrowserConfig) -> Result<Self> {
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
        if pages.is_empty() {
            let _ = browser.new_page("about:blank").await.context("create initial page")?;
        }

        Ok(Self {
            browser,
            _handler_task: handler_task,
        })
    }

    /// Get the current active page (first non-blank page)
    pub async fn current_page(&self) -> Result<chromiumoxide::Page> {
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
            .evaluate(expression)  // Evaluation implements From<&str>
            .await
            .context("evaluate js")?;
        Ok(result.object().value.clone().unwrap_or(Value::Null))
    }

    /// Take a screenshot as base64 PNG
    pub async fn screenshot(&self) -> Result<String> {
        let page = self.current_page().await?;
        let data = page
            .screenshot(ScreenshotParams::default())
            .await
            .context("capture screenshot")?;
        Ok(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, data))
    }

    /// Get the underlying browser for advanced operations
    pub fn browser(&self) -> &Browser {
        &self.browser
    }
}

/// Ensure a debuggable Chromium is running, return a connected CdpBrowser.
pub async fn ensure_browser(config: &crate::config::BrowserConfig) -> Result<CdpBrowser> {
    // Kill any existing process on the CDP port to avoid conflicts with user's browser
    let _ = tokio::process::Command::new("fuser")
        .arg("-k")
        .arg(format!("{}/tcp", config.cdp_port))
        .output()
        .await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Always launch our own isolated instance with dedicated profile
    tracing::info!("launching isolated Chromium on port {} for automation", config.cdp_port);
    CdpBrowser::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    // Tests require a running Chromium — skipped by default
}