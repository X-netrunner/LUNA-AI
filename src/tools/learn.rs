//! tools/learn.rs — One-shot web learning: search + fetch combined
//!
//! Collapses the web_search -> fetch_page two-step chain (which models
//! often botch — wrong URL, forgetting the second call) into a single
//! reliable tool call. Returns combined text for the model to read
//! and then save with `remember` if it's worth keeping.

use super::web;
use anyhow::{Context, Result};
use serde_json::json;

pub async fn learn(
    topic: &str,
    _sudo_pass: Option<&str>,
    tavily_key: Option<&str>,
    gemini_key: Option<&str>,
) -> Result<String> {
    let mut output = String::new();

    // Step 1: search
    let search_result = web::search(topic, tavily_key, gemini_key)
        .await
        .unwrap_or_else(|e| format!("Search failed: {}", e));
    output.push_str("=== Search results ===\n");
    output.push_str(&search_result);
    output.push_str("\n\n");

    // Step 2: if the search gave us a source URL, fetch the full page — but
    // skip junk hosts (social feeds, dictionary definitions) whose scrapes
    // are pure noise, e.g. Instagram returns a login/caption dump.
    let url = pick_fetch_url(&search_result);

    if let Some(url) = url {
        output.push_str(&format!("=== Fetched page: {} ===\n", url));
        let page = fetch_page_firecrawl(&url).await.unwrap_or_else(|e| {
            format!("(Firecrawl failed: {} — try fetch_page directly)", e)
        });
        output.push_str(&page);
    } else {
        output.push_str(
            "(No source URL in search results — the summary above is all that's available. \
             If you know a specific URL, use fetch_page directly.)",
        );
    }

    Ok(output)
}

/// Pick the page to fetch: prefer an explicit `Source:` line, else the first
/// web-result link whose host isn't a social feed or dictionary.
fn pick_fetch_url(text: &str) -> Option<String> {
    if let Some(url) = extract_url(text) {
        if !is_junk_fetch(&url) {
            return Some(url);
        }
    }
    text.lines()
        .filter(|l| l.trim_start().starts_with("- "))
        .find_map(|l| {
            let start = l.rfind('(')?;
            let end = l.rfind(')')?;
            let url = &l[start + 1..end];
            if url.starts_with("http") && !is_junk_fetch(url) {
                Some(url.to_string())
            } else {
                None
            }
        })
}

fn is_junk_fetch(url: &str) -> bool {
    const BAD: &[&str] = &[
        "instagram.com",
        "facebook.com",
        "tiktok.com",
        "pinterest.",
        "twitter.com",
        "x.com/",
        "youtube.com",
        "dictionary.cambridge.org",
        "merriam-webster.com",
        "wiktionary.org",
    ];
    BAD.iter().any(|b| url.contains(b))
}

/// Fetch a page via Firecrawl keyless — returns clean markdown
async fn fetch_page_firecrawl(url: &str) -> Result<String> {
    let body = json!({ "url": url });
    let body_str = body.to_string();

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time", "30",
                "-H", "Content-Type: application/json",
                "-d", &body_str,
                "https://api.firecrawl.dev/v1/scrape",
            ])
            .output()
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    let resp: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout))
            .context("Failed to parse Firecrawl response")?;

    if let Some(err) = resp.get("error") {
        anyhow::bail!("Firecrawl: {}", err);
    }

    let markdown = resp["data"]["markdown"]
        .as_str()
        .context("No markdown in Firecrawl response")?;

    // Truncate to 3000 chars
    let truncated: String = markdown.chars().take(3000).collect();
    Ok(truncated)
}

fn extract_url(text: &str) -> Option<String> {
    text.lines()
        .find(|l| l.starts_with("Source:"))
        .map(|l| l.trim_start_matches("Source:").trim().to_string())
}
