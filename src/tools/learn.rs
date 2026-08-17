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

    // Step 2: if the search gave us a source URL, fetch the full page
    let url = extract_url(&search_result)
        .or_else(|| extract_first_link(&search_result));

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

/// Extract first URL from search results line like "- Title (https://...)"
fn extract_first_link(text: &str) -> Option<String> {
    text.lines()
        .find(|l| l.starts_with("- "))
        .and_then(|l| {
            let start = l.rfind('(')?;
            let end = l.rfind(')')?;
            let url = &l[start + 1..end];
            if url.starts_with("http") {
                Some(url.to_string())
            } else {
                None
            }
        })
}
