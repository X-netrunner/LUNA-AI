//! tools/web.rs — Internet search via DuckDuckGo Instant Answer API
//!
//! Uses DDG's free JSON API — no key required.
//! For richer results it also scrapes the first few web hits via curl.

use anyhow::{Context, Result};

/// Search the web and return a plain-text summary of top results.
/// If `brave_key` is provided, uses the Brave Search API (reliable, JSON).
/// Otherwise falls back to DDG instant answers + HTML scraping.
pub async fn search(query: &str, brave_key: Option<&str>) -> Result<String> {
    // Brave Search API — clean JSON, no CAPTCHAs, 2000 free queries/month
    if let Some(key) = brave_key {
        return search_brave_api(query, key).await;
    }

    let encoded = urlencoding(query);

    // DuckDuckGo Instant Answer API — completely free, no auth
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        encoded
    );

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time", "10",
                "--user-agent", "luna-assistant/1.0",
                &url,
            ])
            .output()
            .context("curl not found — install curl")
    })
    .await
    .context("spawn_blocking panicked")??;

    if !output.status.success() {
        anyhow::bail!("curl failed with exit code {:?}", output.status.code());
    }

    let body = String::from_utf8_lossy(&output.stdout);
    let result = parse_ddg_response(&body, query)?;

    // If instant answer was empty, no HTML fallback — Brave API is the fix
    if result.starts_with("No instant answer") {
        return Ok(result);
    }

    Ok(result)
}

/// Brave Search API — returns clean JSON with titles, URLs, and snippets
async fn search_brave_api(query: &str, api_key: &str) -> Result<String> {
    let encoded = urlencoding(query);
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count=5",
        encoded
    );
    let key = api_key.to_string();

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time", "10",
                "-H", "Accept: application/json",
                "-H", &format!("X-Subscription-Token: {}", key),
                &url,
            ])
            .output()
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    if !output.status.success() {
        anyhow::bail!("Brave API curl failed: {:?}", output.status.code());
    }

    let body = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&body)
        .context("Failed to parse Brave API response")?;

    let results = v["web"]["results"]
        .as_array()
        .context("No results array in Brave response")?;

    if results.is_empty() {
        return Ok(format!("No results found for \"{}\"", query));
    }

    let mut out = format!("Search results for \"{}\":\n", query);
    for r in results.iter().take(5) {
        let title = r["title"].as_str().unwrap_or("");
        let url = r["url"].as_str().unwrap_or("");
        let snippet = r["description"].as_str().unwrap_or("");
        if !title.is_empty() && !url.is_empty() {
            out.push_str(&format!("- {} ({})\n", title, url));
            if !snippet.is_empty() {
                let snip: String = snippet.chars().take(200).collect();
                out.push_str(&format!("  {}\n", snip));
            }
        }
    }
    Ok(out)
}

/// Fallback: scrape Brave Search results when DDG instant answer is empty
fn parse_ddg_response(body: &str, query: &str) -> Result<String> {
    // Parse just the fields we care about without pulling in serde_json
    // (it's already a dep, use it)
    let v: serde_json::Value = serde_json::from_str(body)
        .context("Failed to parse DDG response")?;

    let mut parts: Vec<String> = Vec::new();

    // Instant answer (calculator, conversions, definitions, etc.)
    if let Some(answer) = v["Answer"].as_str() {
        if !answer.is_empty() {
            parts.push(format!("Answer: {}", answer));
        }
    }

    // Abstract text (Wikipedia summary)
    if let Some(text) = v["AbstractText"].as_str() {
        if !text.is_empty() {
            let truncated = crate::util::truncate(text, 800);
            parts.push(format!("Summary: {}", truncated));
            if let Some(src) = v["AbstractURL"].as_str() {
                if !src.is_empty() {
                    parts.push(format!("Source: {}", src));
                }
            }
        }
    }

    // Related topics (top 5)
    if let Some(topics) = v["RelatedTopics"].as_array() {
        let related: Vec<String> = topics
            .iter()
            .filter_map(|t| {
                let text = t["Text"].as_str()?;
                if text.is_empty() {
                    return None;
                }
                let url = t["FirstURL"].as_str().unwrap_or("");
                if url.is_empty() {
                    Some(format!("- {}", text))
                } else {
                    Some(format!("- {} ({})", text, url))
                }
            })
            .take(5)
            .collect();
        if !related.is_empty() {
            parts.push(format!("Related:\n{}", related.join("\n")));
        }
    }

    if parts.is_empty() {
        Ok(format!(
            "No instant answer found for \"{}\". \
             Try a more specific query or use run_shell with `curl` to fetch a specific URL.",
            query
        ))
    } else {
        Ok(parts.join("\n\n"))
    }
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            c => format!("%{:02X}", c as u32),
        })
        .collect()
}
