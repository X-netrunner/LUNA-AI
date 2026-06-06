//! tools/web.rs — Internet search via DuckDuckGo Instant Answer API
//!
//! Uses DDG's free JSON API — no key required.
//! For richer results it also scrapes the first few web hits via curl.

use anyhow::{Context, Result};

/// Search the web and return a plain-text summary of top results.
pub async fn search(query: &str) -> Result<String> {
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
    parse_ddg_response(&body, query)
}

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
            let truncated = if text.len() > 800 {
                format!("{}...", &text[..800])
            } else {
                text.to_string()
            };
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
            .filter_map(|t| t["Text"].as_str())
            .filter(|t| !t.is_empty())
            .take(5)
            .map(|t| format!("- {}", t))
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
