//! tools/web.rs — Web search via Gemini API or DuckDuckGo instant answers
//!
//! Gemini free tier (15 RPM, 1M tokens/day) answers questions directly.
//! Falls back to DDG instant answers when no API key is configured.

use anyhow::{Context, Result};
use serde_json::json;

/// Search the web and return a plain-text summary of top results.
/// If `gemini_key` is provided, sends the question to Gemini which can
/// answer directly from its training data + Google Search grounding.
/// Otherwise falls back to DDG instant answers (limited).
pub async fn search(query: &str, gemini_key: Option<&str>) -> Result<String> {
    // Gemini API — can answer real-time questions directly
    if let Some(key) = gemini_key {
        return search_gemini(query, key).await;
    }

    let encoded = urlencoding(query);

    // DuckDuckGo Instant Answer API — free, no auth, but limited
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

/// Ask Gemini a question — free tier, no scraping, answers directly
async fn search_gemini(query: &str, api_key: &str) -> Result<String> {
    let prompt = format!(
        "Answer this question concisely in 2-3 sentences. \
         If it's about current events, news, or real-time data (songs, weather, \
         stocks, sports scores), give the most recent answer you know. \
         If you're unsure about very recent data, say so. \
         Question: {}",
        query
    );

    let body = json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 300
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
        api_key
    );
    let body_str = body.to_string();

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time", "15",
                "-H", "Content-Type: application/json",
                "-d", &body_str,
                &url,
            ])
            .output()
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    if !output.status.success() {
        anyhow::bail!("Gemini API curl failed: {:?}", output.status.code());
    }

    let resp: serde_json::Value = serde_json::from_str(
        &String::from_utf8_lossy(&output.stdout),
    )
    .context("Failed to parse Gemini response")?;

    // Extract text from the response
    let text = resp["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .unwrap_or("(Gemini returned no text)");

    Ok(text.to_string())
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
