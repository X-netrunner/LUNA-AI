//! tools/web.rs — Web search via Tavily, Gemini, or DuckDuckGo
//!
//! Search priority:
//!   1. Tavily (keyless or keyed) — clean JSON results, purpose-built for AI
//!   2. Gemini (knowledge answers from training data, no web search needed)
//!   3. DDG instant answers (definitions, math, Wikipedia)

use anyhow::{Context, Result};
use serde_json::json;

/// Search the web. Tries Tavily first, then Gemini, then DDG.
pub async fn search(
    query: &str,
    tavily_key: Option<&str>,
    gemini_key: Option<&str>,
) -> Result<String> {
    // 1. Tavily — real web search, works keyless
    match search_tavily(query, tavily_key).await {
        Ok(r) if !r.starts_with("No results") => return Ok(r),
        Err(e) => tracing::warn!("Tavily search failed: {}", e),
        _ => {}
    }

    // 2. Gemini — knowledge answers (no grounding, no rate limit issues)
    if let Some(key) = gemini_key {
        match search_gemini(query, key).await {
            Ok(r) => return Ok(r),
            Err(e) => tracing::warn!("Gemini search failed: {}", e),
        }
    }

    // 3. DDG instant answers — limited but always free
    search_ddg(query).await
}

// ── Tavily ────────────────────────────────────────────────────────────────────

async fn search_tavily(query: &str, api_key: Option<&str>) -> Result<String> {
    let body = json!({
        "query": query,
        "max_results": 5,
        "search_depth": "basic",
        "include_answer": true,
    });
    let body_str = body.to_string();
    let auth_header = match api_key {
        Some(key) => format!("Authorization: Bearer {}", key),
        None => "X-Tavily-Access-Mode: keyless".to_string(),
    };

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time", "15",
                "-H", "Content-Type: application/json",
                "-H", &auth_header,
                "-d", &body_str,
                "https://api.tavily.com/search",
            ])
            .output()
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    if !output.status.success() {
        anyhow::bail!("Tavily curl failed: {:?}", output.status.code());
    }

    let resp: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&output.stdout))
        .context("Failed to parse Tavily response")?;

    if let Some(err) = resp.get("error") {
        anyhow::bail!("Tavily: {}", err);
    }

    let mut out = String::new();

    // Include the AI-generated answer if present
    if let Some(answer) = resp["answer"].as_str() {
        if !answer.is_empty() {
            out.push_str(answer);
            out.push_str("\n\n");
        }
    }

    // Web results
    if let Some(results) = resp["results"].as_array() {
        if results.is_empty() && out.is_empty() {
            return Ok(format!("No results found for \"{}\"", query));
        }
        for r in results.iter().take(5) {
            let title = r["title"].as_str().unwrap_or("");
            let url = r["url"].as_str().unwrap_or("");
            let content = r["content"].as_str().unwrap_or("");
            if !title.is_empty() {
                out.push_str(&format!("- {} ({})\n", title, url));
                if !content.is_empty() {
                    let snippet: String = content.chars().take(200).collect();
                    out.push_str(&format!("  {}\n", snippet));
                }
            }
        }
    }

    if out.trim().is_empty() {
        anyhow::bail!("Tavily returned empty results");
    }

    Ok(out)
}

// ── Gemini (knowledge only, no grounding) ─────────────────────────────────────

async fn search_gemini(query: &str, api_key: &str) -> Result<String> {
    let prompt = format!(
        "Answer this question concisely in 2-3 sentences: {}",
        query
    );

    let body = json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 500
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={}",
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

    let resp: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout))
            .context("Failed to parse Gemini response")?;

    if let Some(err) = resp.get("error") {
        let msg = err["message"].as_str().unwrap_or("unknown error");
        anyhow::bail!("Gemini: {}", msg);
    }

    let parts = resp["candidates"][0]["content"]["parts"]
        .as_array()
        .context("No parts in Gemini response")?;

    let text = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("");

    if text.is_empty() {
        anyhow::bail!("Gemini returned no text");
    }

    Ok(text)
}

// ── DDG instant answers ───────────────────────────────────────────────────────

async fn search_ddg(query: &str) -> Result<String> {
    let encoded = urlencoding(query);
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
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    if !output.status.success() {
        anyhow::bail!("DDG curl failed");
    }

    let body = String::from_utf8_lossy(&output.stdout);
    parse_ddg_response(&body, query)
}

fn parse_ddg_response(body: &str, query: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(body).context("Failed to parse DDG response")?;

    let mut parts: Vec<String> = Vec::new();

    if let Some(answer) = v["Answer"].as_str() {
        if !answer.is_empty() {
            parts.push(format!("Answer: {}", answer));
        }
    }

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
            "No instant answer for \"{}\". Try a more specific query.",
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
