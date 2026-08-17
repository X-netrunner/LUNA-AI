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
    let result = parse_ddg_response(&body, query)?;

    // If instant answer was empty, fall back to scraping HTML search results
    if result.starts_with("No instant answer") {
        return search_html(query).await;
    }

    Ok(result)
}

/// Fallback: scrape Brave Search results when DDG instant answer is empty
async fn search_html(query: &str) -> Result<String> {
    let encoded = urlencoding(query);
    let url = format!(
        "https://search.brave.com/search?q={}&source=web",
        encoded
    );

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-sL",
                "--max-time", "10",
                "--user-agent", "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                &url,
            ])
            .output()
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    if !output.status.success() {
        anyhow::bail!("curl failed");
    }

    let html = String::from_utf8_lossy(&output.stdout);
    let mut results: Vec<(String, String)> = Vec::new();

    // Brave wraps results in <a href="URL"><span class="snippet-title">TITLE</span></a>
    let title_marker = r#"snippet-title""#;
    let mut pos = 0;
    while let Some(m) = html[pos..].find(title_marker) {
        let after = &html[pos + m + title_marker.len()..];
        // Find the opening > of the span
        if let Some(gt) = after.find('>') {
            let text_start = gt + 1;
            if let Some(lt) = after[text_start..].find("</span>") {
                let title = after[text_start..text_start + lt]
                    .replace("<br>", " ")
                    .trim()
                    .to_string();

                // Find the enclosing <a href="...">
                let search_back = &html[..pos + m + title_marker.len()];
                if let Some(a_tag) = search_back.rfind("<a ") {
                    let a_content = &search_back[a_tag..];
                    if let Some(href_start) = a_content.find("href=\"") {
                        let url_start = a_tag + href_start + 6;
                        if let Some(url_end) = html[url_start..].find('"') {
                            let link = html[url_start..url_start + url_end].to_string();
                            if !title.is_empty()
                                && !link.contains("brave.com")
                                && !link.contains("javascript:")
                                && link.starts_with("http")
                            {
                                results.push((title, link));
                            }
                        }
                    }
                }
                pos = pos + m + title_marker.len() + gt + 1;
            } else {
                pos = pos + m + title_marker.len();
            }
        } else {
            break;
        }
        if results.len() >= 5 {
            break;
        }
    }

    if results.is_empty() {
        Ok(format!(
            "No results found for \"{}\". Try a different query.",
            query
        ))
    } else {
        let mut out = format!("Search results for \"{}\":\n", query);
        for (title, link) in &results {
            out.push_str(&format!("- {} ({})\n", title, link));
        }
        Ok(out)
    }
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
