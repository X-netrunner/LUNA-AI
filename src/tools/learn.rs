//! tools/learn.rs — One-shot web learning: search + fetch combined
//!
//! Collapses the web_search -> fetch_page two-step chain (which models
//! often botch — wrong URL, forgetting the second call) into a single
//! reliable tool call. Returns combined text for the model to read
//! and then save with `remember` if it's worth keeping.

use super::{shell, web};
use anyhow::Result;

pub async fn learn(topic: &str, sudo_pass: Option<&str>) -> Result<String> {
    let mut output = String::new();

    // Step 1: search
    let search_result = web::search(topic)
        .await
        .unwrap_or_else(|e| format!("Search failed: {}", e));
    output.push_str("=== Search results ===\n");
    output.push_str(&search_result);
    output.push_str("\n\n");

    // Step 2: if the search gave us a source URL, fetch the full page
    if let Some(url) = extract_url(&search_result) {
        output.push_str(&format!("=== Fetched page: {} ===\n", url));
        let safe_url = url.replace('\'', "'\\''");
        let cmd = format!(
            "lynx -dump -nolist -width=120 '{}' 2>/dev/null \
             || curl -sL --max-time 10 '{}' \
                | sed 's/<[^>]*>//g; s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g' \
                | sed '/^[[:space:]]*$/d' \
                | head -150",
            safe_url, safe_url
        );
        let result = shell::run_command(&cmd, sudo_pass).await?;
        let text = result.stdout.trim();
        let truncated = if text.len() > 3000 {
            text.char_indices()
                .take_while(|(i, _)| *i < 3000)
                .last()
                .map(|(i, c)| &text[..i + c.len_utf8()])
                .unwrap_or(text)
        } else {
            text
        };
        output.push_str(truncated);
    } else {
        output.push_str(
            "(No direct source URL found in the search results — the summary above \
             is all that's available. If you know a specific URL, use fetch_page directly.)"
        );
    }

    Ok(output)
}

fn extract_url(text: &str) -> Option<String> {
    text.lines()
        .find(|l| l.starts_with("Source:"))
        .map(|l| l.trim_start_matches("Source:").trim().to_string())
}
