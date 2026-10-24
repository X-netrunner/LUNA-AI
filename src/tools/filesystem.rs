use anyhow::{Context, Result};
use std::path::Path;

/// Biggest slice `read_file` hands back in one call when no range is asked for.
/// A whole large file dumped into a 16K-token window silently context-shifts in
/// Ollama (see `DEFAULT_NUM_CTX`) and the model loses the start of what it read,
/// so an un-ranged read of a big file returns the head and says how to get more.
const READ_MAX_LINES: usize = 1500;

/// Read a file, optionally a 1-based line range (`offset`..`offset+limit`).
///
/// With no range, a file up to `READ_MAX_LINES` lines comes back whole (the
/// common case, unchanged). A larger one comes back truncated to the head with
/// a one-line note telling the model to re-read with `offset`/`limit` — far
/// better for editing than a blob that overflowed the context window. When a
/// range is given, the slice is prefixed with a `lines X-Y of Z` header so the
/// model knows where it is in the file.
pub async fn read_file(path: &str, offset: Option<usize>, limit: Option<usize>) -> Result<String> {
    let body = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read: {}", path))?;

    // Fast path: no range asked for and the file is small enough to send whole.
    let total = body.lines().count();
    if offset.is_none() && limit.is_none() && total <= READ_MAX_LINES {
        return Ok(body);
    }

    let lines: Vec<&str> = body.lines().collect();
    let ranged = offset.is_some() || limit.is_some();
    // 1-based offset; default to the top of the file.
    let start = offset.map(|o| o.saturating_sub(1)).unwrap_or(0).min(total);
    let count = limit.unwrap_or(READ_MAX_LINES).min(READ_MAX_LINES);
    let end = start.saturating_add(count).min(total);
    let slice = lines[start..end].join("\n");

    if ranged {
        Ok(format!(
            "[{path}: lines {}-{} of {total}]\n{slice}",
            start + 1,
            end
        ))
    } else {
        // Un-ranged read of a big file: head + how to get the rest.
        Ok(format!(
            "[{path}: showing lines 1-{end} of {total} — file is large; \
             re-read with offset/limit (e.g. offset={}) for the rest]\n{slice}",
            end + 1
        ))
    }
}

pub async fn write_file(path: &str, content: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create parent dirs")?;
    }
    tokio::fs::write(path, content)
        .await
        .with_context(|| format!("Failed to write: {}", path))
}

pub async fn edit_file(
    path: &str,
    old_str: Option<&str>,
    new_str: Option<&str>,
    content: Option<&str>,
) -> Result<String> {
    if let Some(parent) = Path::new(path).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create parent dirs")?;
    }

    if let (Some(old), Some(new)) = (old_str, new_str) {
        if !Path::new(path).exists() {
            anyhow::bail!("File '{}' does not exist to perform text editing.", path);
        }
        let original = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read file for editing: {}", path))?;
        if !original.contains(old) {
            anyhow::bail!("Target text to replace was not found in '{}'", path);
        }
        let matches_count = original.matches(old).count();
        // Refuse an ambiguous edit rather than applying it to every match.
        //
        // `str::replace` rewrites ALL occurrences, so a request that reads like
        // one edit ("change PLACEHOLDER to REPLACED") silently becomes an
        // N-line change when the string happens to appear N times. Verified
        // live on 2026-09-30: a three-line file with the token on every line
        // came back "replaced 3 occurrence(s)" and all three lines changed.
        //
        // `self_patch` already refuses ambiguous matches for exactly this
        // reason; the raw write path must agree, or it becomes the easy way to
        // bypass that rule. The fix is to quote more surrounding context, which
        // is what the error says to do.
        if matches_count > 1 {
            anyhow::bail!(
                "Ambiguous edit refused: old_str occurs {matches_count} times in '{path}'. \
                 Rewriting all of them when you may have meant one is how a one-line fix \
                 turns into an N-line change. Include the surrounding lines in old_str so \
                 it matches exactly once, then retry. (If you really do want to change all \
                 {matches_count}, say so and use write_file with the full new content.)"
            );
        }
        let updated = original.replace(old, new);
        tokio::fs::write(path, updated)
            .await
            .with_context(|| format!("Failed to write edited content to: {}", path))?;
        Ok(format!(
            "Successfully edited '{}': replaced {} occurrence(s).",
            path, matches_count
        ))
    } else if let Some(new_content) = content {
        tokio::fs::write(path, new_content)
            .await
            .with_context(|| format!("Failed to write content to: {}", path))?;
        Ok(format!("Successfully wrote content to '{}'", path))
    } else {
        anyhow::bail!("No edit content or old_str/new_str provided for '{}'", path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_and_read_file() {
        let path = std::env::temp_dir().join("luna_test_rw.txt");
        let path_str = path.to_str().unwrap();
        write_file(path_str, "hello world").await.unwrap();
        let res = read_file(path_str, None, None).await.unwrap();
        assert_eq!(res, "hello world");
        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn read_file_range_and_large_file_guard() {
        let path = std::env::temp_dir().join("luna_test_range.txt");
        let path_str = path.to_str().unwrap();
        // 3000 lines: "L1".."L3000"
        let body: String = (1..=3000).map(|i| format!("L{i}\n")).collect();
        write_file(path_str, &body).await.unwrap();

        // Explicit range: lines 10-12, with a locating header.
        let r = read_file(path_str, Some(10), Some(3)).await.unwrap();
        assert!(r.starts_with(&format!("[{path_str}: lines 10-12 of 3000]")), "header: {r}");
        assert!(r.contains("\nL10\nL11\nL12"), "slice body: {r}");
        assert!(!r.contains("L13"), "limit respected");

        // Un-ranged read of a big file is capped to the head with a note, not
        // dumped whole, and the note points at the next page.
        let head = read_file(path_str, None, None).await.unwrap();
        assert!(head.contains(&format!("of 3000")), "reports total: {}", &head[..80]);
        assert!(head.contains("offset="), "tells how to page");
        assert!(head.lines().count() <= READ_MAX_LINES + 1, "head is capped");
        assert!(!head.contains("L2000"), "tail not included");

        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn test_edit_file_replace() {
        let path = std::env::temp_dir().join("luna_test_edit.txt");
        let path_str = path.to_str().unwrap();
        write_file(path_str, "fn main() {\n    println!(\"foo\");\n}\n").await.unwrap();
        edit_file(path_str, Some("println!(\"foo\");"), Some("println!(\"bar\");"), None)
            .await
            .unwrap();
        let res = read_file(path_str, None, None).await.unwrap();
        assert_eq!(res, "fn main() {\n    println!(\"bar\");\n}\n");
        let _ = tokio::fs::remove_file(path).await;
    }

    /// The live failure from 2026-09-30, reproduced exactly: `old_str` on
    /// three lines, and the tool rewrote all three.
    #[tokio::test]
    async fn an_ambiguous_edit_is_refused_not_applied_three_times() {
        let path = std::env::temp_dir().join("luna_test_dup.txt");
        let path_str = path.to_str().unwrap();
        let before = "line one PLACEHOLDER here\nline two PLACEHOLDER here\nline three PLACEHOLDER here\n";
        write_file(path_str, before).await.unwrap();

        let err = edit_file(
            path_str,
            Some("PLACEHOLDER"),
            Some("REPLACED"),
            None,
        )
        .await
        .expect_err("an ambiguous old_str must be refused");

        assert!(err.to_string().contains("Ambiguous"), "got: {err}");
        assert!(err.to_string().contains("3 times"), "got: {err}");
        // The point of the guard: the file must be untouched.
        assert_eq!(
            read_file(path_str, None, None).await.unwrap(),
            before,
            "a refused edit must not have written anything"
        );
        let _ = tokio::fs::remove_file(path).await;
    }

    /// The escape hatch must still work: quoting more context makes the match
    /// unique, which is exactly what the refusal message tells her to do.
    #[tokio::test]
    async fn adding_surrounding_context_disambiguates() {
        let path = std::env::temp_dir().join("luna_test_ctx.txt");
        let path_str = path.to_str().unwrap();
        let before = "line one PLACEHOLDER here\nline two PLACEHOLDER here\n";
        write_file(path_str, before).await.unwrap();
        edit_file(
            path_str,
            Some("line two PLACEHOLDER"),
            Some("line two FIXED"),
            None,
        )
        .await
        .expect("a unique old_str must be accepted");
        assert_eq!(
            read_file(path_str, None, None).await.unwrap(),
            "line one PLACEHOLDER here\nline two FIXED here\n"
        );
        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn a_missing_target_is_still_refused() {
        let path = std::env::temp_dir().join("luna_test_missing.txt");
        let path_str = path.to_str().unwrap();
        write_file(path_str, "alpha beta gamma\n").await.unwrap();
        let err = edit_file(path_str, Some("NOT_PRESENT"), Some("x"), None)
            .await
            .expect_err("absent old_str must be refused");
        assert!(err.to_string().contains("not found"), "got: {err}");
        let _ = tokio::fs::remove_file(path).await;
    }
}
