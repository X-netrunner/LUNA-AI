use anyhow::{Context, Result};
use std::path::Path;

pub async fn read_file(path: &str) -> Result<String> {
    tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read: {}", path))
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
        let res = read_file(path_str).await.unwrap();
        assert_eq!(res, "hello world");
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
        let res = read_file(path_str).await.unwrap();
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
            read_file(path_str).await.unwrap(),
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
            read_file(path_str).await.unwrap(),
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
