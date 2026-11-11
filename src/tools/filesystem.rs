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
    // A folder used to fail with "Is a directory", and the model then guessed at file names (or went looking in other places).
    // Now it gets the listing, newest first, and can read the right file next.
    if tokio::fs::metadata(path)
        .await
        .map(|m| m.is_dir())
        .unwrap_or(false)
    {
        let mut rd = tokio::fs::read_dir(path)
            .await
            .with_context(|| format!("Failed to read folder: {}", path))?;
        let mut items: Vec<(std::time::SystemTime, String, bool)> = Vec::new();
        while let Some(e) = rd.next_entry().await? {
            let meta = tokio::fs::metadata(e.path()).await.ok();
            let when = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            let is_dir = meta.map(|m| m.is_dir()).unwrap_or(false);
            items.push((when, e.file_name().to_string_lossy().to_string(), is_dir));
        }
        items.sort_by_key(|x| std::cmp::Reverse(x.0));
        let list: Vec<String> = items
            .iter()
            .take(200)
            .map(|(_, n, d)| if *d { format!("{}/", n) } else { n.clone() })
            .collect();
        return Ok(format!(
            "[{path} is a folder with {} entries, newest first; read_file one of them]\n{}",
            items.len(),
            list.join("\n")
        ));
    }
    let metadata = tokio::fs::metadata(path).await?;
    if !metadata.is_file() || metadata.len() > 8 * 1024 * 1024 {
        anyhow::bail!("read_file requires a regular text file of at most 8 MiB; use a targeted search for larger files");
    }
    let body = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read: {}", path))?;

    // Fast path: no range asked for and the file is small enough to send whole.
    let total = body.lines().count();
    if offset.is_none() && limit.is_none() && total <= READ_MAX_LINES && body.len() <= 32_000 {
        return Ok(body);
    }

    let lines: Vec<&str> = body.lines().collect();
    let ranged = offset.is_some() || limit.is_some();
    // 1-based offset; default to the top of the file.
    let start = offset.map(|o| o.saturating_sub(1)).unwrap_or(0).min(total);
    let count = limit.unwrap_or(READ_MAX_LINES).min(READ_MAX_LINES);
    let end = start.saturating_add(count).min(total);
    let mut slice = lines[start..end].join("\n");
    if slice.len() > 32_000 {
        let mut cut = 32_000;
        while !slice.is_char_boundary(cut) {
            cut -= 1;
        }
        slice.truncate(cut);
        slice.push_str("\n[Output truncated at 32000 bytes; use a smaller line range or targeted search. This is not the whole file.]");
    }

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

/// Replace through a sibling temporary file, preserving permissions and symlink targets.
async fn atomic_write(path: &str, content: &str, expected: Option<&str>) -> Result<()> {
    let path = path.to_owned();
    let content = content.to_owned();
    let expected = expected.map(str::to_owned);
    tokio::task::spawn_blocking(move || -> Result<()> {
        use std::io::Write;
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        let requested = Path::new(&path);
        let target = if std::fs::symlink_metadata(requested).is_ok() {
            std::fs::canonicalize(requested)?
        } else {
            requested.to_path_buf()
        };
        let parent = target
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let metadata = match std::fs::metadata(&target) {
            Ok(m) => {
                if !m.is_file() || m.nlink() > 1 {
                    anyhow::bail!(
                        "Refusing atomic replacement of a nonregular or hard-linked file: {path}"
                    );
                }
                Some(m)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let tmp = parent.join(format!(".luna-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            file.write_all(content.as_bytes())?;
            if let Some(meta) = metadata {
                file.set_permissions(meta.permissions())?;
            }
            file.sync_all()?;
            if let Some(before) = expected {
                if std::fs::read_to_string(&target)? != before {
                    anyhow::bail!("File changed during editing; re-read {path} before retrying");
                }
            }
            std::fs::rename(&tmp, &target)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    })
    .await?
}

pub async fn write_file(path: &str, content: &str) -> Result<()> {
    atomic_write(path, content, None).await
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

    if content.is_some() && (old_str.is_some() || new_str.is_some()) {
        anyhow::bail!("Choose a targeted replacement or full content, not both");
    }
    if old_str.is_some() != new_str.is_some() {
        anyhow::bail!("Targeted edits require both old_str and new_str");
    }
    if let (Some(old), Some(new)) = (old_str, new_str) {
        if old.is_empty() {
            anyhow::bail!("old_str must not be empty; read the file and quote a unique fragment");
        }
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
        let check = super::verify::syntax_check(path, &updated);
        if let super::verify::SyntaxCheck::Failed(error) = &check {
            anyhow::bail!("Edit was not written: resulting file does not parse: {error}");
        }
        atomic_write(path, &updated, Some(&original)).await?;
        Ok(format!(
            "Successfully edited '{}': replaced {} occurrence(s).\n{}",
            path,
            matches_count,
            check.receipt()
        ))
    } else if let Some(new_content) = content {
        let check = super::verify::syntax_check(path, new_content);
        if let super::verify::SyntaxCheck::Failed(error) = &check {
            anyhow::bail!("Edit was not written: resulting file does not parse: {error}");
        }
        atomic_write(path, new_content, None).await?;
        Ok(format!(
            "Successfully wrote content to '{}'\n{}",
            path,
            check.receipt()
        ))
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
        assert!(
            r.starts_with(&format!("[{path_str}: lines 10-12 of 3000]")),
            "header: {r}"
        );
        assert!(r.contains("\nL10\nL11\nL12"), "slice body: {r}");
        assert!(!r.contains("L13"), "limit respected");

        // Un-ranged read of a big file is capped to the head with a note, not
        // dumped whole, and the note points at the next page.
        let head = read_file(path_str, None, None).await.unwrap();
        assert!(
            head.contains(&"of 3000".to_string()),
            "reports total: {}",
            &head[..80]
        );
        assert!(head.contains("offset="), "tells how to page");
        assert!(head.lines().count() <= READ_MAX_LINES + 1, "head is capped");
        assert!(!head.contains("L2000"), "tail not included");

        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn test_edit_file_replace() {
        let path = std::env::temp_dir().join("luna_test_edit.txt");
        let path_str = path.to_str().unwrap();
        write_file(path_str, "fn main() {\n    println!(\"foo\");\n}\n")
            .await
            .unwrap();
        edit_file(
            path_str,
            Some("println!(\"foo\");"),
            Some("println!(\"bar\");"),
            None,
        )
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
        let before =
            "line one PLACEHOLDER here\nline two PLACEHOLDER here\nline three PLACEHOLDER here\n";
        write_file(path_str, before).await.unwrap();

        let err = edit_file(path_str, Some("PLACEHOLDER"), Some("REPLACED"), None)
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

#[cfg(test)]
mod integrity_tests {
    use super::*;
    #[tokio::test]
    async fn edits_preserve_permissions_and_symlinks_and_reject_stale_content() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir(&dir).unwrap();
        let file = dir.join("code.py");
        let link = dir.join("link.py");
        std::fs::write(&file, "print('old')\n").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o750)).unwrap();
        symlink(&file, &link).unwrap();
        edit_file(link.to_str().unwrap(), Some("'old'"), Some("'new'"), None)
            .await
            .unwrap();
        assert!(std::fs::symlink_metadata(&link).unwrap().is_symlink());
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o750
        );
        assert!(atomic_write(file.to_str().unwrap(), "oops", Some("stale"))
            .await
            .is_err());
        assert!(edit_file(
            file.to_str().unwrap(),
            Some("print('new')"),
            Some("def broken("),
            None
        )
        .await
        .is_err());
        assert!(edit_file(file.to_str().unwrap(), Some(""), Some("x"), None)
            .await
            .is_err());
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "print('new')\n");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[tokio::test]
    async fn long_unicode_lines_are_bounded_and_labelled_partial() {
        let file = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        std::fs::write(&file, "界".repeat(20000)).unwrap();
        let result = read_file(file.to_str().unwrap(), None, None).await.unwrap();
        assert!(result.len() < 33000 && result.contains("not the whole file"));
        std::fs::remove_file(file).unwrap();
    }
}
