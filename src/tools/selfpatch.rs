//! tools/selfpatch.rs — gated self-modification.
//!
//! Luna can propose changes to her own source. Nothing is ever written to the
//! real tree until a proposal has (a) passed `cargo test` against a faithful
//! copy of the *current working tree*, and (b) been explicitly approved.
//!
//! The flow, and the invariant behind each step:
//!
//!   propose  → stage the new content, show a unified diff. Touches nothing.
//!   validate → rsync the real working tree into a scratch dir, overlay the
//!              staged files, run `cargo test` THERE. The real tree is never
//!              modified during validation, so a bad proposal cannot leave
//!              broken source behind.
//!   apply    → refuse unless validation is green AND the user approved.
//!              Timestamped backup first, then the swap.
//!   rollback → restore the backup.
//!
//! Why a scratch copy instead of a `git worktree`: a worktree checks out HEAD,
//! which silently omits every uncommitted change. Luna's tree is normally
//! dirty (that is where her work-in-progress lives), so a HEAD-based gate would
//! validate a *different codebase* than the one that would actually run. We
//! copy the working tree instead — it is ~3 MB without target/.
//!
//! Honest limit: this is a workflow gate, not a sandbox. Luna also holds
//! `run_shell` with sudo, so nothing here stops her from editing her source
//! through the back door. The gate makes the *honest* path safe and cheap; it
//! does not make the dishonest path impossible. The real backstop is the diff
//! the user reads, the green test run, and the backup.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── state ─────────────────────────────────────────────────────────────────────

fn state_dir() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                .join(".local/share")
        })
        .join("luna/selfpatch");
    let _ = std::fs::create_dir_all(&base);
    base
}

fn staging_dir() -> PathBuf {
    let d = state_dir().join("staging");
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Scratch copy of the working tree that tests actually run against.
fn validate_dir() -> PathBuf {
    state_dir().join("validate")
}

/// Separate target dir so validation never clobbers the real build cache
/// (and vice versa). Recompiles the crate each time, but keeps the two trees
/// from thrashing each other's artifacts.
fn validate_target() -> PathBuf {
    state_dir().join("target")
}

fn backup_dir() -> PathBuf {
    let d = state_dir().join("backups");
    let _ = std::fs::create_dir_all(&d);
    d
}

fn state_file() -> PathBuf {
    state_dir().join("proposal.json")
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct StoredFile {
    pub rel: String,
    pub staged: String,
    /// Fingerprint of the ORIGINAL file at stage time. If the real file changes
    /// between propose and apply (the user edits it, another process rewrites
    /// it), applying would silently discard that work — so we refuse instead.
    #[serde(default)]
    pub orig_hash: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Validation {
    pub ok: bool,
    pub exit_code: i32,
    pub summary: String,
    pub at: String,
    /// Content-level problems that `cargo test` cannot see. `#[serde(default)]`
    /// so a `proposal.json` written by an older build still loads.
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct Stored {
    pub reason: String,
    pub files: Vec<StoredFile>,
    pub created: String,
    pub validation: Option<Validation>,
    pub backup_id: Option<String>,
}

fn load() -> Stored {
    std::fs::read_to_string(state_file())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save(s: &Stored) -> Result<()> {
    let body = serde_json::to_string_pretty(s)?;
    std::fs::write(state_file(), body)?;
    Ok(())
}

fn now() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

// ── read receipts ─────────────────────────────────────────────────────────────

/// Files actually read this session, as paths relative to the source root.
///
/// A model that edits from memory will confidently "fix" a file that does not
/// exist, or apply a `find` string that never matched anything. Asking her in a
/// prompt to read first is not enough — she skips it. So `propose` *requires* a
/// receipt, which is only issued by this module's own `read` action. Grounding
/// becomes structural instead of advisory.
fn receipts() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    static R: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    R.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

fn mark_read(rel: &str) {
    if let Ok(mut g) = receipts().lock() {
        g.insert(rel.to_string());
    }
}

fn was_read(rel: &str) -> bool {
    receipts()
        .lock()
        .map(|g| g.contains(rel))
        .unwrap_or(false)
}

/// List the source files she is allowed to touch, so she stops inventing paths.
pub fn list_source_files(root: &Path, filter: &str) -> String {
    fn walk(
        root: &Path,
        dir: &Path,
        depth: usize,
        filter: &str,
        out: &mut String,
        count: &mut usize,
    ) {
        if depth > 2 || *count > 400 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.path());
        for e in entries {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            if p.is_dir() {
                walk(root, &p, depth + 1, filter, out, count);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let rel = p
                    .strip_prefix(root)
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .replace('\\', "/");
                // Optional substring filter. Dumping all ~58 paths at a 7B is
                // how it ends up re-calling `files` instead of moving on to
                // the read; a short list keeps the next step obvious.
                if !filter.is_empty() && !rel.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
                let lines = std::fs::read_to_string(&p)
                    .map(|s| s.lines().count())
                    .unwrap_or(0);
                out.push_str(&format!("  {rel} ({lines} lines)\n"));
                *count += 1;
            }
        }
    }
    let mut out = String::new();
    let mut count = 0usize;
    walk(root, root, 0, filter, &mut out, &mut count);
    if out.is_empty() {
        return if filter.is_empty() {
            "No .rs source files found.\n".into()
        } else {
            format!(
                "No .rs source file matches \"{filter}\". Run action=files with no filter to \
                 see the real list — do not guess a path.\n"
            )
        };
    }
    let head = if filter.is_empty() {
        format!("{count} source file(s) she may edit")
    } else {
        format!("{count} source file(s) matching \"{filter}\"")
    };
    format!("{head}:\n{out}")
}

/// `read` — return a file's exact current content and issue a read receipt.
pub fn read_file(root: &Path, input: &str) -> Result<String> {
    let rel = safe_rel(root, input)?;
    if is_protected(&rel, &crate::config::SelfPatchConfig::default().protected_files) {
        anyhow::bail!("{rel} is protected — it cannot be read for editing either");
    }
    let p = root.join(&rel);
    if !p.is_file() {
        anyhow::bail!(
            "{rel} does not exist. Use action=\"files\" to list the real source files — \
             do not guess paths."
        );
    }
    let body = std::fs::read_to_string(&p).with_context(|| format!("failed to read {rel}"))?;
    mark_read(&rel);
    let lines: Vec<String> = body
        .lines()
        .enumerate()
        .map(|(i, l)| format!("{:>5}| {l}", i + 1))
        .collect();
    Ok(format!(
        "{rel} — {} lines. Copy 'find' strings EXACTLY from this (indentation included).\n\n{}\n",
        body.lines().count(),
        lines.join("\n")
    ))
}

/// One requested edit: replace the FIRST argument's unique occurrence with the
/// second.
pub type Edit = (String, String);

/// A requested change to one file, in whichever shape the caller used.
#[derive(Debug, Clone, Default)]
pub struct ChangeSpec {
    pub file: String,
    /// Whole-file replacement (for small files).
    pub content: Option<String>,
    /// Targeted find/replace pairs (the 7B-friendly shape — preferred).
    pub edits: Vec<Edit>,
    /// Line-anchored edits: (1-based line number, replacement text).
    ///
    /// Added because copying exact text out of a read is the model's weakest
    /// skill, not the tooling's limit. Measured at N=3 on the 7B:
    /// find/replace append succeeded 3/3 but modify failed 3/3 — every attempt
    /// ran to the token cap (`done=length`) because a free-form
    /// `replace` string has no length bound and the model never closed it.
    ///
    /// A line number is an integer, which the model emits reliably, and it
    /// never has to reproduce existing bytes — so the Rust `\n`-escape trap
    /// (the model renders the two-character escape as a real newline, making
    /// byte-exact matching impossible) simply cannot occur. The 14B coder
    /// scored 3/3 on both, so this is about making the *7B* viable.
    ///
    /// Resolved against the real bytes on disk, so results stay byte-exact.
    pub line_edits: Vec<(usize, String)>,
}

/// Turn a line-anchored edit into the `(find, replace)` pair `apply_edits`
/// enforces, using the file's real current text.
///
/// `line_no` is 1-based; `line_no == len+1` appends after the last line, which
/// is the only way to add text to a file (there is no existing text to find).
///
/// If the target line is not unique on its own, the preceding line is folded
/// in to make it unique — still derived from disk, never guessed. If that is
/// also ambiguous we refuse rather than guess, same rule as `apply_edits`.
pub fn resolve_line_edit(orig: &str, line_no: usize, replacement: &str) -> Result<Edit> {
    let lines: Vec<&str> = orig.lines().collect();
    let n = lines.len();

    if line_no == 0 || line_no > n + 1 {
        anyhow::bail!(
            "line {line_no} is outside the file — it has {n} lines, so the only \
             append position is line {}.", n + 1
        );
    }

    // Append: anchor on the last real line and re-emit it with the new text
    // tacked on, so the find string is guaranteed to exist in the file.
    if line_no == n + 1 {
        if n == 0 {
            // Empty file — nothing to anchor on; treat as a whole-file write.
            return Ok((String::new(), replacement.to_string()));
        }
        let last = lines[n - 1];
        return Ok((last.to_string(), format!("{last}\n{replacement}")));
    }

    let target = lines[line_no - 1];
    if target.trim().is_empty() {
        anyhow::bail!(
            "line {line_no} is blank, so it cannot identify a unique place to edit. \
             Pick a line with actual code on it."
        );
    }

    match orig.matches(target).count() {
        1 => Ok((target.to_string(), replacement.to_string())),
        hits if hits > 1 && line_no > 1 => {
            // Disambiguate with the preceding line — still exact disk text.
            let with_prev = format!("{}\n{}", lines[line_no - 2], target);
            match orig.matches(&with_prev).count() {
                1 => Ok((with_prev, format!("{}\n{replacement}", lines[line_no - 2]))),
                _ => anyhow::bail!(
                    "line {line_no} ('{}') and the line before it are not unique \
                     ({hits} and more occurrences). Use find/replace with more \
                     surrounding context instead.",
                    crate::util::truncate(target, 60)
                ),
            }
        }
        hits => anyhow::bail!(
            "line {line_no} could not be located uniquely in the file ({hits} \
             matches). Re-read the file and check the line number."
        ),
    }
}


/// Apply targeted edits to a file's contents.
///
/// Each `find` string must occur **exactly once** — that rule is the whole
/// point. It makes a replacement unambiguous, so she can never silently hit the
/// wrong occurrence, and a drifted source (where her remembered text no longer
/// matches) is rejected instead of producing a mangled file. This is also what
/// makes the shape practical: a 7B can emit a few short exact strings, but
/// cannot reliably reproduce a whole source file.
pub fn apply_edits(orig: &str, edits: &[Edit]) -> Result<String> {
    if edits.is_empty() {
        anyhow::bail!("no edits supplied — give at least one (find, replace) pair");
    }
    let mut out = orig.to_string();
    for (i, (find, replace)) in edits.iter().enumerate() {
        if find.is_empty() {
            anyhow::bail!("edit {}: the 'find' text is empty", i + 1);
        }
        let hits = out.matches(find.as_str()).count();
        match hits {
            0 => {
                return Err(anyhow::anyhow!(
                    "edit {}: the text to find does not appear in the file at all.\n\
                     Your remembered text has drifted from the current source — read the \
                     file again and copy the EXACT current text. Nothing was changed.",
                    i + 1
                ));
            }
            1 => {}
            n => {
                return Err(anyhow::anyhow!(
                    "edit {}: the text to find appears {} times, so the replacement would be \
                     ambiguous. Include more surrounding context so it matches exactly once.",
                    i + 1,
                    n
                ));
            }
        }
        out = out.replacen(find.as_str(), replace, 1);
    }
    if out == orig {
        anyhow::bail!("the edits produced no change — nothing to propose");
    }
    Ok(out)
}

// ── path safety (pure, unit-tested) ───────────────────────────────────────────

/// Normalize a user/model-supplied path into a safe relative path under
/// `root`. Rejects absolute paths and `..` traversal, then confirms the
/// resolved location is genuinely inside `root` (defeats symlink escapes).
pub fn safe_rel(root: &Path, input: &str) -> Result<String> {
    let raw = input.trim();
    if raw.is_empty() {
        anyhow::bail!("empty file path");
    }
    let p = Path::new(raw);
    if p.is_absolute() {
        anyhow::bail!("absolute path not allowed ({raw}) — use a path relative to the source root");
    }
    for c in p.components() {
        if matches!(c, std::path::Component::ParentDir) {
            anyhow::bail!("path traversal (..) is not allowed: {raw}");
        }
    }
    if raw.contains('\0') {
        anyhow::bail!("invalid path");
    }

    let joined = root.join(p);
    let canon_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    // Canonicalize the deepest existing ancestor, then confirm containment.
    let mut probe = joined.clone();
    let canon = loop {
        match probe.canonicalize() {
            Ok(c) => break c,
            Err(_) => {
                if !probe.pop() || probe.as_os_str().is_empty() {
                    break joined.clone();
                }
            }
        }
    };
    if !canon.starts_with(&canon_root) {
        anyhow::bail!("path escapes the source root: {raw}");
    }
    Ok(p.to_string_lossy().replace('\\', "/"))
}

/// Is this file off-limits? Exact match, or inside a protected directory.
pub fn is_protected(rel: &str, list: &[String]) -> bool {
    let rel = rel.trim_start_matches("./");
    list.iter().any(|pat| {
        let pat = pat.trim().trim_start_matches("./").trim_end_matches('/');
        if pat.is_empty() {
            return false;
        }
        rel == pat || rel.starts_with(&format!("{pat}/"))
    })
}

/// Should a raw `write_file` to this path be refused?
///
/// Luna's generic `write_file` tool can touch any path, which would let her
/// edit her own source while skipping the test gate entirely. The gate is only
/// meaningful if the ungated route is closed, so self-source writes are
/// refused here and must go through `self_patch` instead.
///
/// `edit_file` is deliberately NOT covered: it opens a visible editor that the
/// user is sitting in front of, which is user-mediated, not autonomous.
///
/// Honest limit: this closes the tool-layer back door only. She also has
/// `run_shell`, so a determined path via shell is still possible — this is a
/// guardrail, not a sandbox.
pub fn blocks_raw_write(cfg: &crate::config::SelfPatchConfig, path: &str) -> Option<String> {
    if !cfg.enabled {
        return None;
    }
    let expanded = if let Some(rest) = path.strip_prefix("~/") {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest)
    } else {
        PathBuf::from(path)
    };
    let root = PathBuf::from(&cfg.source_dir);
    let canon_root = root.canonicalize().ok()?;
    // Compare against the root even if the file doesn't exist yet, by
    // canonicalizing the deepest existing ancestor of the target.
    let mut probe = expanded.clone();
    let canon = loop {
        match probe.canonicalize() {
            Ok(c) => break c,
            Err(_) => {
                if !probe.pop() || probe.as_os_str().is_empty() {
                    break expanded.clone();
                }
            }
        }
    };
    if canon.starts_with(&canon_root) {
        Some(format!(
            "Refusing to write {} directly — that is Luna's own source and writing it \
             raw would skip the test gate. Use self_patch instead: action=propose \
             (stage + diff), action=validate (cargo test on a scratch copy), then \
             action=apply once the user approves.",
            path
        ))
    } else {
        None
    }
}

// ── change detection ──────────────────────────────────────────────────────────

/// FNV-1a 64-bit. Not a security primitive — just a cheap "did this file
/// change under us" fingerprint, and it avoids pulling in a hashing dep.
pub fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn hash_file(p: &Path) -> Result<u64> {
    let b = std::fs::read(p).with_context(|| format!("failed to read {}", p.display()))?;
    Ok(fingerprint(&b))
}

// ── diff ──────────────────────────────────────────────────────────────────────

/// Unified diff between the real file and the staged version. Read-only:
/// `git diff --no-index` never touches the real tree.
fn diff_for(root: &Path, rel: &str, staged: &Path) -> String {
    let real = root.join(rel);
    let out = std::process::Command::new("git")
        .args(["diff", "--no-index", "--", &real.display().to_string(), &staged.display().to_string()])
        .output();
    let Ok(out) = out else {
        return format!("(could not diff {rel}: git not available)\n");
    };
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    // git prints the real on-disk paths in the header; rewrite them to a normal
    // a/<rel> b/<rel> form so the diff reads as a plain source diff.
    text.lines()
        .map(|l| {
            if l.starts_with("--- ") {
                format!("--- a/{rel}")
            } else if l.starts_with("+++ ") {
                format!("+++ b/{rel}")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn diff_all(cfg: &crate::config::SelfPatchConfig, st: &Stored) -> String {
    let root = PathBuf::from(&cfg.source_dir);
    let mut out = String::new();
    for f in &st.files {
        out.push_str(&diff_for(&root, &f.rel, Path::new(&f.staged)));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

// ── actions ───────────────────────────────────────────────────────────────────

/// Stage a set of file changes. Touches nothing in the real tree.
pub async fn propose(
    cfg: &crate::config::SelfPatchConfig,
    changes: &[ChangeSpec],
    reason: &str,
) -> Result<String> {
    if !cfg.enabled {
        anyhow::bail!("self-modification is disabled in config ([selfpatch] enabled = false)");
    }
    if changes.is_empty() {
        anyhow::bail!("no changes supplied — pass changes=[{{file, content}}, ...]");
    }
    let root = PathBuf::from(&cfg.source_dir);
    if !root.is_dir() {
        anyhow::bail!("source root not found: {}", root.display());
    }

    let existing = load();
    if !existing.files.is_empty() {
        anyhow::bail!(
            "a proposal is already staged ({} file(s)). Review it with \
             self_patch {{action:\"review\"}}, then apply/rollback/discard before staging a new one.",
            existing.files.len()
        );
    }

    let mut files = Vec::new();
    for ch in changes {
        let rel = safe_rel(&root, &ch.file)?;
        if is_protected(&rel, &cfg.protected_files) {
            anyhow::bail!("{rel} is protected — self-modification may not touch it");
        }
        let target = root.join(&rel);
        if !target.is_file() {
            anyhow::bail!(
                "{rel} does not exist. Use self_patch {{action:\"files\"}} to list the real \
                 source files — do not guess paths."
            );
        }
        // Structural grounding: she must have actually read this file, via this
        // tool, before she is allowed to propose a change to it. A model
        // editing from memory invents paths and 'find' strings that match
        // nothing; this turns clause 8 of the Constitution from a request into
        // a precondition.
        if !was_read(&rel) {
            anyhow::bail!(
                "you have not read {rel} yet, so any edit you propose would be a guess. \
                 Read it first: self_patch {{action:\"read\", file:\"{rel}\"}}. Then copy the \
                 exact 'find' text out of that output."
            );
        }
        // Build the new content: either whole-file, or targeted edits applied to
        // the current bytes on disk.
        let new_content = match (&ch.content, ch.edits.is_empty(), ch.line_edits.is_empty()) {
            (Some(c), _, _) => c.clone(),
            (None, false, _) | (None, true, false) => {
                let orig = std::fs::read_to_string(&target)
                    .with_context(|| format!("failed to read {rel}"))?;
                // Line-anchored edits are resolved first, against the same bytes,
                // so they go through the identical exactly-once safety rule.
                let mut all = ch.edits.clone();
                for (line_no, replacement) in &ch.line_edits {
                    let resolved = resolve_line_edit(&orig, *line_no, replacement)
                        .map_err(|e| anyhow::anyhow!("in {rel}: {e:#}"))?;
                    all.push(resolved);
                }
                // Use `{:#}` so the *cause* survives: Luna reads this error text
                // to learn what to fix, so "in foo.rs" alone would teach her
                // nothing. Flatten the chain into one readable message.
                apply_edits(&orig, &all)
                    .map_err(|e| anyhow::anyhow!("in {rel}: {e:#}"))?
            }
            (None, true, true) => anyhow::bail!(
                "{rel}: give edits:[{{find,replace}}], line_edits:[{{at_line,replace_with}}], \
                 or content:\"<full file>\""
            ),
        };
        let staged = staging_dir().join(rel.replace('/', "__"));
        let orig_hash = hash_file(&target)?;
        std::fs::write(&staged, new_content)
            .with_context(|| format!("failed to stage {rel}"))?;
        files.push(StoredFile { rel, staged: staged.display().to_string(), orig_hash });
    }

    let st = Stored {
        reason: reason.to_string(),
        files,
        created: now(),
        validation: None,
        backup_id: None,
    };
    save(&st)?;

    let mut msg = format!(
        "Staged {} file(s) as a proposal. Nothing in the real tree has changed yet.\n\n",
        st.files.len()
    );
    if !reason.trim().is_empty() {
        msg.push_str(&format!("Reason: {reason}\n\n"));
    }
    msg.push_str("```diff\n");
    msg.push_str(diff_all(cfg, &st).trim_end());
    msg.push_str("\n```\n\n");
    msg.push_str(
        "Next: self_patch {{action:\"validate\"}} to run the test suite against this \
         change, then {{action:\"apply\"}} once the user approves. If the user \
         declines, {{action:\"discard\"}}.",
    );
    Ok(msg)
}

/// Run `cargo test` against a scratch copy of the working tree + the proposal.
pub async fn validate(cfg: &crate::config::SelfPatchConfig) -> Result<String> {
    if !cfg.enabled {
        anyhow::bail!("self-modification is disabled in config");
    }
    let st = load();
    if st.files.is_empty() {
        anyhow::bail!("no proposal is staged — propose a change first");
    }
    let root = PathBuf::from(&cfg.source_dir);
    if !root.is_dir() {
        anyhow::bail!("source root not found: {}", root.display());
    }

    let vdir = validate_dir();
    // Faithful copy of the CURRENT working tree (not HEAD), minus the 21 GB
    // target/ and the .git dir.
    let rsync = std::process::Command::new("rsync")
        .args([
            "-a",
            "--delete",
            "--exclude=target/",
            "--exclude=.git",
            &format!("{}/", root.display()),
            &format!("{}/", vdir.display()),
        ])
        .output()
        .context("failed to run rsync (is rsync installed?)")?;
    if !rsync.status.success() {
        anyhow::bail!("rsync failed: {}", String::from_utf8_lossy(&rsync.stderr));
    }

    // Overlay the proposal.
    for f in &st.files {
        let dst = vdir.join(&f.rel);
        if let Some(parent) = dst.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::copy(&f.staged, &dst)
            .with_context(|| format!("failed to overlay {} into the scratch tree", f.rel))?;
    }

    // Run the suite. Failures of ANY kind — compile error, test failure —
    // surface as a non-zero exit, which is what we branch on.
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("test")
        .current_dir(&vdir)
        .env("CARGO_TARGET_DIR", validate_target())
        // Keep the scratch build from thrashing: single job, no network churn.
        .env("CARGO_NET_OFFLINE", "true");
    let out = cmd.output().context("failed to run cargo test")?;
    let code = out.status.code().unwrap_or(-1);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let tail: String = combined
        .lines()
        .rev()
        .take(40)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");

    // Content checks `cargo test` cannot perform. See `content_warnings`.
    let mut warnings: Vec<String> = Vec::new();
    for f in &st.files {
        if let Ok(new) = std::fs::read_to_string(&f.staged) {
            if let Ok(orig) = std::fs::read_to_string(root.join(&f.rel)) {
                for w in content_warnings(&orig, &new) {
                    warnings.push(format!("{}: {w}", f.rel));
                }
            }
        }
    }

    let v = Validation {
        ok: code == 0,
        exit_code: code,
        summary: tail,
        at: now(),
        warnings: warnings.clone(),
    };
    let mut st2 = st.clone();
    st2.validation = Some(v.clone());
    save(&st2)?;

    // The diff goes in the validate result on purpose. `validate` is the last
    // gate before the user approves, and the observed failure was invisible
    // precisely because the model never had the change in front of it at the
    // moment it decided the change was fine.
    let diff = diff_all(cfg, &st);
    let diff_block = format!("\n\n--- staged diff ---\n```diff\n{}\n```", diff.trim_end());

    if !warnings.is_empty() {
        // Lead with the warning. A compile pass is true but is not the point.
        return Ok(format!(
            "Validation: `cargo test` exited {code} (compiles, existing tests pass) — BUT \
             {} content problem(s) that a compiler cannot detect were found. Do NOT tell \
             the user this change is ready. Show the diff, fix the problems, and re-propose.\n\n\
             {}{diff_block}",
            warnings.len(),
            warnings
                .iter()
                .map(|w| format!("  * {w}"))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    if v.ok {
        Ok(
            "Validation PASSED — this means the change COMPILES and the existing test \
             suite still passes. It does NOT mean the change is correct: a compiler \
             cannot see a wrong comment, a duplicated line, or logic that happens to \
             build. Read the diff yourself before asking the user to approve it.\n\
             Tell the user the diff and ask for approval, then self_patch \
             {{action:\"apply\"}}. (She can also {{action:\"rollback\"}} later.)"
                .to_string()
                + &diff_block,
        )
    } else {
        Ok(format!(
            "Validation FAILED — `cargo test` exited {}. The real source was NOT \
             touched; the change exists only in the scratch copy.\n\n\
             Tell the user it failed and show this output:\n\n```\n{}\n```\n\
             Either fix the proposal or {{action:\"discard\"}} it.",
            v.exit_code, v.summary
        ) + &diff_block)
    }
}

/// Show the staged proposal and its state. Read-only.
pub fn review(cfg: &crate::config::SelfPatchConfig) -> Result<String> {
    let st = load();
    if st.files.is_empty() {
        return Ok("No proposal is staged.".into());
    }
    let mut msg = String::new();
    if !st.reason.trim().is_empty() {
        msg.push_str(&format!("Reason: {}\n\n", st.reason));
    }
    msg.push_str(&format!("Staged at: {}\n", st.created));
    match &st.validation {
        Some(v) => {
            msg.push_str(&format!(
                "Validation: {} (cargo test exit {}) at {}\n",
                if v.ok { "PASSED" } else { "FAILED" },
                v.exit_code,
                v.at
            ));
            if !v.warnings.is_empty() {
                // A compile pass is not a correctness pass. Say so here too,
                // so `review` cannot be read as clearance either.
                msg.push_str(&format!(
                    "CONTENT WARNINGS ({}): a compiler cannot see these.\n{}\n",
                    v.warnings.len(),
                    v.warnings
                        .iter()
                        .map(|w| format!("  * {w}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
            }
            msg.push('\n');
        }
        None => msg.push_str("Validation: not run yet\n\n"),
    }
    msg.push_str("```diff\n");
    msg.push_str(diff_all(cfg, &st).trim_end());
    msg.push_str("\n```");
    Ok(msg)
}

/// Content-level defects that `cargo test` cannot see.
///
/// `validate` proves the change COMPILES and that existing tests still pass.
/// That is a much weaker statement than "the change is correct", and on
/// 2026-09-27 the gap was concrete: Luna replaced line 3 with a block that
/// accidentally re-included line 4, so the original line 4 survived and the
/// doc comment ended up with a duplicated sentence. `cargo test` exited 0 and
/// `validate` reported PASSED, which is technically true and practically
/// misleading — the tool's whole job here is to not hand the user false
/// assurance before they approve a change to their own agent.
///
/// This is deliberately narrow: one high-confidence defect class, not a lint
/// engine. A warning that cries wolf on legitimate code is worse than none,
/// because it trains both Luna and the user to skim past it.
///
/// Pure and side-effect free, so it is unit-tested directly.
pub fn content_warnings(orig: &str, new: &str) -> Vec<String> {
    use std::collections::HashMap;

    /// Lines too short or non-alphabetic to be worth reasoning about:
    /// `}`, `};`, `#[test]`, blank lines, `Ok(())`. These legitimately repeat,
    /// and flagging them would be pure noise.
    fn meaningful(s: &str) -> bool {
        let t = s.trim();
        t.len() >= 8 && t.chars().any(char::is_alphabetic)
    }

    fn census(src: &str) -> (HashMap<String, usize>, Vec<&str>) {
        let lines: Vec<&str> = src.lines().collect();
        let mut m: HashMap<String, usize> = HashMap::new();
        for l in &lines {
            if meaningful(l) {
                *m.entry(l.trim().to_string()).or_insert(0) += 1;
            }
        }
        (m, lines)
    }

    let (before, _) = census(orig);
    let (after, lines) = census(new);
    let mut out = Vec::new();

    // A line whose occurrence count CHANGED is what matters. Comparing against
    // the original rather than testing for duplicates outright is what keeps
    // this quiet on source that already contains repeated boilerplate: a line
    // present 3 times before and 3 times after is not a new defect.
    for (text, n) in &after {
        if *n < 2 {
            continue;
        }
        let was = before.get(text).copied().unwrap_or(0);
        if was == *n {
            continue; // unchanged; pre-existing repetition, not her doing
        }
        // Report the first line where it shows up, so she can go look.
        let first = lines
            .iter()
            .position(|l| meaningful(l) && l.trim() == text)
            .map(|i| i + 1)
            .unwrap_or(0);
        out.push(format!(
            "{:?} now appears {n} times (was {was}), first at line {first}.\n\
             \x20     This usually means the replacement text re-included a line that \
             was already there, so the original survived as a duplicate. Remove the \
             extra copy before applying.",
            crate::util::truncate(text, 70)
        ));
    }
    out.sort();
    out
}

/// Current state summary.
pub fn status(cfg: &crate::config::SelfPatchConfig) -> String {
    let st = load();
    let enabled = if cfg.enabled { "enabled" } else { "DISABLED" };
    let mut msg = format!(
        "Self-modification: {enabled} (allow_apply: {}). Source: {}\n",
        cfg.allow_apply, cfg.source_dir
    );
    if st.files.is_empty() {
        msg.push_str("No proposal staged.\n");
        return msg;
    }
    msg.push_str(&format!(
        "Proposal: {} file(s), staged {}, validation {}\n",
        st.files.len(),
        st.created,
        match &st.validation {
            Some(v) if v.ok => "PASSED".to_string(),
            Some(_) => "FAILED".to_string(),
            None => "not run".to_string(),
        }
    ));
    for f in &st.files {
        msg.push_str(&format!("  - {}\n", f.rel));
    }
    msg
}

/// Install a validated proposal into the real tree, with a backup.
pub fn apply(cfg: &crate::config::SelfPatchConfig) -> Result<String> {
    if !cfg.enabled {
        anyhow::bail!("self-modification is disabled in config");
    }
    if !cfg.allow_apply {
        anyhow::bail!("[selfpatch] allow_apply = false — proposals can be staged and validated but not installed");
    }
    let st = load();
    if st.files.is_empty() {
        anyhow::bail!("no proposal is staged");
    }
    match &st.validation {
        Some(v) if v.ok => {}
        Some(v) => anyhow::bail!(
            "refusing to apply: validation FAILED (cargo test exit {}). Fix or discard it.",
            v.exit_code
        ),
        None => anyhow::bail!(
            "refusing to apply: this proposal was never validated. Run \
             self_patch {{action:\"validate\"}} first."
        ),
    }
    let root = PathBuf::from(&cfg.source_dir);
    if !root.is_dir() {
        anyhow::bail!("source root not found: {}", root.display());
    }

    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let bdir = backup_dir().join(&stamp);
    std::fs::create_dir_all(&bdir)?;

    for f in &st.files {
        // Re-check protection at apply time: the list may have changed, and a
        // proposal staged under older config must not slip past a new guard.
        let rel = safe_rel(&root, &f.rel)?;
        if is_protected(&rel, &cfg.protected_files) {
            anyhow::bail!("{rel} is protected — refusing to apply");
        }
        let target = root.join(&rel);
        if !target.is_file() {
            anyhow::bail!("{rel} disappeared since staging — refusing to apply");
        }
        // The file must be exactly as it was when the proposal was staged. If it
        // changed underneath us, applying would silently throw away that work.
        if hash_file(&target)? != f.orig_hash {
            anyhow::bail!(
                "{rel} changed on disk since this proposal was staged (someone else \
                 edited it). Refusing to overwrite their work — discard this proposal \
                 and re-propose against the current file."
            );
        }
        // Back up the original BEFORE overwriting.
        let bak = bdir.join(rel.replace('/', "__"));
        std::fs::copy(&target, &bak).with_context(|| format!("failed to back up {rel}"))?;
        std::fs::copy(&f.staged, &target).with_context(|| format!("failed to write {rel}"))?;
    }

    let mut st2 = st.clone();
    st2.backup_id = Some(stamp.clone());
    save(&st2)?;

    Ok(format!(
        "Applied {} file(s) to {}. Backup saved at {}.\n\n\
         The change is in the source but NOT yet running. Tell the user it is \
         applied, and that it takes effect after a rebuild + daemon restart \
         (systemctl --user restart luna-daemon). If anything looks wrong, \
         self_patch {{action:\"rollback\"}} reverts it.",
        st.files.len(),
        root.display(),
        bdir.display()
    ))
}

/// Restore the most recent backup taken by `apply`.
pub fn rollback(cfg: &crate::config::SelfPatchConfig) -> Result<String> {
    let st = load();
    let root = PathBuf::from(&cfg.source_dir);
    let Some(id) = st.backup_id.clone() else {
        anyhow::bail!("no backup to roll back to (nothing has been applied yet)");
    };
    let bdir = backup_dir().join(&id);
    if !bdir.is_dir() {
        anyhow::bail!("backup {id} is missing from {}", bdir.display());
    }
    let mut restored = 0;
    for f in &st.files {
        let rel = safe_rel(&root, &f.rel)?;
        let bak = bdir.join(rel.replace('/', "__"));
        if !bak.is_file() {
            anyhow::bail!("backup for {rel} is missing — refusing a partial rollback");
        }
        // Verify every backup exists BEFORE restoring any, so we never half-roll.
        let _ = &bak;
        restored += 1;
    }
    for f in &st.files {
        let rel = safe_rel(&root, &f.rel)?;
        let bak = bdir.join(rel.replace('/', "__"));
        std::fs::copy(&bak, root.join(&rel)).with_context(|| format!("failed to restore {rel}"))?;
    }
    let mut st2 = st.clone();
    st2.backup_id = None;
    st2.validation = None;
    save(&st2)?;
    Ok(format!(
        "Rolled back {restored} file(s) to the state from {id} (backup kept at {}).",
        bdir.display()
    ))
}

/// Drop the staged proposal. Does NOT touch the real tree.
pub fn discard() -> Result<String> {
    let st = load();
    if st.files.is_empty() {
        return Ok("Nothing to discard.".into());
    }
    for f in &st.files {
        let _ = std::fs::remove_file(&f.staged);
    }
    let n = st.files.len();
    save(&Stored::default())?;
    Ok(format!(
        "Discarded the staged proposal ({n} file(s)). The real tree was never \
         touched by staging."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    // ── line-anchored edits ───────────────────────────────────────────────────
    //
    // The 7B measured 3/3 on line-anchored append but 0/3 on find/replace
    // modify, so this shape is what makes editing viable on a small model.

    const SRC3: &str = "fn alpha() {}\nfn beta() {}\nfn gamma() {}\n";

    #[test]
    fn line_edit_replaces_a_unique_line() {
        let (find, repl) = resolve_line_edit(SRC3, 2, "fn beta() { /* fixed */ }").unwrap();
        assert_eq!(find, "fn beta() {}");
        assert_eq!(repl, "fn beta() { /* fixed */ }");
        let out = apply_edits(SRC3, &[(find, repl)]).unwrap();
        assert!(out.contains("fn beta() { /* fixed */ }"));
        assert!(out.contains("fn alpha() {}"), "other lines must survive");
        assert!(out.contains("fn gamma() {}"));
    }

    /// Appending is impossible with find/replace — there is no existing text
    /// to find — so this is the shape that makes appends possible at all.
    #[test]
    fn line_edit_appends_past_the_last_line() {
        let n = SRC3.lines().count();
        let (find, repl) = resolve_line_edit(SRC3, n + 1, "fn delta() {}").unwrap();
        assert_eq!(find, "fn gamma() {}", "must anchor on real last-line text");
        let out = apply_edits(SRC3, &[(find, repl)]).unwrap();
        assert!(out.ends_with("fn gamma() {}\nfn delta() {}\n"), "got: {out:?}");
    }

    #[test]
    fn line_edit_appends_to_a_file_with_a_trailing_newline() {
        let (find, repl) = resolve_line_edit(SRC3, 4, "fn delta() {}").unwrap();
        let out = apply_edits(SRC3, &[(find, repl)]).unwrap();
        assert!(out.ends_with("fn delta() {}\n"), "trailing newline kept: {out:?}");
    }

    #[test]
    fn line_edit_rejects_out_of_range_lines() {
        let n = SRC3.lines().count();
        assert!(resolve_line_edit(SRC3, 0, "x").is_err(), "0 is not a line");
        assert!(resolve_line_edit(SRC3, n + 2, "x").is_err(), "past append position");
        let e = resolve_line_edit(SRC3, 99, "x").unwrap_err().to_string();
        assert!(e.contains("outside the file"), "got: {e}");
    }

    #[test]
    fn line_edit_rejects_a_blank_target_line() {
        let src = "fn a() {}\n\nfn b() {}\n";
        let e = resolve_line_edit(src, 2, "anything").unwrap_err().to_string();
        assert!(e.contains("blank"), "got: {e}");
    }

    /// When the target line repeats, fold in the previous line to make it
    /// unique — still from disk, never guessed.
    #[test]
    fn line_edit_disambiguates_a_repeated_line_with_its_neighbour() {
        let src = "fn first() {}\nfn dup() {}\nfn mid() {}\nfn dup() {}\n";
        assert_eq!(src.matches("fn dup() {}").count(), 2, "precondition");
        let (find, repl) = resolve_line_edit(src, 4, "fn dup() { /* second */ }").unwrap();
        assert_eq!(find, "fn mid() {}\nfn dup() {}", "must include the previous line");
        let out = apply_edits(src, &[(find, repl)]).unwrap();
        assert_eq!(out.matches("fn dup() {}").count(), 1, "only the 2nd changed");
        assert!(out.contains("fn dup() { /* second */ }"));
    }

    /// If neither the line nor its neighbour is unique we must refuse rather
    /// than guess — the same rule `apply_edits` enforces.
    ///
    /// Note the 3-line case is NOT ambiguous: in "x\nx\nx\n" the pair
    /// "x\nx" occurs exactly once, so disambiguation legitimately succeeds.
    #[test]
    fn line_edit_refuses_when_neither_line_nor_neighbour_is_unique() {
        let src = "x\nx\nx\nx\n";
        // `matches` is non-overlapping, so this is 2, not 3 — either way the
        // point is only that it is NOT unique.
        assert!(src.matches("x\nx").count() > 1, "precondition: pair is ambiguous");
        let e = resolve_line_edit(src, 2, "y").unwrap_err().to_string();
        assert!(e.contains("not unique") || e.contains("could not be located"), "got: {e}");
    }

    /// The complement of the case above, pinned because it is easy to "fix" by
    /// over-tightening: when the neighbour DOES disambiguate, it must work.
    #[test]
    fn line_edit_succeeds_when_only_the_pair_is_needed() {
        let src = "x\nx\nx\n";
        assert_eq!(src.matches("x").count(), 3, "line alone is ambiguous");
        assert_eq!(src.matches("x\nx").count(), 1, "pair is unique");
        let (find, repl) = resolve_line_edit(src, 2, "y").unwrap();
        assert_eq!(find, "x\nx");
        assert_eq!(repl, "x\ny");
    }

    #[test]
    fn line_edit_whose_replacement_equals_the_original_is_refused() {
        let (find, repl) = resolve_line_edit(SRC3, 2, "fn beta() {}").unwrap();
        let e = apply_edits(SRC3, &[(find, repl)]).unwrap_err().to_string();
        assert!(e.contains("no change"), "got: {e}");
    }

    // ── content warnings (what cargo test cannot see) ─────────────────────

    /// The exact defect from the live 2026-09-27 run, reproduced byte for
    /// byte: replacing line 3 with a block that re-included line 4 left the
    /// original line 4 in place, duplicating a sentence in the doc comment.
    /// `cargo test` exits 0 on this. It must still be reported.
    #[test]
    fn warns_about_the_duplicated_line_a_compiler_cannot_see() {
        // Raw, flush-left strings on purpose: a `"...\n\` continuation eats
        // the leading whitespace of the next line, which silently collapses
        // these fixtures and makes the adjacency being tested disappear.
        let orig = "//! util.rs — Tiny shared helpers\n\
\n\
/// Truncate to at most `max` characters without ever splitting a UTF-8\n\
/// character (byte-slicing a string mid-character panics). No allocation.\n\
pub fn truncate(s: &str, max: usize) -> &str {\n";
        // What Luna actually staged: her replace_with re-included line 4, so
        // the original line 4 survived below the inserted block.
        let new = "//! util.rs — Tiny shared helpers\n\
\n\
/// Truncate to at most `max` characters without ever splitting a UTF-8\n\
/// character (byte-slicing a string mid-character panics). No allocation.\n\
///\n\
/// # Arguments\n\
///\n\
/// * `max` - The maximum number of characters to include in the truncated string.\n\
/// character (byte-slicing a string mid-character panics). No allocation.\n\
pub fn truncate(s: &str, max: usize) -> &str {\n";
        assert_eq!(
            new.lines().count(),
            10,
            "fixture must have the duplicated line at 9, got {:?}",
            new.lines().collect::<Vec<_>>()
        );
        let w = content_warnings(orig, new);
        assert_eq!(w.len(), 1, "expected exactly one warning, got {w:?}");
        assert!(w[0].contains("line 4"), "got: {}", w[0]);
        assert!(w[0].contains("now appears 2 times"), "got: {}", w[0]);
        assert!(w[0].contains("character (byte-slicing"), "got: {}", w[0]);
    }

    /// A correct edit of the same shape must produce NO warning. Without this,
    /// the check would just be noise that gets ignored.
    #[test]
    fn a_correct_edit_produces_no_warning() {
        let orig = "fn alpha() {}\nfn beta() {}\n";
        let new = "/// Doc for beta.\nfn beta() {}\n";
        assert!(content_warnings(orig, new).is_empty(), "{:?}", content_warnings(orig, new));
    }

    /// Lines that legitimately repeat must not be flagged, or the check
    /// cries wolf and both Luna and the user learn to skip it.
    ///
    /// Compared against THEMSELVES, because these warnings are about changes:
    /// source that already contains repeated boilerplate, left alone, is not a
    /// defect. (Comparing against an empty original would be meaningless —
    /// every line would count as newly introduced.)
    #[test]
    fn repeated_boilerplate_is_not_flagged() {
        for src in [
            "fn a() {\n    let x = 1;\n}\nfn b() {\n    let x = 1;\n}\n", // same, not adjacent
            "}\n}\n",                                 // closing braces
            "\n\n",                                     // blank lines
            "#[test]\n#[test]\n",                       // attribute
            "    Ok(())\n    Ok(())\n",                 // short line
            "    let total = compute_total(&items)?;\n    let total = compute_total(&items)?;\n",
        ] {
            assert!(
                content_warnings(src, src).is_empty(),
                "false positive on {src:?}: {:?}",
                content_warnings(src, src)
            );
        }
    }

    /// A line that WAS unique and is now duplicated is the defect, adjacent or
    /// not. This is the case that adjacency-only checking misses.
    #[test]
    fn a_newly_duplicated_line_is_flagged_even_when_not_adjacent() {
        let orig = "aaa bbb ccc\nxxx yyy zzz\n";
        let new = "aaa bbb ccc\nxxx yyy zzz\nqqq rrr sss\naaa bbb ccc\n";
        let w = content_warnings(orig, new);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("now appears 2 times (was 1)"), "got: {}", w[0]);
    }

    #[test]
    fn a_repeated_adjacent_meaningful_line_is_flagged() {
        let src = "    let total = compute_total(&items)?;\n    let total = compute_total(&items)?;\n";
        let w = content_warnings("", src);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("line 1"), "got: {}", w[0]);
    }

    #[test]
    fn empty_and_single_line_content_is_handled() {
        assert!(content_warnings("", "").is_empty());
        assert!(content_warnings("", "only one line").is_empty());
    }

    /// An older `proposal.json` has no `warnings` key; it must still load
    /// rather than failing deserialization at the worst possible moment.
    #[test]
    fn validation_without_a_warnings_field_still_deserializes() {
        let old = r#"{"ok":true,"exit_code":0,"summary":"ok","at":"2026-09-27"}"#;
        let v: Validation = serde_json::from_str(old).expect("legacy state must load");
        assert!(v.ok);
        assert!(v.warnings.is_empty());
    }

    #[test]
    fn line_edit_works_on_an_empty_file() {
        let (find, repl) = resolve_line_edit("", 1, "fn only() {}").unwrap();
        assert_eq!(find, "");
        assert_eq!(repl, "fn only() {}");
    }

    /// A line-anchored edit must never silently no-op.
    #[test]
    fn line_edit_that_would_not_change_anything_is_refused() {
        let (find, repl) = resolve_line_edit(SRC3, 2, "fn beta() {}").unwrap();
        assert!(apply_edits(SRC3, &[(find, repl)]).is_err());
    }

    /// End-to-end through `propose`: the line shape must stage real content,
    /// not silently do nothing. Mirrors the existing edits-shape lifecycle test.
    #[tokio::test]
    #[ignore = "touches the real staging dir"]
    async fn the_line_edits_shape_works_end_to_end() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        let scratch_rel = "src/selfpatch_e2e_lines.rs";
        let scratch = root.join(scratch_rel);
        std::fs::write(&scratch, "fn one() {}\nfn two() {}\n").unwrap();
        read_file(&root, scratch_rel).unwrap();
        propose(
            &cfg,
            &[ChangeSpec {
                file: scratch_rel.into(),
                line_edits: vec![(3, "fn three() {}".into())],
                ..Default::default()
            }],
            "append via line anchor",
        )
        .await
        .unwrap();
        let review = review(&cfg).unwrap();
        assert!(review.contains("fn three() {}"), "staged diff was: {review}");
        // Discard on the way out. A staged proposal BLOCKS every later proposal
        // ("a proposal is already staged"), so leaving one behind makes the
        // next real self-patch fail for a reason unrelated to it — this test's
        // residue blocked a live run on 2026-09-27.
        discard().unwrap();
        let _ = std::fs::remove_file(&scratch);
    }

    /// End-to-end through the REAL `validate()`: a proposal that compiles
    /// cleanly but re-included a neighbouring line must NOT come back as a
    /// bare "PASSED".
    ///
    /// This is the case that actually happened on 2026-09-27. The staged file
    /// below is byte-for-byte the shape she produced: the replacement for line 2
    /// quoted lines 2 AND 3, so the untouched original line 3 survived below
    /// the insertion. `cargo test` exits 0 on it, because `fn gamma() {}`
    /// appearing twice is valid Rust and the scratch file is not even declared
    /// as a module, so nothing compiles it at all.
    ///
    /// Slow by nature (it rsyncs the tree and runs `cargo test`), hence
    /// `#[ignore]`. It is the only test that covers the validate -> warning ->
    /// tool-message wiring end to end, so it is worth running deliberately.
    #[tokio::test]
    #[ignore = "touches the real staging dir and runs cargo test"]
    async fn validate_reports_a_duplicate_that_the_compiler_cannot_see() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        let scratch_rel = "src/selfpatch_e2e_dup.rs";
        let scratch = root.join(scratch_rel);
        std::fs::write(&scratch, "fn alpha() {}\nfn beta() {}\nfn gamma() {}\n").unwrap();
        read_file(&root, scratch_rel).unwrap();

        propose(
            &cfg,
            &[ChangeSpec {
                file: scratch_rel.into(),
                // The defect: this covers line 2 but quotes line 3 as well, so
                // the real line 3 is left behind underneath.
                line_edits: vec![(2, "fn beta() {}\nfn gamma() {}".into())],
                ..Default::default()
            }],
            "reproduce the duplicate-line defect",
        )
        .await
        .unwrap();

        // Confirm the staged content really is the broken shape, otherwise
        // this test could pass for the wrong reason.
        let st = load();
        let staged = std::fs::read_to_string(&st.files[0].staged).unwrap();
        assert_eq!(
            staged.matches("fn gamma() {}").count(),
            2,
            "fixture must actually be duplicated, got: {staged:?}"
        );

        let out = validate(&cfg).await.unwrap();

        assert!(
            out.contains("content problem"),
            "validate hid a content defect behind a compile pass. Output was:\n{out}"
        );
        assert!(
            out.contains("fn gamma"),
            "the duplicated line must be named, not just counted. Output was:\n{out}"
        );
        assert!(
            out.contains("```diff"),
            "validate must show the diff; the model cannot check what it cannot see"
        );
        // The honest framing: the compile result is still reported truthfully.
        assert!(
            out.contains("cargo test"),
            "the compile result must still be reported, not replaced by the warning"
        );
        // `review` must carry the warning too, so it cannot be laundered by
        // reading `review` instead of `validate`.
        let rev = review(&cfg).unwrap();
        assert!(
            rev.contains("CONTENT WARNINGS"),
            "review dropped the content warnings. review was:\n{rev}"
        );

        discard().unwrap();
        let _ = std::fs::remove_file(&scratch);
    }



    fn root() -> PathBuf {
        PathBuf::from("/home/netrunner/Projects/luna-stable")
    }

    #[test]
    fn accepts_a_normal_relative_path() {
        assert_eq!(safe_rel(&root(), "src/daemon/mod.rs").unwrap(), "src/daemon/mod.rs");
    }

    #[test]
    fn rejects_absolute_paths() {
        assert!(safe_rel(&root(), "/etc/passwd").is_err());
        assert!(safe_rel(&root(), "/home/netrunner/Projects/luna-stable/src/util.rs").is_err());
    }

    #[test]
    fn rejects_traversal() {
        assert!(safe_rel(&root(), "../../../etc/shadow").is_err());
        assert!(safe_rel(&root(), "src/../../outside.rs").is_err());
        assert!(safe_rel(&root(), "..").is_err());
    }

    #[test]
    fn rejects_empty_path() {
        assert!(safe_rel(&root(), "   ").is_err());
    }

    #[test]
    fn protected_list_blocks_exact_and_nested() {
        let list: Vec<String> = ["src/config.rs", "src/llm", "Cargo.toml"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(is_protected("src/config.rs", &list));
        assert!(is_protected("src/llm/react.rs", &list), "nested under a protected dir");
        assert!(is_protected("Cargo.toml", &list));
        // Not protected:
        assert!(!is_protected("src/tools/pkgupdate.rs", &list));
        assert!(!is_protected("src/llmfoo/x.rs", &list), "prefix must match a path segment");
        assert!(!is_protected("src/config.rsx", &list), "exact match, not a string prefix");
    }

    #[test]
    fn the_safety_core_is_actually_protected_by_default() {
        // The defaults must cover her constitution, routing, and tool-call guard.
        let cfg = crate::config::SelfPatchConfig::default();
        for critical in [
            "src/config.rs",
            "src/llm/escalation.rs",
            "src/llm/react.rs",
            "src/tools/selfpatch.rs",
            "Cargo.toml",
        ] {
            assert!(
                is_protected(critical, &cfg.protected_files),
                "{critical} must be protected by default"
            );
        }
    }

    #[tokio::test]
    #[ignore = "touches real files; run explicitly with -- --ignored"]
    async fn proposing_without_reading_first_is_refused() {
        // The structural grounding guard: a model that edits from memory
        // invents paths and 'find' strings. This is the whole point.
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        let scratch = root.join("src/selfpatch_e2e_receipt.rs");
        std::fs::write(&scratch, "fn x() {}\n").unwrap();
        let rel = "src/selfpatch_e2e_receipt.rs";

        // 1. propose WITHOUT reading -> refused
        let err = propose(
            &cfg,
            &[ChangeSpec {
                file: rel.into(),
                content: None,
                edits: vec![("fn x() {}".into(), "fn x() { /* fixed */ }".into())],
                ..Default::default()
            }],
            "e2e: no read",
        )
        .await
        .expect_err("proposing without reading must be refused");
        assert!(err.to_string().contains("have not read"), "got: {err}");

        // 2. an invented file is refused before that even
        let err = propose(
            &cfg,
            &[ChangeSpec {
                file: "src/tools/session_logger.rs".into(),
                content: None,
                edits: vec![("x".into(), "y".into())],
                ..Default::default()
            }],
            "e2e: invented path",
        )
        .await
        .expect_err("an invented file must be refused");
        assert!(err.to_string().contains("does not exist"), "got: {err}");

        // 3. read it, then the identical edit is accepted
        let body = read_file(&root, rel).expect("read should succeed and issue a receipt");
        assert!(body.contains("fn x()"), "read must return the real content");
        propose(
            &cfg,
            &[ChangeSpec {
                file: rel.into(),
                content: None,
                edits: vec![("fn x() {}".into(), "fn x() { /* fixed */ }".into())],
                ..Default::default()
            }],
            "e2e: after reading",
        )
        .await
        .expect("propose after a real read must be allowed");

        discard().unwrap();
        let _ = std::fs::remove_file(&scratch);
    }

    #[tokio::test]
    #[ignore = "touches real files; run explicitly with -- --ignored"]
    async fn file_listing_shows_real_files_not_guesses() {
        let cfg = crate::config::SelfPatchConfig::default();
        let out = list_source_files(&PathBuf::from(&cfg.source_dir), "");
        assert!(out.contains("src/config.rs"), "must list a known real file");
        assert!(out.contains("src/tools/selfpatch.rs"), "must list her own module");
        assert!(!out.contains("target/"), "must not walk into target/");
    }

    #[test]
    fn a_filter_narrows_the_listing_without_guessing() {
        let cfg = crate::config::SelfPatchConfig::default();
        let root = PathBuf::from(&cfg.source_dir);
        let all = list_source_files(&root, "");
        let only = list_source_files(&root, "selfpatch");
        assert!(all.contains("src/tools/selfpatch.rs"), "sanity: unfiltered");
        assert!(only.contains("src/tools/selfpatch.rs"), "filter must match");
        assert!(!only.contains("src/config.rs"), "filter must EXCLUDE others");
        // A filter matching nothing must say so and point back at the real list,
        // rather than returning an empty result she might read as "no files".
        let miss = list_source_files(&root, "zzz_no_such_file_zzz");
        assert!(miss.contains("No .rs source file matches"), "got: {miss}");
        assert!(miss.contains("action=files"), "must redirect her: {miss}");
    }

    #[test]
    fn a_targeted_edit_replaces_exactly_once() {
        let src = "fn a() { 1 }\nfn b() { 2 }\n";
        let out = apply_edits(src, &[("fn b() { 2 }".into(), "fn b() { 3 }".into())]).unwrap();
        assert_eq!(out, "fn a() { 1 }\nfn b() { 3 }\n");
    }

    #[test]
    fn multiple_edits_apply_in_order() {
        let src = "alpha beta gamma\n";
        let out = apply_edits(
            src,
            &[
                ("beta".into(), "B".into()),
                ("gamma".into(), "G".into()),
            ],
        )
        .unwrap();
        assert_eq!(out, "alpha B G\n");
    }

    #[test]
    fn an_ambiguous_find_is_refused_not_guessed() {
        // Two identical lines: replacing "x" would hit the wrong one.
        let src = "let x = 1;\nlet x = 1;\n";
        let err = apply_edits(src, &[("let x = 1;".into(), "let x = 2;".into())])
            .expect_err("ambiguous find must be refused");
        assert!(err.to_string().contains("appears 2 times"), "got: {err}");
    }

    #[test]
    fn a_stale_find_is_refused_rather_than_mangling_the_file() {
        // She remembered text that no longer exists — the common real failure.
        let src = "fn current() { 1 }\n";
        let err = apply_edits(src, &[("fn old_name()".into(), "x".into())])
            .expect_err("a find that does not exist must be refused");
        assert!(err.to_string().contains("does not appear"), "got: {err}");
    }

    #[test]
    fn empty_find_and_no_op_edits_are_refused() {
        let src = "hello\n";
        assert!(apply_edits(src, &[]).is_err(), "no edits at all");
        assert!(apply_edits(src, &[("".into(), "x".into())]).is_err(), "empty find");
        assert!(
            apply_edits(src, &[("hello".into(), "hello".into())]).is_err(),
            "an edit that changes nothing is not a proposal"
        );
    }

    #[test]
    fn fingerprint_detects_change_and_ignores_no_change() {
        assert_eq!(fingerprint(b"same"), fingerprint(b"same"));
        assert_ne!(fingerprint(b"same"), fingerprint(b"different"));
        assert_ne!(fingerprint(b""), fingerprint(b"x"));
    }

    #[test]
    fn a_proposal_is_not_considered_validated_until_it_runs() {
        // Guards the apply-time precondition: a fresh Stored has no validation.
        let st = Stored::default();
        assert!(st.validation.is_none());
        assert!(st.files.is_empty());
    }

    #[test]
    fn raw_writes_into_her_own_source_are_refused() {
        let cfg = crate::config::SelfPatchConfig::default();
        let root = cfg.source_dir.clone();
        // A real source file:
        assert!(blocks_raw_write(&cfg, &format!("{root}/src/tools/pkgupdate.rs")).is_some());
        // A new file she is inventing:
        assert!(blocks_raw_write(&cfg, &format!("{root}/src/tools/brand_new.rs")).is_some());
        // The root itself:
        assert!(blocks_raw_write(&cfg, &root).is_some());
        // Via ~ expansion:
        let rest = root.trim_start_matches(&format!("{}/", std::env::var("HOME").unwrap_or_default()));
        assert!(blocks_raw_write(&cfg, &format!("~/{rest}/src/util.rs")).is_some());
    }

    #[test]
    fn raw_writes_elsewhere_are_still_allowed() {
        let cfg = crate::config::SelfPatchConfig::default();
        // The guard must not break her normal file-writing work.
        assert!(blocks_raw_write(&cfg, "/tmp/notes.txt").is_none());
        assert!(blocks_raw_write(&cfg, "/home/netrunner/Projects/sih/src/main.py").is_none());
        assert!(blocks_raw_write(&cfg, "~/.config/fish/config.fish").is_none());
    }

    #[test]
    fn guard_is_inert_when_selfpatch_is_disabled() {
        let cfg = crate::config::SelfPatchConfig {
            enabled: false,
            ..Default::default()
        };
        let root = cfg.source_dir.clone();
        assert!(blocks_raw_write(&cfg, &format!("{root}/src/util.rs")).is_none());
    }

    // ── End-to-end lifecycle ──────────────────────────────────────────────
    // These actually shell out to rsync + cargo, so they are opt-in:
    //   cargo test selfpatch -- --ignored --nocapture
    //
    // The point of the first one is the whole safety claim: a proposal that
    // breaks a real, compiled source file must be CAUGHT, and the real tree
    // must be byte-identical afterwards.
    // ── Dispatch ──────────────────────────────────────────────────────────
    // The lifecycle tests call the functions directly. This one goes through
    // the real `execute()` entry point with a real JSON tool call, which is
    // the path a model actually takes — so it proves the match arm and the
    // argument parsing are wired up, not just the functions.
    #[tokio::test]
    #[ignore = "touches real config/state; run explicitly with -- --ignored"]
    async fn the_tool_dispatches_through_execute() {
        use crate::llm::ollama::{ToolCall, ToolCallFunction};
        let _ = discard();
        let cfg = crate::config::LunaConfig::default();

        // action=status is the safe default path.
        let call = ToolCall {
            function: ToolCallFunction {
                name: "self_patch".into(),
                arguments: serde_json::json!({ "action": "status" }),
            },
        };
        let out = crate::tools::execute(&call, &cfg).await.expect("status should dispatch");
        assert!(out.contains("Self-modification"), "got: {out}");

        // An unknown action must fall back to status, not blow up.
        let call = ToolCall {
            function: ToolCallFunction {
                name: "self_patch".into(),
                arguments: serde_json::json!({ "action": "nonsense" }),
            },
        };
        crate::tools::execute(&call, &cfg).await.expect("unknown action should not error");

        // propose with no changes must be rejected cleanly, not panic.
        let call = ToolCall {
            function: ToolCallFunction {
                name: "self_patch".into(),
                arguments: serde_json::json!({ "action": "propose" }),
            },
        };
        let err = crate::tools::execute(&call, &cfg).await.expect_err("empty propose must fail");
        assert!(
            err.to_string().contains("needs a 'changes' array"),
            "the empty-propose error must show her the right shape, got: {err}"
        );
    }

    #[tokio::test]
    #[ignore = "invokes cargo; run explicitly with -- --ignored"]
    async fn a_breaking_proposal_is_caught_and_never_touches_the_real_tree() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        let util = root.join("src/util.rs");
        let before = std::fs::read_to_string(&util).expect("src/util.rs must exist");

        // 0. she must read the file first (structural grounding guard)
        read_file(&root, "src/util.rs").expect("read issues the receipt");

        // 1. propose a change that genuinely breaks compilation
        let broken = format!("{before}\nfn selfpatch_e2e_broken() -> i32 {{ \"nope\" }}\n");
        let out = propose(
            &cfg,
            &[ChangeSpec { file: "src/util.rs".into(), content: Some(broken), edits: vec![], ..Default::default() }],
            "e2e: deliberate compile break",
        )
        .await
        .expect("propose should stage the change");
        assert!(out.contains("Nothing in the real tree has changed"), "got: {out}");
        assert!(out.contains("```diff"), "propose must show a diff, got: {out}");

        // 2. staging must not have modified the real file
        assert_eq!(
            std::fs::read_to_string(&util).unwrap(),
            before,
            "propose modified the real source"
        );

        // 3. validation must FAIL
        let v = validate(&cfg).await.expect("validate should run");
        assert!(v.contains("FAILED"), "a compile break must fail validation, got: {v}");

        // 4. apply must REFUSE a failed proposal
        let err = apply(&cfg).expect_err("apply must refuse a failed proposal");
        assert!(
            err.to_string().contains("refusing"),
            "apply should refuse, got: {err}"
        );

        // 5. the real source is still byte-identical
        assert_eq!(
            std::fs::read_to_string(&util).unwrap(),
            before,
            "the real source was modified by a failed proposal"
        );

        discard().unwrap();
    }

    #[tokio::test]
    #[ignore = "invokes cargo; run explicitly with -- --ignored"]
    async fn a_clean_proposal_validates_applies_and_rolls_back() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        // A scratch file cargo never compiles (not declared as a module), so
        // this isolates the apply/rollback mechanics from build behaviour.
        let scratch = root.join("src/selfpatch_e2e_scratch.rs");
        std::fs::write(&scratch, "original\n").unwrap();
        let before = std::fs::read_to_string(&scratch).unwrap();

        read_file(&root, "src/selfpatch_e2e_scratch.rs").unwrap();
        propose(
            &cfg,
            &[ChangeSpec {
                file: "src/selfpatch_e2e_scratch.rs".into(),
                content: Some("improved\n".into()),
                edits: vec![],
                ..Default::default()
            }],
            "e2e: happy path",
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&scratch).unwrap(), before, "staging wrote early");

        assert!(validate(&cfg).await.unwrap().contains("PASSED"), "clean change must pass");

        let applied = apply(&cfg).expect("apply should succeed after a green validation");
        assert!(applied.contains("Applied"), "got: {applied}");
        assert!(
            std::fs::read_to_string(&scratch).unwrap() == "improved\n",
            "apply did not install the staged content"
        );

        let back = rollback(&cfg).expect("rollback should succeed");
        assert!(back.contains("Rolled back"), "got: {back}");
        assert_eq!(
            std::fs::read_to_string(&scratch).unwrap(),
            before,
            "rollback did not restore the original"
        );

        discard().unwrap();
        let _ = std::fs::remove_file(&scratch);
    }

    #[tokio::test]
    #[ignore = "invokes cargo; run explicitly with -- --ignored"]
    async fn a_proposal_goes_stale_if_the_file_changes_underneath_it() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        let scratch = root.join("src/selfpatch_e2e_stale.rs");
        std::fs::write(&scratch, "original\n").unwrap();

        read_file(&root, "src/selfpatch_e2e_stale.rs").unwrap();
        propose(
            &cfg,
            &[ChangeSpec { file: "src/selfpatch_e2e_stale.rs".into(), content: Some("mine\n".into()), edits: vec![], ..Default::default() }],
            "e2e: stale detection",
        )
        .await
        .unwrap();
        assert!(validate(&cfg).await.unwrap().contains("PASSED"));

        // Someone else edits the file after staging + validation.
        std::fs::write(&scratch, "someone elses work\n").unwrap();

        let err = apply(&cfg).expect_err("apply must refuse a stale proposal");
        assert!(
            err.to_string().contains("changed on disk"),
            "expected a staleness refusal, got: {err}"
        );
        // And their work is intact.
        assert_eq!(
            std::fs::read_to_string(&scratch).unwrap(),
            "someone elses work\n",
            "a stale apply destroyed someone else's edit"
        );

        discard().unwrap();
        let _ = std::fs::remove_file(&scratch);
    }

    #[tokio::test]
    #[ignore = "invokes cargo; run explicitly with -- --ignored"]
    async fn the_edits_shape_works_end_to_end() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let root = PathBuf::from(&cfg.source_dir);
        let scratch = root.join("src/selfpatch_e2e_edits.rs");
        std::fs::write(&scratch, "fn alpha() { 1 }\nfn beta() { 2 }\n").unwrap();
        read_file(&root, "src/selfpatch_e2e_edits.rs").unwrap();

        // A stale find must be refused and change nothing.
        let err = propose(
            &cfg,
            &[ChangeSpec {
                file: "src/selfpatch_e2e_edits.rs".into(),
                content: None,
                edits: vec![("fn nonexistent()".into(), "x".into())],
                ..Default::default()
            }],
            "e2e: stale find",
        )
        .await
        .expect_err("a stale find must be refused");
        assert!(err.to_string().contains("does not appear"), "got: {err}");

        // A real find must stage, validate, and apply.
        read_file(&root, "src/selfpatch_e2e_edits.rs").unwrap();
        propose(
            &cfg,
            &[ChangeSpec {
                file: "src/selfpatch_e2e_edits.rs".into(),
                content: None,
                edits: vec![("fn beta() { 2 }".into(), "fn beta() { 42 }".into())],
                ..Default::default()
            }],
            "e2e: real edit",
        )
        .await
        .unwrap();
        // Staging must not have written to the real file yet.
        assert_eq!(
            std::fs::read_to_string(&scratch).unwrap(),
            "fn alpha() { 1 }\nfn beta() { 2 }\n",
            "staging leaked into the real file"
        );

        assert!(validate(&cfg).await.unwrap().contains("PASSED"));
        apply(&cfg).expect("apply should succeed");
        assert_eq!(
            std::fs::read_to_string(&scratch).unwrap(),
            "fn alpha() { 1 }\nfn beta() { 42 }\n",
            "the targeted edit was not applied correctly"
        );

        rollback(&cfg).unwrap();
        assert_eq!(std::fs::read_to_string(&scratch).unwrap(), "fn alpha() { 1 }\nfn beta() { 2 }\n");
        discard().unwrap();
        let _ = std::fs::remove_file(&scratch);
    }

    #[tokio::test]
    #[ignore = "invokes cargo; run explicitly with -- --ignored"]
    async fn a_protected_file_cannot_even_be_proposed() {
        let cfg = crate::config::SelfPatchConfig::default();
        let _ = discard();
        let err = propose(
            &cfg,
            &[ChangeSpec { file: "src/llm/react.rs".into(), content: Some("// hijacked".into()), edits: vec![], ..Default::default() }],
            "e2e: try to rewrite her own guardrail",
        )
        .await
        .expect_err("her guardrails must be unproposable");
        assert!(err.to_string().contains("protected"), "got: {err}");
        // and nothing was staged
        assert!(load().files.is_empty(), "a refused proposal must leave no state");
    }
}
