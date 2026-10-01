//! tools/verify.rs — is this file actually valid, or just plausible?
//!
//! A model that writes source code into a `write_file` call sometimes
//! double-escapes the newlines: the `content` arrives holding the two
//! characters `\` and `n` instead of line breaks. Observed 2026-10-01 from
//! `dagbs/qwen2.5-coder-7b-instruct-abliterated`, 1 of 3 samples: a 552-byte
//! "reverse shell" that contained `socket`, `def ` and `import ` — every token a
//! plausibility check looks for — and could not be compiled.
//!
//! The temptation is to rewrite the content when it looks escaped. That is a
//! guess, and the guess has a real failure mode: a minified `.json` file
//! legitimately contains escaped newlines, and unescaping it corrupts the file
//! that was correct.
//!
//! So the repair is gated on evidence instead of on appearance: the file as
//! written must *fail* verification, and the unescaped version must *pass* it.
//! If neither can be checked, nothing is touched and the user is told the file
//! may be broken rather than being handed a silent guess.

use std::path::Path;

/// What a verification check concluded about a piece of source.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Verdict {
    /// Parses / is syntactically valid.
    Valid,
    /// Does not parse. Carries the checker's own complaint.
    Invalid,
    /// No checker exists for this file type, so nothing can be claimed either
    /// way. Distinct from `Valid` on purpose: "we did not look" must never be
    /// reported as "it is fine".
    Unverifiable,
}

/// Can this content be checked at all, for this path?
fn checker_for(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|e| e.to_str()) {
        // Compiled by the interpreter. Cheap, authoritative.
        Some("py") => Some("python3 -m py_compile"),
        // Deliberately NOT shell, despite `bash -n` being available.
        //
        // `bash -n` accepts `echo one\necho two` happily: the literal backslash-n
        // is a valid argument to `echo`, so the damaged file parses. The check
        // would report every escaped shell script as Valid, the repair would
        // never fire, and the guard would look like it was covering `.sh` while
        // covering nothing. Absence of evidence is not evidence, so the honest
        // entry is no checker at all.
        Some("sh" | "bash" | "zsh") => None,
        // Never: escaped newlines are *required* inside JSON string values, so
        // "unescaping" one of these is guaranteed to break a correct file.
        // Listed explicitly so the exclusion is a decision on record rather than
        // an absence someone later mistakes for an oversight.
        Some("json") => None,
        _ => None,
    }
}

/// Run the syntax check for `path` over `content`.
///
/// Writes to a scratch file rather than the real path: this runs *before* the
/// write is committed, and a checker that truncates the user's file to test it
/// would be worse than the bug it is looking for.
fn verify_content(path: &Path, content: &str, checker: &str) -> Verdict {
    let scratch = std::env::temp_dir().join(format!(
        "luna_verify_{}_{}{}",
        std::process::id(),
        // Distinct per call, so two checks cannot collide.
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_default()
    ));
    if std::fs::write(&scratch, content).is_err() {
        return Verdict::Unverifiable;
    }
    let mut cmd = match checker {
        "python3 -m py_compile" => {
            let mut c = std::process::Command::new("python3");
            c.args(["-m", "py_compile"]).arg(&scratch);
            c
        }
        "bash -n" => {
            let mut c = std::process::Command::new("bash");
            c.args(["-n"]).arg(&scratch);
            c
        }
        _ => return Verdict::Unverifiable,
    };
    let verdict = match cmd.output() {
        Ok(o) if o.status.success() => Verdict::Valid,
        Ok(_) => Verdict::Invalid,
        Err(_) => Verdict::Unverifiable,
    };
    std::fs::remove_file(&scratch).ok();
    verdict
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Turn the two-character sequence `\n` into real line breaks.
fn unescape_newlines(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}

/// Does this content look double-escaped?
///
/// Strict on purpose. Zero real newlines *and* several escape sequences. A file
/// with even one real newline is a normal multi-line file and is left alone, so
/// the only thing this can ever fire on is content that has no line breaks at
/// all — which a multi-line source file cannot be.
fn looks_double_escaped(content: &str) -> bool {
    !content.contains('\n') && content.matches("\\n").count() >= 3
}

/// Outcome of the pre-write check.
#[derive(Debug, PartialEq)]
pub enum PreWrite {
    /// Content is fine as-is, or cannot be judged. Write it unchanged.
    Clean,
    /// Content was escaped and the unescaped form is verifiably better.
    Repaired(String),
    /// Content was escaped and the unescaped form is *also* broken. Writing the
    /// original and saying so beats writing a guess.
    StillBroken(String),
}

/// Check content destined for `path`, repairing double-escaped newlines when
/// — and only when — the repair is provably an improvement.
pub fn check_before_write(path: &str, content: &str) -> PreWrite {
    let p = Path::new(path);
    let Some(checker) = checker_for(p) else {
        return PreWrite::Clean;
    };
    if !looks_double_escaped(content) {
        return PreWrite::Clean;
    }
    // Pointless work unless the current form is actually broken.
    if verify_content(p, content, checker) != Verdict::Invalid {
        return PreWrite::Clean;
    }
    let fixed = unescape_newlines(content);
    match verify_content(p, &fixed, checker) {
        Verdict::Valid => PreWrite::Repaired(fixed),
        // Unverifiable here means the fix could not be proven, so it is not
        // applied. Guessing is what produced the bug in the first place.
        _ => PreWrite::StillBroken(content.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD_PY: &str = "import socket\n\ndef main():\n    print('x')\n";
    const ESCAPED_PY: &str =
        "import socket\\nimport subprocess\\n\\ndef reverse_shell(ip, port):\\n    s = 1\\n";

    #[test]
    fn repairs_an_escaped_file_that_is_provably_broken() {
        // Written as the model emitted it, to a real path.
        let dir = std::env::temp_dir().join(format!("luna_vf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("poc.py");

        // Confirm the premise: the escaped form does not compile.
        assert_eq!(verify_content(&target, ESCAPED_PY, "python3 -m py_compile"), Verdict::Invalid);

        match check_before_write(target.to_str().unwrap(), ESCAPED_PY) {
            PreWrite::Repaired(fixed) => {
                // The repair must actually be valid Python, and must contain
                // real newlines — otherwise the fix is cosmetic.
                assert!(fixed.contains('\n'));
                assert_eq!(
                    verify_content(&target, &fixed, "python3 -m py_compile"),
                    Verdict::Valid
                );
                assert_eq!(fixed, unescape_newlines(ESCAPED_PY));
            }
            other => panic!("expected Repaired, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A correct multi-line file must pass through untouched. This is the one
    /// that matters most: a repair that fires on good input corrupts output
    /// that was already right.
    #[test]
    fn a_correct_multiline_file_is_untouched() {
        assert_eq!(check_before_write("/tmp/x.py", GOOD_PY), PreWrite::Clean);
    }

    /// A minified JSON file legitimately contains escaped newlines and must be
    /// preserved — unescaping it would corrupt a correct file.
    #[test]
    fn json_is_never_repaired() {
        let json = "{\"body\": \"a\\nb\\nc\\nd\"}";
        assert_eq!(check_before_write("/tmp/x.json", json), PreWrite::Clean);
    }

    /// With no checker for the extension, nothing is claimed and nothing is
    /// changed. Reporting "probably fine" would be asserting something nobody
    /// checked.
    #[test]
    fn uncheckable_types_are_left_alone() {
        assert_eq!(
            check_before_write("/tmp/x.txt", ESCAPED_PY),
            PreWrite::Clean
        );
    }

    /// Real newlines present means it is a normal file, not an escaped one.
    #[test]
    fn any_real_newline_disqualifies_the_repair() {
        let mixed = "first line\nsecond\\nthird\\nfourth";
        assert!(!looks_double_escaped(mixed));
        assert_eq!(check_before_write("/tmp/x.py", mixed), PreWrite::Clean);
    }

    /// Only escaped content counts, and only in bulk.
    #[test]
    fn a_stray_escape_in_a_one_line_file_is_not_enough() {
        assert!(!looks_double_escaped("value is a\\nb"));
        assert!(!looks_double_escaped("a\\nb\\nc"));
        assert!(looks_double_escaped("a\\nb\\nc\\nd"));
    }

    /// Escaped and still broken either way: report it, do not guess.
    #[test]
    fn escaped_and_unfixable_is_reported_not_guessed() {
        let dir = std::env::temp_dir().join(format!("luna_vf2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("bad.py");
        // Escaped newlines and genuinely invalid syntax underneath.
        let broken = "def f(:\\n  pass\\n  ???\\n";
        match check_before_write(target.to_str().unwrap(), broken) {
            PreWrite::StillBroken(s) => assert_eq!(s, broken),
            other => panic!("expected StillBroken, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Shell scripts are NOT checked, and the test records why. `bash -n` accepts
    /// `echo one\necho two` — the literal `\n` is a legal argument — so it
    /// reports damaged shell scripts as valid and the repair would never fire.
    /// Claiming coverage here would be a guard that looks armed and is not.
    #[test]
    fn shell_scripts_are_declared_unverifiable_not_claimed() {
        let dir = std::env::temp_dir().join(format!("luna_vf3_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("p.sh");
        let escaped = "echo one\\necho two\\necho three\\n";
        assert_eq!(
            verify_content(&target, escaped, "bash -n"),
            Verdict::Valid,
            "if this ever becomes Invalid the exclusion above needs revisiting"
        );
        assert_eq!(
            check_before_write(target.to_str().unwrap(), escaped),
            PreWrite::Clean,
            "shell is left alone"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}