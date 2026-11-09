//! Repository-scoped context and explicit verification; commands are never guessed or executed during context lookup.
use anyhow::{Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::io::Write;

const FILES: &[&str] = &[
    "AGENTS.md", ".codex-state.md", "README.md", "Cargo.toml", "package.json",
    "pyproject.toml", "pytest.ini", "Makefile", "requirements.txt", "setup.py",
    "go.mod", "tsconfig.json", "CMakeLists.txt",
];

fn root(path: &str) -> Result<PathBuf> {
    let start = std::fs::canonicalize(path).context("Project path does not exist")?;
    let start = if start.is_file() { start.parent().unwrap_or(&start).to_path_buf() } else { start };
    let mut manifest = None;
    for dir in start.ancestors() {
        if dir.join(".git").exists() { return Ok(dir.to_path_buf()); }
        if manifest.is_none() && ["Cargo.toml","package.json","pyproject.toml","go.mod","requirements.txt","Makefile"].iter().any(|f| dir.join(f).is_file()) {
            manifest = Some(dir.to_path_buf());
        }
    }
    if let Some(root) = manifest { return Ok(root); }
    if start.is_dir() { return Ok(start); }
    anyhow::bail!("No repository or project manifest found; provide a project directory")
}

pub fn context(path: &str) -> Result<String> {
    let root = root(path)?;
    let mut hash = Sha256::new();
    hash.update(root.to_string_lossy().as_bytes());
    let mut documents = Vec::new();
    for name in FILES {
        let file = root.join(name);
        if let Ok(meta) = file.metadata() {
            if !meta.is_file() || meta.len() > 1024 * 1024 { continue; }
            let body = std::fs::read_to_string(file)?;
            hash.update(name.as_bytes());
            hash.update(body.as_bytes());
            documents.push(json!({"file":name,"excerpt":crate::util::truncate(&body, 3000)}));
        }
    }
    // Hash tracked and nonignored source, so changes invalidate learned project notes.
    if let Ok(output) = std::process::Command::new("git").args(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        .current_dir(&root).output() {
        if output.status.success() {
            let mut paths: Vec<_> = output.stdout.split(|b| *b == 0).filter(|b| !b.is_empty()).collect();
            paths.sort(); paths.dedup();
            for name in paths {
                let path = root.join(std::ffi::OsStr::from_bytes(name));
                if let Ok(meta) = std::fs::symlink_metadata(&path) {
                    if meta.is_file() {
                        hash.update(name);
                        if meta.len() <= 1024 * 1024 {
                            if let Ok(bytes) = std::fs::read(path) { hash.update(bytes); }
                        } else {
                            hash.update(meta.len().to_le_bytes());
                            hash.update(format!("{:?}", meta.modified()).as_bytes());
                        }
                    }
                }
            }
        }
    }
    let fingerprint = format!("{:x}", hash.finalize());
    let cache_dir = dirs::data_local_dir().context("No data directory")?.join("luna/projects");
    std::fs::create_dir_all(&cache_dir)?;
    let key = format!("{:x}", Sha256::digest(root.to_string_lossy().as_bytes()));
    let cache = cache_dir.join(format!("{key}.json"));
    let previous = std::fs::read(&cache).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    let changed = previous.as_ref().is_some_and(|v| v["fingerprint"] != fingerprint);
    let report = json!({"root":root,"fingerprint":fingerprint,"context_changed":changed,
        "scope":"Repository documents, not user facts. Re-read source before editing. Changed fingerprint invalidates prior context.",
        "documents":documents,"test_command":test_command(&root),
        "notes": if changed { json!([]) } else { previous.as_ref().map(|v| v["notes"].clone()).unwrap_or(json!([])) }});
    let bytes = serde_json::to_vec_pretty(&report)?;
    let tmp = cache.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(tmp, cache)?;
    Ok(String::from_utf8(bytes)?)
}

pub fn remember(path: &str, note: &str) -> Result<String> {
    if note.trim().is_empty() || note.chars().count() > 500 || note.chars().any(char::is_control) {
        anyhow::bail!("Project note must be a single line of at most 500 characters");
    }
    let mut report: serde_json::Value = serde_json::from_str(&context(path)?)?;
    let mut notes = report["notes"].as_array().cloned().unwrap_or_default();
    let value = json!({"text":note,"source":"agent supplied; verify before relying on it"});
    if !notes.contains(&value) { notes.push(value); }
    if notes.len()>20 { notes.remove(0); }
    report["notes"]=json!(notes);
    let root = root(path)?;
    let key = format!("{:x}", Sha256::digest(root.to_string_lossy().as_bytes()));
    let cache = dirs::data_local_dir().context("No data directory")?.join("luna/projects").join(format!("{key}.json"));
    let tmp = cache.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.sync_all()?;
    std::fs::rename(tmp, cache)?;
    Ok("Project note saved; source changes invalidate it. This is not a verification receipt.".into())
}

fn test_command(root: &Path) -> Option<Vec<String>> {
    if root.join("Cargo.toml").exists() { return Some(vec!["cargo".into(),"test".into()]); }
    if let Ok(raw) = std::fs::read_to_string(root.join("package.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if v["scripts"]["test"].as_str().is_some() {
                return Some(vec!["npm".into(),"test".into()]);
            }
        }
    }
    if root.join("pytest.ini").exists()
        || (root.join("tests").is_dir() && (root.join("pyproject.toml").exists() || root.join("requirements.txt").exists() || root.join("setup.py").exists()))
        || (root.join("pyproject.toml").exists() && root.join("tests").is_dir())
    {
        return Some(vec!["python3".into(),"-m".into(),"pytest".into()]);
    }
    if root.join("go.mod").exists() {
        return Some(vec!["go".into(),"test".into(),"./...".into()]);
    }
    if root.join("Makefile").exists() {
        if let Ok(raw) = std::fs::read_to_string(root.join("Makefile")) {
            if raw.lines().any(|l| l.starts_with("test:") || l.starts_with("check:")) {
                return Some(vec!["make".into(),"test".into()]);
            }
        }
    }
    None
}

pub async fn verify(path: &str) -> Result<String> {
    let root = root(path)?;
    let command = test_command(&root).context("No known test command; inspect project instructions and use run_shell for its documented checks")?;
    // timeout also bounds subprocesses in its process group.
    let output = tokio::process::Command::new("timeout").args(["--kill-after=5","300"])
        .args(&command).current_dir(&root).kill_on_drop(true).output().await?;
    let report = json!({"root":root,"command":command,"passed":output.status.success(),
        "exit_code":output.status.code(),"stdout":crate::util::truncate(&String::from_utf8_lossy(&output.stdout),12000),
        "stderr":crate::util::truncate(&String::from_utf8_lossy(&output.stderr),6000),
        "scope":"Test command only; review git diff separately"});
    if !output.status.success() { anyhow::bail!("Project verification failed: {report}"); }
    Ok(report.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn command_selection_requires_evidence() {
        let d=std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir(&d).unwrap();
        assert!(test_command(&d).is_none());
        std::fs::write(d.join("package.json"), "{}").unwrap();
        assert!(test_command(&d).is_none());
        std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
        assert_eq!(test_command(&d).unwrap(), ["cargo","test"]);
        assert!(std::process::Command::new("git").args(["init","-q"]).arg(&d).status().unwrap().success());
        remember(d.to_str().unwrap(), "Use cargo test for this repository").unwrap();
        let before: serde_json::Value = serde_json::from_str(&context(d.to_str().unwrap()).unwrap()).unwrap();
        assert_eq!(before["notes"].as_array().unwrap().len(),1);
        std::fs::write(d.join("source.rs"), "changed").unwrap();
        let after: serde_json::Value = serde_json::from_str(&context(d.to_str().unwrap()).unwrap()).unwrap();
        assert_eq!(after["context_changed"], true);
        assert_eq!(after["notes"].as_array().unwrap().len(),0);
        let key = format!("{:x}", Sha256::digest(d.canonicalize().unwrap().to_string_lossy().as_bytes()));
        let _ = std::fs::remove_file(dirs::data_local_dir().unwrap().join("luna/projects").join(format!("{key}.json")));
        std::fs::remove_dir_all(d).unwrap();
    }
}
