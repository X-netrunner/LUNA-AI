//! tools/backup.rs — Report on and trigger the home-directory backup
//!
//! Delegates to `safety_check.sh backup` for the actual rsync work so
//! the incremental logic, lockfile, unmount safety and sudo handling
//! are all handled once in the shell script.

use anyhow::Result;

const MNT: &str = "/mnt/backup";
const DEV: &str = "/dev/sda1";
const UUID: &str = "28df6eaf-a08b-41f4-bc89-2478bb619f8d";

/// Short, model-friendly status of the backup drive / last snapshot.
pub async fn status() -> Result<String> {
    let mut out = String::new();

    let dev_exists = std::path::Path::new(DEV).exists();
    let mounted = mountpoint(MNT).await;

    if !dev_exists {
        out.push_str(&format!("Drive {} not connected (plug it in to back up).", DEV));
        return Ok(out);
    }
    if mounted {
        out.push_str(&format!("Drive {} mounted at {}.\n", DEV, MNT));
    } else {
        out.push_str(&format!("Drive {} present but not mounted at {}.\n", DEV, MNT));
    }

    let blkid = crate::tools::shell::run_command(
        &format!("lsblk -n -o UUID {}", DEV),
        None,
    ).await;
    if let Ok(o) = blkid {
        let id = o.stdout.trim();
        if !id.is_empty() {
            out.push_str(&format!("UUID: {}{}\n", id,
                if id == UUID { " ✓" } else { " (unexpected)" }));
        }
    }

    if mounted {
        if let Some(latest) = latest_snapshot().await {
            out.push_str(&format!("Last snapshot: {}\n", latest));
        }
        if let Some(size) = dir_size(MNT).await {
            out.push_str(&format!("Total usage:  {}\n", size));
        }
        if let Some(free) = free_space(MNT).await {
            out.push_str(&format!("Free space:   {}\n", free));
        }
    } else {
        out.push_str("Mount the drive to see snapshot info.\n");
    }
    Ok(out)
}

/// Trigger a backup via the installed safety_check script (backup mode).
pub async fn run(sudo_pass: Option<&str>) -> Result<String> {
    crate::tools::safety::run("backup", sudo_pass, false).await
}

async fn mountpoint(path: &str) -> bool {
    crate::tools::shell::run_command(
        &format!("mountpoint -q {} && echo YES", path),
        None,
    ).await
        .map(|o| o.stdout.trim().eq_ignore_ascii_case("YES"))
        .unwrap_or(false)
}

async fn latest_snapshot() -> Option<String> {
    let root = format!("{}/arch-backup", MNT);
    let out = crate::tools::shell::run_command(
        &format!("ls -1dt {}/ 2>/dev/null | head -1", root),
        None,
    ).await.ok()?;
    let name = out.stdout.trim();
    if name.is_empty() { return None; }
    // date_YYYY-MM-DD_HHMM / latest symlink — resolve to base name
    let base = std::path::Path::new(name)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| name.to_string());
    // If latest symlink, point to actual dir name
    let dir = if base == "latest" {
        crate::tools::shell::run_command(
            &format!("basename $(readlink -f {}/latest)", root),
            None,
        ).await.ok()?.stdout.trim().to_string()
    } else {
        base
    };
    Some(dir)
}

async fn dir_size(path: &str) -> Option<String> {
    let out = crate::tools::shell::run_command(
        &format!("du -sh {} 2>/dev/null | cut -f1", path),
        None,
    ).await.ok()?;
    let s = out.stdout.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

async fn free_space(path: &str) -> Option<String> {
    let out = crate::tools::shell::run_command(
        &format!("df -h {} | tail -1 | awk '{{print $4}}'", path),
        None,
    ).await.ok()?;
    let s = out.stdout.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}