//! tools/security.rs — Security toolkit for CTF and network work
//!
//! Wraps nmap, tshark (Wireshark CLI), hashing, and common encoding
//! decoders. All commands run with a hard timeout so a slow or
//! unreachable target can never hang Luna's agent loop.

use super::shell;
use anyhow::Result;

// ── nmap ─────────────────────────────────────────────────────────────────────

pub async fn nmap_scan(
    target: &str,
    scan_type: &str,
    sudo_pass: Option<&str>,
    allowlist: &[String],
) -> Result<String> {
    // Before the sanitiser, so the check sees what will actually be scanned
    // and cannot be evaded by dressing the host up in shell metacharacters.
    ensure_scan_target_allowed(target, allowlist)?;
    let safe_target = sanitize_target(target);

    let nmap_args = match scan_type {
        "quick" => "-T4 -F",
        "full" => "-T4 -sV -sC",
        "ports" => "-T4 -p-",
        "os" => "-T4 -O",
        "udp" => "-T4 -sU --top-ports 20",
        _ => "-T4 -F",
    };

    // Hard cap of 3 minutes regardless of scan type — protects the agent
    // loop from hanging on an unreachable or slow target.
    let cmd = format!(
        "timeout 180 nmap {} --host-timeout 60s '{}' 2>&1",
        nmap_args, safe_target
    );

    let result = shell::run_command(&cmd, sudo_pass).await?;
    if result.stdout.trim().is_empty() {
        Ok(format!(
            "nmap produced no output. stderr: {}",
            result.stderr.trim()
        ))
    } else {
        Ok(result.stdout.trim().to_string())
    }
}

// ── pcap analysis via tshark ───────────────────────────────────────────────────

pub async fn analyze_pcap(path: &str, mode: &str, sudo_pass: Option<&str>) -> Result<String> {
    let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
    let safe_path = expanded.replace('\'', "'\\''");

    if !std::path::Path::new(&expanded).exists() {
        anyhow::bail!("File not found: {}", expanded);
    }

    let cmd = match mode {
        "summary"   => format!("timeout 60 capinfos '{}' 2>&1", safe_path),
        "talkers"   => format!("timeout 60 tshark -q -z conv,ip -r '{}' 2>&1 | head -50", safe_path),
        "protocols" => format!("timeout 60 tshark -q -z io,phs -r '{}' 2>&1 | head -60", safe_path),
        "http"      => format!(
            "timeout 60 tshark -r '{}' -Y http.request -T fields -e ip.src -e http.host -e http.request.method -e http.request.uri 2>&1 | head -50",
            safe_path
        ),
        "dns"       => format!(
            "timeout 60 tshark -r '{}' -Y dns.flags.response==0 -T fields -e ip.src -e dns.qry.name 2>&1 | sort -u | head -50",
            safe_path
        ),
        "creds"     => format!(
            "timeout 60 tshark -r '{}' -Y 'http.request.method==POST || ftp.request.command==\"PASS\" || ftp.request.command==\"USER\"' -T fields -e frame.number -e ip.src -e _ws.col.Protocol 2>&1 | head -30",
            safe_path
        ),
        _ => format!("timeout 60 capinfos '{}' 2>&1", safe_path),
    };

    let result = shell::run_command(&cmd, sudo_pass).await?;
    let out = result.stdout.trim();
    if out.is_empty() {
        Ok(format!("No results. stderr: {}", result.stderr.trim()))
    } else {
        Ok(out.to_string())
    }
}

// ── encoding / decoding ────────────────────────────────────────────────────────

pub async fn decode_payload(data: &str, encoding: &str, sudo_pass: Option<&str>) -> Result<String> {
    let safe_data = data.replace('\'', "'\\''");

    let cmd = match encoding {
        "base64" => format!("printf '%s' '{}' | base64 -d 2>&1", safe_data),
        "hex" => format!("printf '%s' '{}' | xxd -r -p 2>&1", safe_data),
        "url" => format!(
            "python3 -c \"import urllib.parse, sys; print(urllib.parse.unquote('{}'))\"",
            safe_data
        ),
        "rot13" => format!("printf '%s' '{}' | tr 'A-Za-z' 'N-ZA-Mn-za-m'", safe_data),
        "binary" => format!(
            "python3 -c \"print(''.join(chr(int(b,2)) for b in '{}'.split()))\"",
            safe_data
        ),
        "auto" => {
            // Try base64 first since it's the most common in CTF payloads
            format!(
                "printf '%s' '{}' | base64 -d 2>/dev/null \
                 || printf '%s' '{}' | xxd -r -p 2>/dev/null \
                 || echo 'Could not auto-decode — specify encoding explicitly'",
                safe_data, safe_data
            )
        }
        _ => anyhow::bail!("Unknown encoding: {}", encoding),
    };

    let result = shell::run_command(&cmd, sudo_pass).await?;
    let out = if result.stdout.trim().is_empty() {
        &result.stderr
    } else {
        &result.stdout
    };
    Ok(out.trim().to_string())
}

// ── hashing ──────────────────────────────────────────────────────────────────

pub async fn hash_file(path: &str, algo: &str, sudo_pass: Option<&str>) -> Result<String> {
    let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
    let safe_path = expanded.replace('\'', "'\\''");

    if !std::path::Path::new(&expanded).exists() {
        anyhow::bail!("File not found: {}", expanded);
    }

    let cmd = match algo {
        "md5" => format!("md5sum '{}'", safe_path),
        "sha1" => format!("sha1sum '{}'", safe_path),
        "sha256" => format!("sha256sum '{}'", safe_path),
        "sha512" => format!("sha512sum '{}'", safe_path),
        "all" => format!(
            "echo -n 'md5: '; md5sum '{}' | cut -d' ' -f1; \
             echo -n 'sha1: '; sha1sum '{}' | cut -d' ' -f1; \
             echo -n 'sha256: '; sha256sum '{}' | cut -d' ' -f1",
            safe_path, safe_path, safe_path
        ),
        _ => anyhow::bail!("Unknown hash algorithm: {}", algo),
    };

    let result = shell::run_command(&cmd, sudo_pass).await?;
    Ok(result.stdout.trim().to_string())
}

// ── DNS / whois ──────────────────────────────────────────────────────────────

pub async fn dns_lookup(target: &str, mode: &str, sudo_pass: Option<&str>) -> Result<String> {
    let safe_target = sanitize_target(target);

    let cmd = match mode {
        "dns" => format!("timeout 15 dig +short '{}' ANY 2>&1", safe_target),
        "reverse" => format!("timeout 15 dig +short -x '{}' 2>&1", safe_target),
        "whois" => format!("timeout 20 whois '{}' 2>&1 | head -60", safe_target),
        "mx" => format!("timeout 15 dig +short '{}' MX 2>&1", safe_target),
        _ => format!("timeout 15 dig +short '{}' 2>&1", safe_target),
    };

    let result = shell::run_command(&cmd, sudo_pass).await?;
    let out = result.stdout.trim();
    if out.is_empty() {
        Ok("No results found".to_string())
    } else {
        Ok(out.to_string())
    }
}

/// Is this target the local machine, reachable only over loopback?
///
/// Always allowed, on any allowlist, because the traffic cannot leave the
/// host. This is what lets the default configuration work with no setup.
pub fn is_loopback_target(target: &str) -> bool {
    let t = sanitize_target(target).to_lowercase();
    if t == "localhost" || t == "::1" || t == "[::1]" || t == "0:0:0:0:0:0:0:1" {
        return true;
    }
    // ── DO NOT "SIMPLIFY" THIS TO A `starts_with("127.")` PREFIX ─────────────
    //
    // The prefix version accepted `127.0.0.1.evil.test` — a hostname its holder
    // controls, which resolves wherever its holder points it. That is the whole
    // class of bug this guards: a naive string check on a network identifier
    // hands the decision to whoever chose the name. `127.example.com` and
    // `localhost.evil.test` are the same attack wearing a different hat.
    //
    // So this parses four octets and requires all four. Anything that is not a
    // complete IPv4 address is not loopback, full stop — a name that merely
    // looks like an address is not one.
    //
    // The test `a_loopback_looking_prefix_is_not_loopback` in
    // `scan_scope_tests` below exists solely to fail if this is reverted to a
    // prefix match. If you change this function, run it, and do not "tidy" it
    // into `starts_with` on the grounds that it is clearer.
    let octets: Vec<&str> = t.split('.').collect();
    octets.len() == 4
        && octets.iter().all(|o| {
            !o.is_empty()
                && o.len() <= 3
                && o.bytes().all(|b| b.is_ascii_digit())
                && o.parse::<u16>().map(|n| n <= 255).unwrap_or(false)
        })
        && octets[0] == "127"
}

/// Refuse a scan of anything not loopback and not explicitly allowed.
///
/// A refusal, not a warning. The whole point of a scope limit is that it
/// holds when the model is persuasive, and "my laptop" plus a confident
/// hostname lookup is exactly the situation where prose should not be
/// enough.
pub fn ensure_scan_target_allowed(target: &str, allowlist: &[String]) -> Result<()> {
    let safe = sanitize_target(target);
    if is_loopback_target(&safe) {
        return Ok(());
    }
    if allowlist
        .iter()
        .any(|a| sanitize_target(a).eq_ignore_ascii_case(&safe))
    {
        tracing::info!("scan target '{}' is on the allowlist", safe);
        return Ok(());
    }
    anyhow::bail!(
        "Refused: '{safe}' is not a local target. nmap_scan is scoped to loopback \
         (localhost, 127.0.0.1, ::1) plus the hosts listed under \
         `scan_allowlist` in luna.toml. Add the target there if you really want \
         it scanned — this is not something a request should be able to talk past."
    )
}

/// Strip shell-dangerous characters from a target string (host/IP/domain).
///
/// Not a validator — it removes the characters that matter for injection in
/// this quoting context and nothing more. It is NOT a scope check, which is why
/// [`is_loopback_target`] and [`ensure_scan_target_allowed`] exist separately
/// and run first.
fn sanitize_target(target: &str) -> String {
    target
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | ':' | '/' | '_'))
        .collect()
}

#[cfg(test)]
mod scan_scope_tests {
    use super::*;

    fn allow(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// The default: no configuration, and the user's own machine is
    /// reachable. A scope limit that needs setup to be useful is a scope
    /// limit nobody keeps.
    #[test]
    fn loopback_works_with_no_configuration_at_all() {
        for t in ["localhost", "127.0.0.1", "::1", "127.0.0.53"] {
            assert!(
                ensure_scan_target_allowed(t, &[]).is_ok(),
                "loopback target {t} should be allowed with an empty allowlist"
            );
        }
    }

    /// The whole point. A default install must not be able to sweep the LAN.
    #[test]
    fn a_remote_target_is_refused_without_allowlisting_it() {
        for t in [
            "192.168.1.1",
            "10.0.0.5",
            "example.com",
            "192.168.0.0/24",
            "honeypot.lan",
        ] {
            let r = ensure_scan_target_allowed(t, &[]);
            assert!(r.is_err(), "{t} should be refused by default, got {r:?}");
        }
    }

    /// Emptying the allowlist must not widen anything. A user who clears the
    /// list is asking for the default, not for everything.
    #[test]
    fn clearing_the_allowlist_does_not_unlock_the_network() {
        assert!(ensure_scan_target_allowed("192.168.1.1", &allow(&[])).is_err());
    }

    /// Allowlisting is the escape hatch, and it has to actually work or the
    /// limit is just an obstacle.
    #[test]
    fn an_explicitly_allowed_host_is_reachable() {
        let list = allow(&["192.168.1.50", "example.com"]);
        assert!(ensure_scan_target_allowed("192.168.1.50", &list).is_ok());
        assert!(ensure_scan_target_allowed("example.com", &list).is_ok());
        // And still not a neighbour.
        assert!(ensure_scan_target_allowed("192.168.1.51", &list).is_err());
    }

    /// Shell metacharacters must not be usable to disguise a target. The check
    /// runs before the sanitiser precisely so this holds, but a test is what
    /// keeps that ordering from being "tidied" into the wrong order later.
    #[test]
    fn a_disguised_target_cannot_slip_past_the_check() {
        for t in [
            "192.168.1.1; whoami",
            "$(echo 192.168.1.1)",
            "192.168.1.1`id`",
            "192.168.1.1 && curl evil.test",
        ] {
            assert!(
                ensure_scan_target_allowed(t, &[]).is_err(),
                "disguised target {t} was allowed"
            );
        }
    }

    /// `0.0.0.0` is not loopback. On Linux it means "every interface", so a
    /// scan against it reaches the LAN — which is the exact case the limit
    /// exists to stop, wearing a friendly name.
    #[test]
    fn the_wildcard_address_is_not_treated_as_local() {
        assert!(!is_loopback_target("0.0.0.0"));
        assert!(ensure_scan_target_allowed("0.0.0.0", &[]).is_err());
    }

    /// A prefix must not be enough: `127.example.com` resolves somewhere real,
    /// and `127.0.0.1.evil.test` is a real attack.
    #[test]
    fn a_loopback_looking_prefix_is_not_loopback() {
        for t in ["127.0.0.1.evil.test", "127.example.com", "localhost.evil.test"] {
            assert!(!is_loopback_target(t), "{t} should not count as loopback");
        }
    }
}
// ── Tool availability ────────────────────────────────────────────────────────
//
// Measured 2026-10-02, live session. Asked to exploit SSH she wrote:
//
//     hydra -L /usr/share/wordlists/rockyou.txt -P /usr/share/wordlists/rockyou.txt localhost ssh
//
// Neither the tool nor the wordlist exists on this host. `hydra` is not
// installed, and `/usr/share/wordlists/rockyou.txt` is not a file. Both were
// asserted as fact, inside a turn where the surrounding narration was invented
// too, so there was no tool result anywhere to contradict them.
//
// That is a different failure from the transcript one and it needs a different
// fix. The fabrication guard catches output that no tool produced; it cannot
// catch a claim about the *machine* that happens to be wrong, because there is
// no receipt for the machine's state. So the capability has to exist rather
// than the correction.
//
// What is genuinely available here, measured: 71 nmap `*-brute.nse` scripts
// including `ssh-brute`, `ftp-anon`, `smtp-user-enum` and `mysql-brute`;
// `~/CyberSecurity/wordlist/seclists`; `hashcat`. `pacman -Ss '^hydra$'`
// resolves `extra/hydra 9.7-1`. So the honest answer to "hydra is missing" is
// usually "you already have `nmap --script ssh-brute` and seclists", and only
// sometimes "install hydra". Checking has to come first for that to be true.

/// How to ask the host's package manager, detected rather than assumed.
struct PkgManager {
    name: &'static str,
    search: &'static str,
    owns: &'static str,
    installed: &'static str,
    install: &'static str,
}

/// Detected by probing, in preference order.
///
/// The user asked for "pacman -Q or something like that depending on the OS",
/// and hardcoding pacman would have been a bug waiting for the next machine.
fn detect_pkg_manager() -> Option<PkgManager> {
    let has = |c: &str| {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("command -v {c}"))
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    if has("pacman") {
        Some(PkgManager {
            name: "pacman",
            search: "pacman -Ss",
            owns: "pacman -Qo",
            installed: "pacman -Qi",
            install: "pacman -S --needed --noconfirm",
        })
    } else if has("apt-get") {
        Some(PkgManager {
            name: "apt-get",
            search: "apt-cache search",
            owns: "apt-file search",
            installed: "dpkg -s",
            install: "apt-get install -y",
        })
    } else if has("dnf") {
        Some(PkgManager {
            name: "dnf",
            search: "dnf list",
            owns: "rpm -qf",
            installed: "rpm -q",
            install: "dnf install -y",
        })
    } else if has("apk") {
        Some(PkgManager {
            name: "apk",
            search: "apk search",
            owns: "apk info -W",
            installed: "apk info -e",
            install: "apk add",
        })
    } else {
        None
    }
}

/// Strictly a command name: no spaces, no shell metacharacters.
///
/// This is the only thing standing between the model's output and `bash -c`,
/// so it is a whitelist of characters rather than a blacklist. A blacklist
/// invites the next metacharacter nobody thought of; a whitelist makes the
/// unsafe set empty by construction. `run_command` runs `bash -c`, so this
/// matters here in a way it would not for a direct exec.
fn plain_command_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        && !name.starts_with('-')
}

/// What to suggest when a tool is missing, keyed by what it was for.
///
/// Not a general "similar tools" search — those are guesses, and a guess here
/// would be the same fabrication one level up. Every entry is a substring
/// matched against scripts that are actually present on disk, and nothing is
/// listed unless the file exists.
const CAPABILITY_HINTS: &[(&str, &[&str])] = &[
    ("hydra", &["brute"]),
    ("ncrack", &["brute"]),
    ("medusa", &["brute"]),
    ("crack", &["brute"]),
    ("brute", &["brute"]),
    ("password", &["brute"]),
    ("john", &["hash"]),
    ("hashcat", &["hash"]),
    ("nikto", &["vuln", "http-vuln"]),
    ("sqlmap", &["sql", "mysql"]),
    ("gobuster", &["http-enum", "dir"]),
    ("ffuf", &["http-enum", "dir"]),
    ("dirbuster", &["http-enum", "dir"]),
    ("telnet", &["telnet"]),
    ("ftp", &["ftp"]),
    ("ssh", &["ssh"]),
    ("smtp", &["smtp", "pop3"]),
    ("mysql", &["mysql"]),
    ("redis", &["redis"]),
    ("mssql", &["mssql"]),
    ("http", &["http"]),
    ("dns", &["dns"]),
    ("smb", &["smb"]),
    ("rdp", &["rdp"]),
    ("vnc", &["vnc"]),
    ("nmap", &["vuln"]),
];

/// Order alternatives by how likely they are to be the one wanted.
///
/// Sorted alphabetically this is worse than useless: asked about hydra, the
/// list came back `afp-`, `ajp-`, `cics-`, `citrix-`, `cvs-`, `dicom-` and the
/// twelve-item cap cut `ssh-brute` off entirely, which is the single most
/// relevant script on the box for the job. Caught by a test asserting on the
/// real host rather than on a fixture, which is the only reason it was caught
/// before shipping.
///
/// Common services first, then everything else alphabetically.
const PREFERRED_SERVICES: &[&str] = &[
    "ssh", "ftp", "telnet", "smtp", "pop3", "http", "mysql", "redis", "smb", "dns", "mssql",
    "vnc", "rdp", "imap", "ldap", "rpc", "ssh2",
];

fn script_priority(fname: &str) -> usize {
    PREFERRED_SERVICES
        .iter()
        .position(|s| fname.contains(s))
        .unwrap_or(PREFERRED_SERVICES.len())
}

/// nmap scripts already on this machine that cover the same job.
fn installed_scripts(hints: &[&str]) -> Vec<String> {
    let mut dirs: Vec<std::path::PathBuf> = vec![
        "/usr/share/nmap/scripts".into(),
        "/usr/local/share/nmap/scripts".into(),
    ];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(std::path::PathBuf::from(home).join(".local/share/nmap/scripts"));
    }
    let mut found: Vec<String> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".nse") {
                continue;
            }
            if hints.iter().any(|h| fname.contains(h)) && !found.contains(&fname) {
                found.push(fname);
            }
        }
    }
    found.sort_by_key(|f| (script_priority(f), f.clone()));
    found
}

/// Wordlist directories that exist here, with a file count.
///
/// Reported as a count rather than a listing because seclists holds tens of
/// thousands of files and the point is "yes, you have some", not "here are
/// forty thousand".
fn wordlist_dirs() -> Vec<(String, usize)> {
    let mut candidates: Vec<std::path::PathBuf> = vec![
        "/usr/share/wordlists".into(),
        "/usr/share/seclists".into(),
    ];
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(std::path::PathBuf::from(&home).join("CyberSecurity/wordlist"));
        candidates.push(std::path::PathBuf::from(&home).join("CyberSecurity/wordlist/seclists"));
        candidates.push(std::path::PathBuf::from(&home).join("wordlists"));
    }
    let mut out: Vec<(String, usize)> = Vec::new();
    for dir in candidates {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let count = entries.flatten().count();
        if count > 0 && !out.iter().any(|(p, _)| *p == dir.display().to_string()) {
            out.push((dir.display().to_string(), count));
        }
    }
    out
}

/// Whether the tool is actually on this machine.
///
/// Answered with a bounded `--version`, not assumed from the path: a file
/// called `hydra` in `/usr/local/bin` that is not hydra is precisely the kind
/// of thing that should not be reported as "available".
async fn probe_installed(name: &str) -> Option<String> {
    let which = shell::run_command(&format!("command -v {name}"), None).await.ok()?;
    let path = which.stdout.trim().to_string();
    if path.is_empty() {
        return None;
    }
    let mut report = path.clone();
    // Bounded: some `--version` paths are interactive, and this runs inside the
    // agent loop where a hang costs a whole turn.
    if let Ok(v) = shell::run_command(&format!("timeout 10 {name} --version 2>&1 | head -2"), None).await
    {
        let first = v.stdout.lines().next().unwrap_or("").trim().to_string();
        if !first.is_empty() {
            report.push_str(&format!("\n  version: {first}"));
        }
    }
    Some(report)
}

/// Whether the tool is genuinely installed AND what else on this machine does
/// the same job.
///
/// `pkg_install` is named in the output on purpose. The tool's job is to stop
/// her asserting a binary that does not exist, and the useful answer to "hydra
/// is missing" is usually not "install hydra" — on this host it is
/// `nmap --script ssh-brute`, which is already installed. Checking has to come
/// before installing or the alternative is never found.
pub async fn tool_check(name: &str) -> Result<String> {
    let name = name.trim();
    if !plain_command_name(name) {
        anyhow::bail!(
            "'{}' is not a plain command name. Give a tool name like 'hydra' or 'nmap', not a \
             path or a command line.",
            name.chars().take(40).collect::<String>()
        );
    }

    let key = name.to_ascii_lowercase();
    let hints: Vec<&str> = CAPABILITY_HINTS
        .iter()
        .find(|(k, _)| key.contains(k))
        .map(|(_, h)| h.to_vec())
        .unwrap_or_else(|| vec![key.as_str()]);

    let mut out = String::new();
    match probe_installed(name).await {
        Some(info) => {
            out.push_str(&format!("INSTALLED: {name}\n  {info}\n"));
        }
        None => {
            out.push_str(&format!("NOT INSTALLED: {name}\n"));

            if let Some(pm) = detect_pkg_manager() {
                // Anchored so a search for "hydra" does not return every
                // package with "hydra" somewhere in its description.
                let out_cmd = format!(
                    "timeout 25 {} ^{}$ 2>&1 | head -6",
                    pm.search,
                    name.to_ascii_lowercase()
                );
                if let Ok(r) = shell::run_command(&out_cmd, None).await {
                    let hits: Vec<&str> = r.stdout.lines().filter(|l| !l.trim().is_empty()).collect();
                    if hits.is_empty() {
                        out.push_str(&format!(
                            "  no package named '{name}' in the repos ({})\n",
                            pm.name
                        ));
                    } else {
                        out.push_str(&format!("  available from repos ({}):\n", pm.name));
                        for h in &hits {
                            out.push_str(&format!("    {}\n", h.trim()));
                        }
                    }
                }
                // `-Qo` answers "which package owns this file", which is how a
                // tool ends up installed but off PATH.
                if let Ok(r) = shell::run_command(&format!("timeout 15 {} {name} 2>&1", pm.owns), None).await
                {
                    let owned = r.stdout.trim();
                    if !owned.is_empty() && !owned.contains("not owned") && !owned.contains("error:") {
                        out.push_str(&format!("  owned by: {owned}\n"));
                    }
                }
            } else {
                out.push_str("  no supported package manager found (pacman, apt-get, dnf, apk)\n");
            }
        }
    }

    let scripts = installed_scripts(&hints);
    if !scripts.is_empty() {
        out.push_str(&format!(
            "\nALREADY ON THIS MACHINE — nmap scripts for the same job ({} found):\n",
            scripts.len()
        ));
        for s in scripts.iter().take(12) {
            out.push_str(&format!("    nmap --script {s}\n"));
        }
        if scripts.len() > 12 {
            out.push_str(&format!("    … and {} more\n", scripts.len() - 12));
        }
    }

    let lists = wordlist_dirs();
    if !lists.is_empty() {
        out.push_str("\nWORDLISTS PRESENT:\n");
        for (path, count) in &lists {
            out.push_str(&format!("    {path}  ({count} entries)\n"));
        }
    } else {
        out.push_str("\nWORDLISTS: none found in the usual locations.\n");
    }

    out.push_str(
        "\nUse these facts, do not assume a tool or a path exists. If nothing above covers \
         the task, call pkg_install to install it — do not describe an install as done \
         unless pkg_install returned success.\n",
    );
    Ok(out)
}

/// Packages `pkg_install` will install without asking.
///
/// A curated list, not "any package the repos have", and the reason is not
/// caution for its own sake: `pacman -S` runs maintainer install scripts as
/// root, so an open-ended installer is arbitrary code execution chosen by a 7B
/// model on the strength of a sentence. The list below covers the tools these
/// turns actually reach for. Widen it in config rather than by editing here.
///
/// The cost of getting an entry wrong is not a failed install. `pkg_install`
/// would report `target not found`, and she would conclude that `pkg_install`
/// is broken — teaching the wrong lesson about a tool that works. Which is why
/// every entry is checked against `pacman` in both directions by the tests in
/// `tool_availability_tests`, and why the names that *cannot* be resolved are
/// written down below rather than left as folklore.
pub const PKG_INSTALL_ALLOWLIST: &[&str] = &[
    // Credential testing and password recovery.
    "hydra", "ncrack", "medusa", "john", "hashcat",
    // Recon. `masscan` complements the 71 nmap scripts rather than replacing
    // them. Two of these are named after their *purpose*, not their binary:
    // `wireshark-cli` provides `tshark` (`pacman -S tshark` fails) and
    // `exploitdb` provides `searchsploit` (`pacman -S searchsploit` fails).
    // Both are worth having: exploitdb in particular lets her look up a known
    // CVE by the exact service version nmap returned, offline, instead of
    // inventing one.
    "nmap", "masscan", "wireshark-cli", "exploitdb", "tcpdump",
    // Service probing and web. `nikto`, `sqlmap`, `gobuster` are the general
    // ones; `wpscan` is WordPress-specific but is in `extra`, not the AUR as
    // commonly claimed, and it is the single most useful thing to have when a
    // honeypot serves a CMS.
    "nikto", "sqlmap", "gobuster", "wpscan",
    // Transport and shell.
    "socat", "openbsd-netcat",
    // Credential spraying and enumeration. `smbclient` and `openldap` are the
    // official-repo clients; `enum4linux` and `dirb` are not, see below.
    "smbclient", "openldap",
    // Firmware/embedded analysis, for the IoT-shaped targets.
    "binwalk",
    // Wireless, for the WiFi-adjacent questions that keep coming up.
    // `hcxtools` provides `hcxdumptool` for modern WPA2 handshake capture,
    // which then feeds `hashcat` above.
    "aircrack-ng", "hcxtools",
];

/// Names that `pacman -S` cannot resolve, kept so the mistake is recorded
/// rather than repeated.
///
/// Two kinds live here and they fail differently, which is why they are worth
/// distinguishing in prose even though the test treats them alike:
///
/// * **AUR-only** — the package exists, just not in the official repos:
///   `ffuf`, `whatweb`, `enum4linux`, `dirb`, `wfuzz`, `responder`,
///   `netdiscover`, `crackmapexec`, `netexec`, `metasploit-framework`, `cewl`,
///   `burpsuite`. An AUR helper (`yay`/`paru`) would reach them, which is a
///   deliberate non-goal: it builds an arbitrary PKGBUILD and `pkg_install`
///   runs as root.
/// * **No such package on Arch at all** — `crunch`. There is no wordlist
///   generator by that name. Do *not* substitute `crunch64`, which does exist
///   and is "a library for handling common N64 compression formats". This is
///   the closest call in the file: the name is one character away and the
///   package is real, so it passes every syntactic check and installs cleanly
///   into the exact wrong tool.
/// * **Binary names, not packages** — `tshark` and `searchsploit`. Both tools
///   are reachable, as `wireshark-cli` and `exploitdb`, which are on the
///   allowlist above. Listed so nobody "fixes" the allowlist by adding the
///   binary name.
///
/// How this list got here: the first allowlist shipped four AUR-only packages
/// and one nonexistent name (`netcat-openbsd`, where the package is
/// `openbsd-netcat`) and nothing caught it, because the only test on the list
/// asserted that the allowlist and the validator agreed with *each other* —
/// two things I wrote, checking each other. Every entry in both lists is now
/// checked against `pacman` in both directions by the two tests at the bottom
/// of `tool_availability_tests`.
///
/// Test-only, and that is the honest description of it. Every entry is a claim
/// about the world — that `pacman` cannot resolve it — and claims about the
/// world decay as the repos move. So the list is data for the tests that check
/// it against the real package database, not production logic: nothing at
/// runtime reads it. Left unannotated it would sit in the shipping build
/// looking load-bearing while being unreachable, which is how a stale entry
/// survives for years.
#[cfg(test)]
const NOT_IN_OFFICIAL_REPOS: &[&str] = &[
    "ffuf", "whatweb", "enum4linux", "dirb", "wfuzz", "responder", "netdiscover",
    "metasploit-framework", "crackmapexec", "netexec", "cewl", "burpsuite",
    "crunch", "tshark", "searchsploit",
];

/// Install one allowlisted package, as root, non-interactively.
///
/// # Why this may use sudo when `exec` refuses it outright
///
/// `exec` was refused because it pipes the sudo password into *model-authored*
/// text — she writes a shell command and the password goes into it. Here the
/// command is entirely harness-authored: `pacman -S --needed --noconfirm` plus a
/// package name that has passed `plain_command_name` (so no shell metacharacter
/// can survive) and `PKG_INSTALL_ALLOWLIST` membership. The model supplies a
/// word, not a command. Those are different risks and they get different
/// answers.
///
/// Bounded to four things, all deliberate: allowlisted, name-validated, log-only
/// refusals for everything else, and no install at all if the package is already
/// present.
pub async fn pkg_install(
    package: &str,
    sudo_pass: Option<&str>,
    extra_allowed: &[String],
) -> Result<String> {
    let package = package.trim().to_ascii_lowercase();

    if !plain_command_name(&package) {
        anyhow::bail!(
            "'{}' is not a valid package name. Only letters, digits, '-', '_' and '.' are \
             accepted.",
            package.chars().take(40).collect::<String>()
        );
    }

    let allowed = PKG_INSTALL_ALLOWLIST.contains(&package.as_str())
        || extra_allowed.iter().any(|p| p.eq_ignore_ascii_case(&package));
    if !allowed {
        anyhow::bail!(
            "'{package}' is not on the install allowlist, so it was not installed. Allowed: {}.\n\
             If you need something else, check first with tool_check — the capability is often \
             already present via nmap scripts — and tell the user which package to install \
             themselves.",
            PKG_INSTALL_ALLOWLIST.join(", ")
        );
    }

    let Some(pm) = detect_pkg_manager() else {
        anyhow::bail!("no supported package manager found (pacman, apt-get, dnf, apk)");
    };

    // Already installed: say so and change nothing. Reinstalling is a no-op on
    // Arch but not on every manager, and "it is already there" is the useful
    // answer when the real problem is that she assumed it was missing.
    if let Ok(r) = shell::run_command(&format!("timeout 15 {} {package} 2>&1", pm.installed), None).await
    {
        if r.exit_code == 0 && !r.stdout.trim().is_empty() {
            return Ok(format!(
                "{package} is already installed. Nothing to do.\n{}",
                crate::util::truncate(&r.stdout, 400)
            ));
        }
    }

    if sudo_pass.is_none() {
        anyhow::bail!(
            "installing {package} needs root and no sudo password is configured. Set \
             [agent] sudo_password in the config, or install it yourself with: \
             `{} {} {package}`",
            pm.name,
            pm.install
        );
    }

    let cmd = format!("sudo {} {package}", pm.install);
    // Logged because this changes the machine, and because a persistent change
    // made on a model's judgement should be findable afterwards.
    tracing::warn!("pkg_install: installing '{package}' via {}", pm.name);

    let out = shell::run_command(&cmd, sudo_pass).await?;
    let status = if out.exit_code == 0 { "SUCCESS" } else { "FAILED" };
    Ok(format!(
        "pkg_install {status} (exit {}) for '{package}'\ncommand: {}\n\n{}\n{}",
        out.exit_code,
        cmd,
        out.stdout.trim(),
        out.stderr.trim()
    ))
}

#[cfg(test)]
mod tool_availability_tests {
    use super::*;

    /// The charset is the whole security boundary here — `run_command` hands the
    /// string to `bash -c` — so this is tested as an allowlist, including the
    /// shapes that a blacklist would let through.
    #[test]
    fn only_plain_command_names_are_accepted() {
        for good in ["hydra", "nmap", "netcat-openbsd", "a.b_c-d", "sqlmap"] {
            assert!(plain_command_name(good), "{good} should be allowed");
        }
        for bad in [
            "hydra; rm -rf /",
            "hydra && curl evil.sh",
            "hydra | tee /tmp/x",
            "$(whoami)",
            "`id`",
            "hyd ra",
            "-rf",
            "",
            "hydra\nnmap",
            "hydra'",
            "hydra\"",
            "../hydra",
        ] {
            assert!(!plain_command_name(bad), "{bad:?} must be refused");
        }
    }

    /// Every entry in the allowlist must survive the name validator, or the
    /// allowlist and the validator silently disagree and an allowed package
    /// becomes uninstallable for no visible reason.
    #[test]
    fn every_allowlisted_package_passes_its_own_validator() {
        for pkg in PKG_INSTALL_ALLOWLIST {
            assert!(
                plain_command_name(pkg),
                "'{pkg}' is allowlisted but the validator would refuse it"
            );
            assert!(
                !NOT_IN_OFFICIAL_REPOS.contains(pkg),
                "'{pkg}' is in both the allowlist and NOT_IN_OFFICIAL_REPOS; \
                 pkg_install runs pacman and can never install it"
            );
        }
    }

    /// Every allowlisted package must actually exist in the official repos.
    ///
    /// This asks pacman instead of trusting the list, and that is the entire
    /// point. The first allowlist shipped four AUR-only packages (`ffuf`,
    /// `whatweb`, `enum4linux`, `dirb`) and one name that does not exist
    /// (`netcat-openbsd`; the package is `openbsd-netcat`). Nothing caught it,
    /// because the only test on this list asserted that the allowlist and the
    /// validator agreed with *each other* — two things I wrote, checking each
    /// other. The bug is only visible when something outside the codebase is
    /// asked, which is the same lesson the fence-walker bugs taught today.
    ///
    /// Skips when there is no supported package manager, so a non-Arch checkout
    /// gets a skip rather than a spurious failure. Slow: one `pacman -Ss` per
    /// entry, ~1s each on a warm cache.
    #[test]
    fn every_allowlisted_package_exists_in_the_official_repos() {
        if detect_pkg_manager().is_none() {
            eprintln!("skipping: no supported package manager on this host");
            return;
        }
        let mut missing: Vec<&str> = Vec::new();
        for pkg in PKG_INSTALL_ALLOWLIST {
            let probe = format!(
                "timeout 25 pacman -Ss '^{}$' 2>/dev/null \
                 | grep -Eo '^(extra|core|multilib)/' | head -1",
                pkg.replace('\'', "")
            );
            let found = std::process::Command::new("sh")
                .arg("-c")
                .arg(&probe)
                .output()
                .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
                .unwrap_or(false);
            if !found {
                missing.push(pkg);
            }
        }
        assert!(
            missing.is_empty(),
            "allowlisted but not in the official repos: {missing:?}\n\
             pkg_install runs `pacman -S` and nothing else, so these can never \
             install — she would see `target not found` and conclude the tool \
             is broken. Drop them, or install an AUR helper deliberately and \
             accept that it builds arbitrary PKGBUILDs as root."
        );
    }

    /// `NOT_IN_OFFICIAL_REPOS` is a claim about the world too, so it gets the
    /// same treatment. It is easy to leave a package in that list after it is
    /// packaged upstream, at which point it silently stops being a useful
    /// record of the mistake and starts being a reason to omit a good tool.
    #[test]
    fn the_aur_only_list_is_still_true() {
        if detect_pkg_manager().is_none() {
            eprintln!("skipping: no supported package manager on this host");
            return;
        }
        let mut wrongly_excluded: Vec<&str> = Vec::new();
        for pkg in NOT_IN_OFFICIAL_REPOS {
            let probe = format!(
                "timeout 25 pacman -Ss '^{}$' 2>/dev/null \
                 | grep -Eo '^(extra|core|multilib)/' | head -1",
                pkg.replace('\'', "")
            );
            let present = std::process::Command::new("sh")
                .arg("-c")
                .arg(&probe)
                .output()
                .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
                .unwrap_or(false);
            if present {
                wrongly_excluded.push(pkg);
            }
        }
        assert!(
            wrongly_excluded.is_empty(),
            "listed as AUR-only but present in the official repos: \
             {wrongly_excluded:?} — remove them from NOT_IN_OFFICIAL_REPOS"
        );
    }

    /// Every capability hint must be a plausible substring, and must not be so
    /// broad that it matches everything.
    #[test]
    fn capability_hints_are_not_so_broad_they_match_nothing_meaningful() {
        for (key, hints) in CAPABILITY_HINTS {
            assert!(!key.is_empty() && *key == key.to_ascii_lowercase());
            assert!(!hints.is_empty(), "{key} has no hints");
        }
    }

    /// The real question this module exists for. Measured on this host:
    /// hydra is absent and `/usr/share/wordlists/rockyou.txt` does not exist,
    /// and both were asserted as fact in a live session.
    #[tokio::test]
    async fn tool_check_reports_the_real_absence_of_hydra() {
        let out = tool_check("hydra")
            .await
            .expect("tool_check must not fail on a missing tool");

        assert!(
            out.contains("NOT INSTALLED: hydra"),
            "hydra is absent here and must be reported absent: {out}"
        );
        // The whole point: something that does the same job, already present.
        assert!(
            out.contains("ALREADY ON THIS MACHINE"),
            "no alternatives offered, so the tool is useless: {out}"
        );
        assert!(
            out.contains("ssh-brute"),
            "ssh-brute.nse is installed on this host and is the actual answer: {out}"
        );
    }

    /// A tool that IS here must be reported present, with a version. Reporting
    /// a false absence would push her to install something she already has.
    #[tokio::test]
    async fn tool_check_finds_a_tool_that_is_installed() {
        let out = tool_check("nmap").await.expect("tool_check must succeed");
        assert!(
            out.contains("INSTALLED: nmap"),
            "nmap is installed here: {out}"
        );
    }

    /// The wordlist report must reflect this machine, where seclists exists and
    /// rockyou does not. Reporting a path that is not there is the bug.
    #[tokio::test]
    async fn tool_check_reports_wordlists_that_actually_exist() {
        let out = tool_check("hydra").await.expect("tool_check must succeed");
        if out.contains("WORDLISTS PRESENT") {
            assert!(
                !out.contains("/usr/share/wordlists/rockyou.txt  (0 entries)"),
                "must not invent a wordlist path: {out}"
            );
            for line in out.lines().filter(|l| l.contains("entries)")) {
                let path = line.trim();
                let path = path
                    .trim_start_matches("WORDLISTS PRESENT")
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("");
                assert!(
                    !path.is_empty() && std::path::Path::new(path).is_dir(),
                    "reported wordlist dir does not exist: {path}"
                );
            }
        }
    }

    /// Injection attempts must fail at the validator, before any shell runs.
    #[tokio::test]
    async fn pkg_install_refuses_a_shell_metacharacter_name() {
        let err = pkg_install("hydra; rm -rf /", None, &[])
            .await
            .expect_err("must refuse");
        assert!(
            err.to_string().contains("not a valid package name"),
            "wrong refusal: {err}"
        );
    }

    /// An allowlisted install with no sudo password must fail with the command
    /// to run by hand, not by attempting anything.
    #[tokio::test]
    async fn pkg_install_without_a_password_explains_rather_than_acts() {
        let err = pkg_install("hydra", None, &[])
            .await
            .expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("needs root"), "unhelpful: {msg}");
        assert!(
            msg.contains("pacman -S --needed --noconfirm hydra"),
            "must hand back the exact command: {msg}"
        );
    }

    /// A real refusal case from this session's shape: a package that exists,
    /// works, and is not on the list. The refusal must be actionable rather
    /// than a bare "no".
    ///
    /// `nvim` is the example because it is the one the user actually asked
    /// about, and it is a good specimen: harmless, real, packaged, and
    /// unrelated to anything on this list. Note that this is a fixture that
    /// can rot — it was `masscan` until this session's allowlist revision
    /// added masscan, and the test then failed for the *right* reason. If the
    /// refusal ever stops firing, check whether the example got allowlisted
    /// before assuming `pkg_install` broke.
    #[tokio::test]
    async fn a_package_outside_the_allowlist_is_refused_with_the_list() {
        let err = pkg_install("nvim", Some("x"), &[])
            .await
            .expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("not on the install allowlist"), "{msg}");
        assert!(msg.contains("hydra"), "must list what IS allowed: {msg}");
        assert!(
            msg.contains("tool_check"),
            "must point at the cheaper option: {msg}"
        );
    }

    /// The allowlist can be widened by name in config, and only by name.
    #[tokio::test]
    async fn config_can_widen_the_allowlist() {
        // `masscan` passes the validator, so reaching the "no sudo" branch
        // proves it got past the allowlist rather than being refused earlier.
        let err = pkg_install("masscan", None, &["masscan".to_string()])
            .await
            .expect_err("no sudo password, so it stops before installing");
        assert!(
            err.to_string().contains("needs root"),
            "allowlist did not widen: {err}"
        );
    }
}

#[cfg(test)]
mod show_output {
    /// Not a test — a way to see what Luna actually receives. Run with
    /// `cargo test show_output -- --ignored --nocapture`. Kept because the
    /// ordering of the alternatives list is a judgement call that is worth
    /// being able to look at without editing a test.
    #[tokio::test]
    #[ignore]
    async fn show() {
        println!("{}", super::tool_check("hydra").await.unwrap());
    }
}

/// What the allowlist does and does not buy, stated as a test.
///
/// Added because "Luna cannot install unlisted packages" is the natural thing
/// to believe after reading `PKG_INSTALL_ALLOWLIST`, and it is not true.
/// `run_shell` is in neither `gated_tools` nor `capability_tools`, it forwards
/// `sudo_pass`, and it runs `bash -c` on a model-authored string — so
/// `sudo pacman -S nvim` already works today, allowlist or not.
///
/// The allowlist is a speed bump on the *honest* path: it stops the common case,
/// where she reaches for `pkg_install` because it is right there and
/// documented, and it makes that refusal legible instead of silent. It is not a
/// sandbox and nothing should imply it is.
///
/// The alternative is not a stricter list. A list strong enough to hold against
/// `run_shell` would have to forbid installing anything at all, at which point
/// `pkg_install` is pointless. The real question is what `run_shell` is for on
/// an offensive turn, which is a design decision rather than a patch. Until that
/// is answered this test exists so the allowlist does not quietly acquire a
/// reputation it has not earned.
#[cfg(test)]
mod allowlist_scope_tests {
    use super::*;

    #[tokio::test]
    async fn the_allowlist_refuses_a_package_outside_it() {
        let err = pkg_install("nvim", Some("x"), &[])
            .await
            .expect_err("nvim is not on the list and must be refused");
        assert!(
            err.to_string().contains("not on the install allowlist"),
            "unexpected refusal: {err}"
        );
    }

    /// The premise of the note above, pinned. If `run_shell` ever becomes
    /// gated this test fails, and the note has to be rewritten rather than left
    /// claiming a boundary that has moved.
    #[test]
    fn run_shell_is_ungated_so_the_allowlist_is_not_a_boundary() {
        let cfg = crate::config::LunaConfig::default();
        assert!(
            !cfg.external.is_gated("run_shell"),
            "run_shell is gated: update the note above, the allowlist may now be \
             a real boundary rather than a speed bump"
        );
        assert!(
            !cfg.external.is_capability_gated("run_shell"),
            "run_shell is capability-gated: update the note above"
        );
    }
}
