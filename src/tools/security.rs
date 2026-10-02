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

/// Strip shell-dangerous characters from a target string (host/IP/domain).
/// Not a full validator — just removes the characters that matter for
/// command injection in this specific quoting context.
/// Is this target the local machine, reachable only over loopback?
///
/// Always allowed, on any allowlist, because the traffic cannot leave the
/// host. This is what lets the default configuration work with no setup.
pub fn is_loopback_target(target: &str) -> bool {
    let t = sanitize_target(target).to_lowercase();
    if t == "localhost" || t == "::1" || t == "[::1]" || t == "0:0:0:0:0:0:0:1" {
        return true;
    }
    // Parsed as four octets, not string-matched on a "127." prefix.
    //
    // The prefix version accepted `127.0.0.1.evil.test`, which is a hostname
    // its holder controls, not an address. Anything that does not parse as a
    // complete IPv4 address is not loopback, full stop.
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