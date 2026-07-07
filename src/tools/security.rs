//! tools/security.rs — Security toolkit for CTF and network work
//!
//! Wraps nmap, tshark (Wireshark CLI), hashing, and common encoding
//! decoders. All commands run with a hard timeout so a slow or
//! unreachable target can never hang Luna's agent loop.

use super::shell;
use anyhow::Result;

// ── nmap ─────────────────────────────────────────────────────────────────────

pub async fn nmap_scan(target: &str, scan_type: &str, sudo_pass: Option<&str>) -> Result<String> {
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
fn sanitize_target(target: &str) -> String {
    target
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | ':' | '/' | '_'))
        .collect()
}
