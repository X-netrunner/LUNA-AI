//! tools/spotify.rs — Spotify Web API integration
//!
//! Controls the user's Spotify via the Web API: play Liked Songs, playlists,
//! artists, albums, search, and transport control. Auth is the Authorization
//! Code flow with PKCE: `luna --spotify-auth` opens the browser, the user
//! approves, Spotify redirects to 127.0.0.1:8888/callback which this process
//! catches, and the long-lived refresh token is stored in the OS keyring.
//!
//! Requires:
//! - Spotify Premium (playback endpoints are 403 on free accounts)
//! - A Spotify app created at https://developer.spotify.com/dashboard
//!   (`luna --set-key spotify_id`; no client secret is needed with PKCE)
//! - A Spotify client playing on some device on the same account
//! - The app's Redirect URI set to http://127.0.0.1:8888/callback
//!
//! When playback control is unavailable (no device / revoked token), the
//! caller handles the error and may fall back to MPRIS.

use crate::config::LunaConfig;
use anyhow::{anyhow, Context, Result};
use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const API_URL: &str = "https://api.spotify.com/v1";
const TOKEN_URL: &str = "https://accounts.spotify.com/api/token";
const AUTH_URL: &str = "https://accounts.spotify.com/authorize";
/// Local loopback callback registered on the Spotify app. Must match the
/// "Redirect URIs" entry EXACTLY (scheme/host/port/path).
const REDIRECT_URI: &str = "http://127.0.0.1:8888/callback";
/// Scopes requested at authorization. `user-modify-playback-state` is what
/// allows Luna to actually play/pause on the user's devices.
const OAUTH_SCOPE: &str = "user-read-private user-read-email \
                           user-read-playback-state user-modify-playback-state \
                           user-read-currently-playing \
                           playlist-read-private playlist-read-collaborative \
                           user-library-read";
/// Keyring entry holding the long-lived OAuth refresh token.
const REFRESH_KEYRING: &str = "spotify_refresh";

// ── In-memory access-token cache ──────────────────────────────────────────────
static TOKEN_CACHE: OnceLock<Mutex<Option<(String, Instant)>>> = OnceLock::new();

fn token_cache() -> &'static Mutex<Option<(String, Instant)>> {
    TOKEN_CACHE.get_or_init(|| Mutex::new(None))
}

// ── OAuth (Authorization Code + PKCE) ─────────────────────────────────────────
//
// Spotify does NOT allow the OAuth Device Flow for apps created through the
// developer dashboard ("unauthorized_client — Client not allowed"). Desktop
// apps therefore use the Authorization Code flow with PKCE: the browser opens
// Spotify's authorize page, the user approves, Spotify redirects the browser to
// our loopback URI (127.0.0.1:8888/callback) which this process momentarily
// listens on and exchanges the code for tokens. No client secret is involved.

#[derive(Debug, Deserialize)]
struct TokenResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    refresh_token: String,
    #[serde(default)]
    expires_in: u64,
    #[serde(default)]
    #[serde(rename = "error")]
    error: Option<String>,
    #[serde(default)]
    #[serde(rename = "error_description")]
    error_description: Option<String>,
}

fn urandom_bytes(n: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom").context("open /dev/urandom")?;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).context("read /dev/urandom")?;
    Ok(buf)
}

/// (code_verifier, code_challenge) for PKCE S256. Verifier is 64 random bytes
/// base64url-encoded (no padding) — comfortably within the 43–128 char range.
fn pkce_pair() -> Result<(String, String)> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use sha2::{Digest, Sha256};
    let raw = urandom_bytes(64)?;
    let verifier = URL_SAFE_NO_PAD.encode(&raw);
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(digest);
    Ok((verifier, challenge))
}

fn random_hex(n: usize) -> Result<String> {
    Ok(urandom_bytes(n)?
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect())
}

/// Open the user's default browser (best effort). Returns whether it was
/// spawned; the caller prints the URL so it can be opened manually otherwise.
async fn open_browser(url: &str) -> bool {
    for cmd in ["xdg-open", "gio", "open"] {
        if let Ok(status) = tokio::process::Command::new(cmd)
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
        {
            if status.success() {
                return true;
            }
        }
    }
    false
}

/// Run the Spotify OAuth (PKCE) flow: open the browser, catch the redirect on
/// 127.0.0.1:8888/callback, exchange the code, store the refresh token in the
/// keyring, and return it.
pub async fn authorize(client_id: &str) -> Result<String> {
    authorize_with(client_id, |line| println!("{}", line)).await
}

/// Same as [`authorize`] but lets the caller decide where each step's message
/// goes (CLI prints to stdout; the TUI setup screen feeds it to a panel).
pub async fn authorize_with<F>(client_id: &str, mut emit: F) -> Result<String>
where
    F: FnMut(&str),
{
    let (verifier, challenge) = pkce_pair()?;
    let state = random_hex(16)?;

    let url = format!(
        "{}?client_id={}&response_type=code&redirect_uri={}&scope={}&state={}&code_challenge={}&code_challenge_method=S256",
        AUTH_URL,
        client_id,
        urlencode(REDIRECT_URI),
        urlencode(OAUTH_SCOPE),
        state,
        challenge
    );

    emit("Opening your browser — approve Luna's access to Spotify…");
    if !open_browser(&url).await {
        emit(&format!("Couldn't open a browser automatically. Open this URL manually:\n  {}", url));
    }

    // Catch the redirect. Loopback + ephemeral + state-pinned; Spotify codes
    // expire in ~10 minutes.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8888")
        .await
        .context("Could not bind 127.0.0.1:8888 for the OAuth callback — is another process using that port?")?;
    emit("Waiting for you to approve in the browser…");
    let (mut sock, _addr) = match tokio::time::timeout(
        Duration::from_secs(300),
        listener.accept(),
    )
    .await
    {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => anyhow::bail!("Callback accept failed: {e}"),
        Err(_) => anyhow::bail!("Timed out waiting for Spotify authorization (5 min)"),
    };

    // Read the request head up to the blank line.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut head = Vec::new();
    let mut buf = [0u8; 512];
    let mut total = 0usize;
    loop {
        let n = sock.read(&mut buf).await.context("read callback request")?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
        total += n;
        if head.windows(4).any(|w| w == b"\r\n\r\n") || total > 16384 {
            break;
        }
    }

    let head_str = String::from_utf8_lossy(&head);
    let query = head_str
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.split('?').nth(1))
        .unwrap_or("");

    let params: Vec<(String, String)> = query
        .split('&')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((k.to_string(), urldecode(v)))
        })
        .collect();
    let code = params
        .iter()
        .find(|(k, _)| k == "code")
        .map(|(_, v)| v.clone());
    let cb_state = params
        .iter()
        .find(|(k, _)| k == "state")
        .map(|(_, v)| v.clone());

    if code.is_none() {
        let _ = sock.write_all(
            b"HTTP/1.1 400 Bad Request\r\nContent-Length: 46\r\nConnection: close\r\n\r\nNo authorization code in the callback URL.",
        ).await;
        anyhow::bail!("Spotify redirected without an authorization code.");
    }
    if cb_state.as_deref() != Some(&state) {
        anyhow::bail!("OAuth state mismatch — possible CSRF, aborting.");
    }

    let _ = sock.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Length: 55\r\nConnection: close\r\n\r\nSpotify authorized. You can close this tab and return to Luna.",
    ).await;
    let _ = sock.shutdown().await;

    emit("Authorization received — swapping code for tokens…");

    let client = Client::new();
    let token: TokenResponse = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code.as_deref().unwrap_or_default()),
            ("redirect_uri", REDIRECT_URI),
            ("client_id", client_id),
            ("code_verifier", &verifier),
        ])
        .send()
        .await
        .context("Failed to exchange authorization code")?
        .json()
        .await
        .context("Failed to parse token response")?;

    if let Some(err) = token.error.as_deref() {
        if !err.is_empty() {
            anyhow::bail!(
                "Spotify token exchange failed: {} — {}",
                err,
                token.error_description.as_deref().unwrap_or("")
            );
        }
    }
    if token.refresh_token.is_empty() {
        anyhow::bail!("Spotify returned a token without a refresh_token");
    }
    crate::config::keyring_set(REFRESH_KEYRING, &token.refresh_token)?;
    Ok(token.refresh_token)
}

/// Fetch a fresh access token: use the cached one if still valid, else
/// exchange the stored refresh token.
async fn access_token(config: &LunaConfig) -> Result<String> {
    let refresh = config
        .spotify
        .refresh_token
        .as_deref()
        .ok_or_else(|| anyhow!("Spotify not configured — run `luna --spotify-auth`"))?;

    if let Ok(guard) = token_cache().lock() {
        if let Some((tok, at)) = guard.as_ref() {
            if at.elapsed() < Duration::from_secs(3500) {
                return Ok(tok.clone());
            }
        }
    }

    let client_id = config
        .spotify
        .client_id
        .as_deref()
        .ok_or_else(|| anyhow!("[spotify] client_id not set"))?;

    let (token, _expires_in, new_refresh) = refresh_token(client_id, refresh).await?;

    if let Some(new_refresh) = new_refresh {
        crate::config::keyring_set(REFRESH_KEYRING, &new_refresh).ok();
    }
    if let Ok(mut guard) = token_cache().lock() {
        *guard = Some((token.clone(), Instant::now()));
    }
    Ok(token)
}

/// POST a refresh-token grant. Returns (access_token, expires_in, new_refresh).
/// PKCE clients authenticate with their client_id in the body, no secret.
async fn refresh_token(client_id: &str, refresh: &str) -> Result<(String, u64, Option<String>)> {
    let client = Client::new();

    let token: TokenResponse = client
        .post(TOKEN_URL)
        .form(&[
            ("client_id", client_id),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
        ])
        .send()
        .await
        .context("Failed to reach Spotify accounts API")?
        .json()
        .await
        .context("Failed to parse refresh response")?;

    if let Some(err) = token.error.as_deref() {
        if !err.is_empty() {
            anyhow::bail!(
                "Spotify refresh failed: {} — {}",
                err,
                token.error_description.as_deref().unwrap_or("")
            );
        }
    }
    if token.access_token.is_empty() {
        anyhow::bail!("Spotify returned an empty access token");
    }
    let new_refresh = if token.refresh_token.is_empty() {
        None
    } else {
        Some(token.refresh_token)
    };
    Ok((token.access_token, token.expires_in, new_refresh))
}

/// Percent-encode for use in a query string.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn urldecode(s: &str) -> String {
    // Replace "+" then %XX sequences.
    let bytes = s.replace('+', " ");
    let mut out = Vec::with_capacity(bytes.len());
    let b = bytes.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("00"), 16)
            {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ── API helpers ───────────────────────────────────────────────────────────────

/// Pretty error from a Spotify error body: `{"error": {"status", "message"}}`
/// or the accounts-style `{"error": "msg"}`.
fn spotify_error_body(status: reqwest::StatusCode, body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(msg) = v["error"]["message"].as_str() {
            return format!("Spotify {}: {}", v["error"]["status"].as_i64().unwrap_or(status.as_u16() as i64), msg);
        }
        if let Some(msg) = v["error"].as_str() {
            return format!("Spotify error: {}", msg);
        }
    }
    format!("Spotify HTTP {}: {}", status, body.trim())
}

async fn api_send(
    client: &Client,
    method: &str,
    path: &str,
    query: Option<&[(&str, &str)]>,
    body: Option<Value>,
    config: &LunaConfig,
) -> Result<Value> {
    let token = access_token(config).await?;
    let url = format!("{}{}", API_URL, path);
    let mut req = client
        .request(
            match method {
                "PUT" => reqwest::Method::PUT,
                "POST" => reqwest::Method::POST,
                _ => reqwest::Method::GET,
            },
            &url,
        )
        .bearer_auth(&token);

    if let Some(q) = query {
        req = req.query(q);
    }
    if let Some(b) = body {
        req = req.json(&b);
    }

    let resp = req.send().await.context("Failed to reach Spotify API")?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if status.is_success() {
        if text.trim().is_empty() {
            // 204 No Content (player actions) — represent as "ok"
            return Ok(Value::String("ok".into()));
        }
        let v = serde_json::from_str(&text)
            .context("Failed to parse Spotify response")?;
        return Ok(v);
    }

    let msg = spotify_error_body(status, &text);
    if status == reqwest::StatusCode::UNAUTHORIZED {
        // Stale token — clear the cache so the next call re-auths.
        if let Ok(mut guard) = token_cache().lock() {
            *guard = None;
        }
        anyhow::bail!("{} — run `luna --spotify-auth` to re-authorize", msg);
    }
    anyhow::bail!("{}", msg)
}

// ── Actions ───────────────────────────────────────────────────────────────────

/// Dispatch a spotify tool invocation. `args` mirrors the tool parameters.
pub async fn run(
    action: &str,
    args: &Value,
    config: &LunaConfig,
) -> Result<String> {
    let client = Client::new();
    match action {
        "now" => now(&client, config).await,
        "pause" => {
            api_send(&client, "PUT", "/me/player/pause", None, None, config).await?;
            Ok("Paused Spotify.".into())
        }
        "resume" => {
            play(&client, None, None, config).await?;
            Ok("Playing Spotify.".into())
        }
        "next" => {
            api_send(&client, "POST", "/me/player/next", None, None, config).await?;
            Ok("Skipped to the next track.".into())
        }
        "previous" => {
            api_send(&client, "POST", "/me/player/previous", None, None, config).await?;
            Ok("Went back to the previous track.".into())
        }
        "shuffle" => {
            let state = args["on"].as_bool().unwrap_or(true);
            api_send(
                &client,
                "PUT",
                "/me/player/shuffle",
                Some(&[("state", if state { "true" } else { "false" })]),
                None,
                config,
            )
            .await?;
            Ok(if state { "Shuffle on." } else { "Shuffle off." }.into())
        }
        "play_liked" => play_liked(&client, args, config).await,
        "search" => search(&client, args, config).await,
        "play_track" => play_track(&client, args, config).await,
        "play_playlist" => play_playlist(&client, args, config).await,
        "play_artist" => play_artist(&client, args, config).await,
        "play_album" => play_album(&client, args, config).await,
        "devices" => devices(&client, config).await,
        other => Err(anyhow!(
            "Unknown spotify action '{}'. Valid: now, pause, resume, next, previous, \
             shuffle, play_liked, search, play_track, play_playlist, play_artist, \
             play_album, devices",
            other
        )),
    }
}

async fn now(client: &Client, config: &LunaConfig) -> Result<String> {
    let v = api_send(client, "GET", "/me/player", None, None, config).await?;
    let item = &v["item"];
    let name = item["name"].as_str().unwrap_or("unknown");
    if v["item"].is_null() {
        let device = v["device"]["name"].as_str().unwrap_or("?");
        let playing = if v["is_playing"].as_bool() == Some(true) { "playing" } else { "paused" };
        return Ok(format!("Spotify is open ({}, {}) but no track is loaded.", device, playing));
    }
    let artists: Vec<&str> = item["artists"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x["name"].as_str()).collect())
        .unwrap_or_default();
    let album = item["album"]["name"].as_str().unwrap_or("?");
    let devices: Vec<&str> = v["device"]["name"].as_str().into_iter().collect();
    let device = devices.first().copied().unwrap_or("?");
    let playing = if v["is_playing"].as_bool() == Some(true) { "Playing" } else { "Paused" };
    let shuffle = if v["shuffle_state"].as_bool() == Some(true) { ", shuffle on" } else { "" };
    Ok(format!(
        "{}: \"{}\" by {} ({} album) on {}{}.",
        playing, name, artists.join(", "), album, device, shuffle
    ))
}

/// Play with graceful fallback to a real device. `uris` or `context_uri` must
/// be provided when resuming a specific thing.
async fn play(
    client: &Client,
    context_uri: Option<&str>,
    uris: Option<Vec<&str>>,
    config: &LunaConfig,
) -> Result<()> {
    let body = if let Some(u) = context_uri {
        json!({ "context_uri": u })
    } else if let Some(u) = uris {
        json!({ "uris": u })
    } else {
        Value::Null
    };
    let body_opt = if body.is_null() { None } else { Some(body) };
    match api_send(client, "PUT", "/me/player/play", None, body_opt.clone(), config).await {
        Ok(_) => Ok(()),
        Err(e) => {
            // "No active device" — find one and retry with an explicit device_id.
            if e.to_string().contains("No active device")
                || e.to_string().contains("active device found")
            {
                let devs = api_send(client, "GET", "/me/player/devices", None, None, config).await?;
                let id = devs["devices"]
                    .as_array()
                    .and_then(|d| d.iter().find(|x| x["is_active"].as_bool() == Some(true)))
                    .or_else(|| devs["devices"].as_array().and_then(|d| d.first()))
                    .and_then(|d| d["id"].as_str())
                    .ok_or_else(|| anyhow!("No Spotify device available — open the Spotify app on any device"))?;
                api_send(client, "PUT", "/me/player/play", Some(&[("device_id", id)]), body_opt, config).await?;
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

async fn play_liked(client: &Client, args: &Value, config: &LunaConfig) -> Result<String> {
    let me = api_send(client, "GET", "/me", None, None, config).await?;
    let id = me["id"]
        .as_str()
        .ok_or_else(|| anyhow!("Could not resolve your Spotify user id"))?;
    let context = format!("spotify:user:{}:collection", id);
    play(client, Some(&context), None, config).await?;
    let with_shuffle = args["shuffle"].as_bool().unwrap_or(false);
    if with_shuffle {
        api_send(client, "PUT", "/me/player/shuffle", Some(&[("state", "true")]), None, config).await?;
        Ok("Playing your Liked Songs, shuffled.".into())
    } else {
        Ok("Playing your Liked Songs.".into())
    }
}

async fn search(client: &Client, args: &Value, config: &LunaConfig) -> Result<String> {
    let query = args["query"].as_str().unwrap_or("");
    if query.trim().is_empty() {
        return Ok("No search query provided.".to_string());
    }
    let v = api_send(
        client,
        "GET",
        "/search",
        Some(&[("q", query), ("type", "track"), ("limit", "5")]),
        None,
        config,
    )
    .await?;
    let tracks = &v["tracks"]["items"];
    if tracks.as_array().map(|a| a.is_empty()).unwrap_or(true) {
        return Ok(format!("No tracks found for '{}'.", query));
    }
    let lines: Vec<String> = tracks
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|t| {
            let name = t["name"].as_str()?;
            let artist = t["artists"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|a| a["name"].as_str())
                .unwrap_or("?");
            let uri = t["uri"].as_str().unwrap_or("?");
            Some(format!("- {} — {} ({})", name, artist, uri))
        })
        .collect();
    Ok(lines.join("\n"))
}

async fn play_track(client: &Client, args: &Value, config: &LunaConfig) -> Result<String> {
    let query = args["query"].as_str().unwrap_or("");
    if query.trim().is_empty() {
        return Ok("No track query provided.".to_string());
    }
    let v = api_send(
        client,
        "GET",
        "/search",
        Some(&[("q", query), ("type", "track"), ("limit", "1")]),
        None,
        config,
    )
    .await?;
    let mut tracks = v["tracks"]["items"].as_array().cloned().unwrap_or_default();
    if tracks.is_empty() {
        return Ok(format!("No track found for '{}'.", query));
    }
    let first = tracks.remove(0);
    let name = first["name"].as_str().unwrap_or("?");
    let artist = first["artists"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|a| a["name"].as_str())
        .unwrap_or("?");
    let uri = first["uri"].as_str().unwrap_or("");
    play(client, None, Some(vec![uri]), config).await?;
    Ok(format!("Playing \"{}\" by {}.", name, artist))
}

async fn play_playlist(client: &Client, args: &Value, config: &LunaConfig) -> Result<String> {
    let wanted = args["playlist"].as_str().unwrap_or("");
    if wanted.trim().is_empty() {
        return Ok("No playlist name provided.".to_string());
    }
    let v = api_send(client, "GET", "/me/playlists", Some(&[("limit", "50")]), None, config).await?;
    let items = v["items"].as_array().cloned().unwrap_or_default();
    let wanted = wanted.to_lowercase();
    let matches: Vec<&Value> = items.iter().filter(|p| {
        p["name"].as_str().map(|n| n.to_lowercase().contains(&wanted)).unwrap_or(false)
    }).collect();
    let pl = match matches.first() {
        Some(p) => *p,
        None => {
            let names: Vec<String> = items.iter().filter_map(|p| p["name"].as_str().map(str::to_string)).take(10).collect();
            if names.is_empty() {
                return Ok("No playlists found on this account.".to_string());
            }
            return Ok(format!(
                "No playlist matching '{}'. Your playlists include:\n{}",
                wanted, names.join("\n")
            ));
        }
    };
    let name = pl["name"].as_str().unwrap_or("?");
    let uri = pl["uri"].as_str().unwrap_or("");
    play(client, Some(uri), None, config).await?;
    Ok(format!("Playing playlist \"{}\".", name))
}

async fn play_artist(client: &Client, args: &Value, config: &LunaConfig) -> Result<String> {
    let query = args["query"].as_str().unwrap_or("");
    if query.trim().is_empty() {
        return Ok("No artist query provided.".to_string());
    }
    let v = api_send(client, "GET", "/search", Some(&[("q", query), ("type", "artist"), ("limit", "1")]), None, config).await?;
    let mut artists = v["artists"]["items"].as_array().cloned().unwrap_or_default();
    if artists.is_empty() {
        return Ok(format!("No artist found for '{}'.", query));
    }
    let a = artists.remove(0);
    let name = a["name"].as_str().unwrap_or("?");
    let uri = a["uri"].as_str().unwrap_or("");
    play(client, Some(uri), None, config).await?;
    Ok(format!("Playing {} radio.", name))
}

async fn play_album(client: &Client, args: &Value, config: &LunaConfig) -> Result<String> {
    let query = args["query"].as_str().unwrap_or("");
    if query.trim().is_empty() {
        return Ok("No album query provided.".to_string());
    }
    let v = api_send(client, "GET", "/search", Some(&[("q", query), ("type", "album"), ("limit", "1")]), None, config).await?;
    let mut albums = v["albums"]["items"].as_array().cloned().unwrap_or_default();
    if albums.is_empty() {
        return Ok(format!("No album found for '{}'.", query));
    }
    let a = albums.remove(0);
    let name = a["name"].as_str().unwrap_or("?");
    let artist = a["artists"]
        .as_array()
        .and_then(|x| x.first())
        .and_then(|x| x["name"].as_str())
        .unwrap_or("?");
    let uri = a["uri"].as_str().unwrap_or("");
    play(client, Some(uri), None, config).await?;
    Ok(format!("Playing \"{}\" by {}.", name, artist))
}

async fn devices(client: &Client, config: &LunaConfig) -> Result<String> {
    let v = api_send(client, "GET", "/me/player/devices", None, None, config).await?;
    let list = v["devices"].as_array().cloned().unwrap_or_default();
    if list.is_empty() {
        return Ok("No Spotify devices found — open the Spotify app on any device.".to_string());
    }
    let lines: Vec<String> = list
        .iter()
        .filter_map(|d| {
            let name = d["name"].as_str()?;
            let kind = d["type"].as_str().unwrap_or("device");
            let active = if d["is_active"].as_bool() == Some(true) { " (active)" } else { "" };
            let vol = d["volume_percent"].as_i64().unwrap_or(0);
            Some(format!("- {} [{}] volume {}%{}", name, kind, vol, active))
        })
        .collect();
    Ok(lines.join("\n"))
}