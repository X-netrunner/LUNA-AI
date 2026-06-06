//! tts/piper.rs — TTS via Kokoro (replaces Piper)
//!
//! Kokoro runs via Python subprocess using the kokoro-onnx package.
//! Text is passed via the LUNA_TTS_TEXT env var — never interpolated
//! into the script string — so apostrophes, quotes, and special chars
//! can't break the Python syntax.

use anyhow::{Context, Result};
use std::process::Stdio;

/// Speak text using Kokoro TTS
pub async fn speak(text: &str, config: &crate::config::LunaConfig) -> Result<()> {
    let path = synthesize_to_file(text, config).await?;
    play_wav(&path).await?;
    tokio::fs::remove_file(&path).await.ok();
    Ok(())
}

/// Synthesize text to a temp wav file, return the path
pub async fn synthesize_to_file(text: &str, config: &crate::config::LunaConfig) -> Result<String> {
    let text = text.trim();
    if text.is_empty() {
        anyhow::bail!("Empty text — skipping TTS");
    }
    let out_path = format!("/tmp/luna_tts_{}.wav", uuid::Uuid::new_v4());
    let model = config.voice.piper_model.to_string_lossy().into_owned();

    let voice = config.voice.piper_bin.to_string_lossy().into_owned();
    let voice = if voice.is_empty() || voice == "/usr/bin/piper-tts" {
        "af_heart".to_string()
    } else {
        voice
    };

    let voices_bin = model.replace("kokoro-v1.0.onnx", "voices-v1.0.bin");

    let python = dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/luna/rvc_env/bin/python3")
        .to_string_lossy()
        .into_owned();

    let out = out_path.clone();

    // Text is injected via env var LUNA_TTS_TEXT — never interpolated into the
    // script string. This means apostrophes, quotes, backslashes, and every
    // other special character work correctly without any escaping.
    let script = format!(
        r#"
import os, sys
from kokoro_onnx import Kokoro
import soundfile as sf
kokoro = Kokoro('{model}', '{voices_bin}')
text = os.environ['LUNA_TTS_TEXT']
samples, sr = kokoro.create(text, voice='{voice}', speed=1.0, lang='en-us')
sf.write('{out}', samples, sr)
"#
    );

    let status = tokio::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .env("LUNA_TTS_TEXT", text) // ← safe: env vars handle any character
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to spawn python kokoro")?;

    if !status.success() {
        anyhow::bail!("Kokoro TTS failed");
    }

    Ok(out_path)
}

/// Play a wav file using rodio
async fn play_wav(path: &str) -> Result<()> {
    let path = path.to_string();
    tokio::task::spawn_blocking(move || {
        use rodio::{Decoder, OutputStream, Sink};
        use std::fs::File;
        use std::io::BufReader;

        let (_stream, stream_handle) =
            OutputStream::try_default().context("No audio output device found")?;
        let sink = Sink::try_new(&stream_handle).context("Failed to create audio sink")?;
        let file = BufReader::new(
            File::open(&path).with_context(|| format!("Failed to open wav: {}", path))?,
        );
        let source = Decoder::new(file).context("Failed to decode wav")?;
        sink.append(source);
        sink.sleep_until_end();
        Ok::<(), anyhow::Error>(())
    })
    .await
    .context("Audio playback panicked")??;

    Ok(())
}
