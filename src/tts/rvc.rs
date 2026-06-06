//! tts/rvc.rs — Jinx voice via RVC (Retrieval Voice Conversion)
//!
//! Pipeline:
//!   1. Receive wav from piper (basic voice)
//!   2. Call rvc_infer.py as a subprocess with the Jinx model
//!   3. Play the converted wav via rodio

use anyhow::{Context, Result};
use std::process::Stdio;

pub async fn convert_and_play(input_wav: &str, config: &crate::config::LunaConfig) -> Result<()> {
    let output_wav = format!("/tmp/luna_jinx_{}.wav", uuid::Uuid::new_v4());
    convert(input_wav, &output_wav, config).await?;
    play_wav(&output_wav).await?;
    tokio::fs::remove_file(input_wav).await.ok();
    tokio::fs::remove_file(&output_wav).await.ok();
    Ok(())
}

async fn convert(input_wav: &str, output_wav: &str, config: &crate::config::LunaConfig) -> Result<()> {
    let rvc_script = config
        .voice
        .rvc_script
        .as_ref()
        .context("rvc_script not set in config — add it to luna.toml")?
        .to_string_lossy()
        .into_owned();

    let rvc_model = config
        .voice
        .rvc_model
        .as_ref()
        .context("rvc_model not set in config — add it to luna.toml")?
        .to_string_lossy()
        .into_owned();

    let rvc_index = rvc_model
        .replace(".pth", ".index")
        .replace("Jinx", "added_Jinx_v2");

    let python = dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/luna/rvc_env/bin/python3")
        .to_string_lossy()
        .into_owned();

    tracing::info!("Running RVC conversion: {} → {}", input_wav, output_wav);

    let status = tokio::process::Command::new(&python)
        .args([&rvc_script, input_wav, output_wav, &rvc_model, &rvc_index, "0"])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .await
        .with_context(|| format!("Failed to spawn python RVC script at {}", python))?;

    if !status.success() {
        anyhow::bail!("RVC conversion failed with exit code: {:?}", status.code());
    }

    Ok(())
}

async fn play_wav(path: &str) -> Result<()> {
    let path = path.to_string();
    tokio::task::spawn_blocking(move || {
        use rodio::{Decoder, OutputStream, Sink};
        use std::fs::File;
        use std::io::BufReader;

        let (_stream, stream_handle) =
            OutputStream::try_default().context("No audio output device")?;
        let sink = Sink::try_new(&stream_handle).context("Failed to create sink")?;
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
