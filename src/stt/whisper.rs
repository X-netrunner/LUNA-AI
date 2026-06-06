//! stt/whisper.rs — Speech to text via whisper-cli subprocess
//!
//! Calls whisper-cli as a subprocess, reads the output txt file.
//! Avoids all CUDA compilation issues with whisper-rs bindings.

use anyhow::{Context, Result};
use std::process::Stdio;

// Whisper hallucinates these tokens on silence/noise — filter them out.
// Keep this list growing if you see new ones in logs.
const HALLUCINATION_PATTERNS: &[&str] = &[
    "[music]",
    "[silence]",
    "[blank_audio]",
    "(music)",
    "(silence)",
    "[ music ]",
    "[ silence ]",
    "thank you.",
    "thanks for watching",
    "thanks for watching.",
    "please subscribe",
];

pub struct WhisperStt {
    model_path: String,
    /// Initial prompt fed to Whisper — massively helps with accents and
    /// domain-specific words. Think of it as priming the decoder.
    /// Set to None to disable.
    initial_prompt: Option<String>,
}

impl WhisperStt {
    pub fn new(model_path: &str) -> Self {
        Self {
            model_path: model_path.to_string(),
            // Default prompt: casual English conversation style.
            // This nudges Whisper toward common phrasing and helps it not
            // over-correct non-native speakers toward "standard" accents.
            initial_prompt: Some(
                "This is a conversation with an AI assistant named Luna. \
                 The user may have an accent. Transcribe exactly what is said."
                    .into(),
            ),
        }
    }

    /// Create with a custom initial prompt (or None to disable)
    pub fn with_prompt(model_path: &str, prompt: Option<String>) -> Self {
        Self {
            model_path: model_path.to_string(),
            initial_prompt: prompt,
        }
    }

    /// Transcribe a wav file to text using whisper-cli
    pub async fn transcribe(&self, wav_path: &str) -> Result<String> {
        tracing::debug!("Transcribing: {}", wav_path);

        let mut cmd = tokio::process::Command::new("whisper-cli");
        cmd.args([
            "--model",
            &self.model_path,
            "--file",
            wav_path,
            "--output-txt",
            "--language",
            "en",
            // Best of N beam search — improves accuracy at small cost
            "--best-of",
            "5",
            "--beam-size",
            "5",
            // No timestamps — cleaner output for assistant use
            "--no-timestamps",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null());

        // Add initial prompt if set — this is the #1 knob for accent adaptation
        if let Some(ref prompt) = self.initial_prompt {
            cmd.args(["--prompt", prompt]);
        }

        cmd.output()
            .await
            .context("Failed to spawn whisper-cli — is whisper.cpp installed?")?;

        // whisper-cli writes output to <wav_path>.txt
        let txt_path = format!("{}.txt", wav_path);

        let raw = tokio::fs::read_to_string(&txt_path)
            .await
            .unwrap_or_default();

        // Clean up temp file
        tokio::fs::remove_file(&txt_path).await.ok();

        let text = clean_transcript(&raw);
        if text.is_empty() {
            tracing::debug!("Transcription empty (silence or hallucination filtered)");
        } else {
            tracing::info!("Transcribed: \"{}\"", text);
            tracing::debug!("Transcribed: \"{}\"", text);
        }

        Ok(text)
    }
}

/// Strip whisper hallucinations and normalise whitespace.
fn clean_transcript(raw: &str) -> String {
    let lower = raw.trim().to_lowercase();

    // Exact-match hallucination check
    if HALLUCINATION_PATTERNS.iter().any(|p| lower == *p) {
        return String::new();
    }

    // Partial-match: if the whole transcript is essentially just a hallucination
    // token (with maybe minor punctuation), drop it
    let stripped = lower
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_string();
    if HALLUCINATION_PATTERNS
        .iter()
        .any(|p| stripped == p.trim_matches(|c: char| c.is_ascii_punctuation()))
    {
        return String::new();
    }

    raw.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_music_hallucination() {
        assert_eq!(clean_transcript("[Music]"), "");
        assert_eq!(clean_transcript("[SILENCE]"), "");
        assert_eq!(clean_transcript("  [blank_audio]  "), "");
        assert_eq!(clean_transcript("Thank you."), "");
    }

    #[test]
    fn keeps_real_speech() {
        assert_eq!(
            clean_transcript("Hey Luna, what time is it?"),
            "Hey Luna, what time is it?"
        );
        assert_eq!(clean_transcript("Open the terminal"), "Open the terminal");
    }
}
