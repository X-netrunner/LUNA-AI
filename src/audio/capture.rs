//! audio/capture.rs — Two recording modes
//!
//! 1. `record_until_silence()` — used after wake word is confirmed.
//!    Records one full utterance (speech + trailing silence) and returns wav.
//!
//! 2. `listen_for_wake_word()` — continuous loop.
//!    Keeps listening, running Whisper on each detected utterance,
//!    and returns as soon as the wake word is heard.
//!    Caller then immediately calls record_until_silence() for the command.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, PartialEq)]
enum VadState {
    Waiting,
    Speaking,
    Silence,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Continuously listen and run Whisper on each utterance until the wake word
/// is detected. Returns the full transcribed utterance — the caller can strip
/// the wake word off the front to use it as an inline command (so
/// "luna what's the time" works in one breath).
///
/// `stt` is called on each captured chunk. If the returned text contains
/// the wake word, we return. Otherwise we discard and keep listening.
pub async fn listen_for_wake_word(
    sample_rate: u32,
    silence_ms: u64,
    wake_aliases: &[String],
    stt: &crate::stt::whisper::WhisperStt,
) -> Result<String> {
    loop {
        // Record one utterance (blocks until speech + silence)
        let wav_path = match record_until_silence(sample_rate, silence_ms).await {
            Ok(p) => p,
            Err(_) => continue, // timeout / no speech — keep waiting
        };

        let text = match stt.transcribe(&wav_path).await {
            Ok(t) => t,
            Err(_) => {
                tokio::fs::remove_file(&wav_path).await.ok();
                continue;
            }
        };
        tokio::fs::remove_file(&wav_path).await.ok();

        if text.is_empty() {
            continue;
        }

        let lower = text.to_lowercase();
        tracing::debug!("Wake word check: {:?}", lower);

        if contains_wake_alias(&lower, wake_aliases) {
            tracing::info!("Wake word detected in: {:?}", text);
            return Ok(text);
        }
        // Not the wake word — discard and keep listening silently
    }
}

/// True if the (lowercased) text contains any wake alias. Single-word
/// aliases like "luna" require a whole-word match so "lunatic"/"deluna"
/// never trigger; multi-word aliases just need a substring match.
pub fn contains_wake_alias(lower: &str, wake_aliases: &[String]) -> bool {
    wake_aliases.iter().any(|alias| {
        let alias_lower = alias.to_lowercase();
        if alias_lower.split_whitespace().count() == 1 {
            let word = alias_lower.trim();
            lower
                .split(|c: char| !c.is_alphanumeric())
                .any(|w| w == word)
        } else {
            lower.contains(&alias_lower)
        }
    })
}

/// Strip the wake alias out of a transcribed utterance and return whatever
/// came after it. Returns `None` if no alias was found, or `Some("")` if the
/// utterance was just the wake word on its own.
pub fn strip_wake_word(text: &str, wake_aliases: &[String]) -> Option<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        for alias in wake_aliases {
            let alias_lower = alias.to_lowercase();
            let alias_words: Vec<&str> = alias_lower.split_whitespace().collect();

            if alias_words.len() == 1 {
                if clean == alias_words[0] {
                    return Some(words[i + 1..].join(" "));
                }
            } else if i + alias_words.len() <= words.len()
                && alias_words.iter().enumerate().all(|(j, aw)| {
                    words[i + j].trim_matches(|c: char| !c.is_alphanumeric()) == *aw
                })
            {
                return Some(words[i + alias_words.len()..].join(" "));
            }
        }
    }

    None
}

/// Record one complete utterance (speech then silence) to a temp wav.
/// Returns the path. Errors if no speech detected within 30s.
pub async fn record_until_silence(sample_rate: u32, silence_ms: u64) -> Result<String> {
    let wav_path = format!("/tmp/luna_input_{}.wav", uuid::Uuid::new_v4());
    let wav_path_clone = wav_path.clone();

    tokio::task::spawn_blocking(move || record_blocking(&wav_path_clone, sample_rate, silence_ms))
        .await
        .context("Audio capture panicked")??;

    Ok(wav_path)
}

// ── Core recording ────────────────────────────────────────────────────────────

fn record_blocking(wav_path: &str, sample_rate: u32, silence_ms: u64) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input device found")?;

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // ── Calibration: 400ms ambient sample ────────────────────────────────────
    let baseline_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let baseline_clone = Arc::clone(&baseline_buf);
    let target = sample_rate as usize / 2; // 0.5s worth of samples

    let calib = device
        .build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mut buf = baseline_clone.lock().unwrap();
                if buf.len() < target {
                    buf.extend_from_slice(data);
                }
            },
            |e| tracing::error!("Calibration error: {}", e),
            None,
        )
        .context("Calibration stream failed")?;
    calib.play()?;
    std::thread::sleep(Duration::from_millis(500));
    drop(calib);

    let baseline_rms = {
        let buf = baseline_buf.lock().unwrap();
        (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    };

    // 3x ambient = speech; floor at 0.02 so dead-quiet rooms still work
    let speech_threshold = (baseline_rms * 3.0).max(0.02);
    tracing::debug!(
        "Ambient RMS: {:.4} | Speech threshold: {:.4}",
        baseline_rms,
        speech_threshold
    );

    // ── Recording ─────────────────────────────────────────────────────────────
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_w = Arc::clone(&samples);

    let state: Arc<Mutex<VadState>> = Arc::new(Mutex::new(VadState::Waiting));
    let state_w = Arc::clone(&state);

    let silence_since: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    let silence_since_w = Arc::clone(&silence_since);

    let loud_streak: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let loud_streak_w = Arc::clone(&loud_streak);
    const CONFIRM_FRAMES: usize = 3;

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _| {
                let rms = (data.iter().map(|s| s * s).sum::<f32>() / data.len() as f32).sqrt();
                let loud = rms > speech_threshold;

                let mut st = state_w.lock().unwrap();
                match *st {
                    VadState::Waiting => {
                        let mut streak = loud_streak_w.lock().unwrap();
                        if loud {
                            *streak += 1;
                            if *streak >= CONFIRM_FRAMES {
                                *st = VadState::Speaking;
                                *streak = 0;
                            }
                        } else {
                            *streak = 0;
                        }
                    }
                    VadState::Speaking => {
                        samples_w.lock().unwrap().extend_from_slice(data);
                        if !loud {
                            *st = VadState::Silence;
                            *silence_since_w.lock().unwrap() = Some(Instant::now());
                        }
                    }
                    VadState::Silence => {
                        samples_w.lock().unwrap().extend_from_slice(data);
                        if loud {
                            *st = VadState::Speaking;
                            *silence_since_w.lock().unwrap() = None;
                        }
                    }
                }
            },
            |e| tracing::error!("Audio stream error: {}", e),
            None,
        )
        .context("Failed to build recording stream")?;

    stream.play()?;

    let silence_duration = Duration::from_millis(silence_ms);
    let wait_start = Instant::now();

    loop {
        std::thread::sleep(Duration::from_millis(40));

        match state.lock().unwrap().clone() {
            VadState::Waiting => {
                // Hard timeout waiting for any speech at all
                if wait_start.elapsed() > Duration::from_secs(30) {
                    drop(stream);
                    anyhow::bail!("No speech detected within 30s");
                }
            }
            VadState::Speaking => {}
            VadState::Silence => {
                if let Some(since) = *silence_since.lock().unwrap() {
                    if since.elapsed() >= silence_duration {
                        break;
                    }
                }
            }
        }

        if samples.lock().unwrap().len() > sample_rate as usize * 60 {
            break; // hard cap 60s
        }
    }

    drop(stream);

    let samples = samples.lock().unwrap();
    if samples.is_empty() {
        anyhow::bail!("No audio captured");
    }

    let mut writer = hound::WavWriter::create(
        wav_path,
        hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .context("Failed to create wav")?;

    for &s in samples.iter() {
        writer.write_sample((s * i16::MAX as f32) as i16).ok();
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aliases() -> Vec<String> {
        vec![
            "luna".into(),
            "hey luna".into(),
            "hello luna".into(),
            "hay luna".into(),
        ]
    }

    #[test]
    fn single_word_alias_needs_whole_word() {
        let a = aliases();
        assert!(contains_wake_alias("luna what's the time", &a));
        assert!(contains_wake_alias("hey luna", &a));
        assert!(!contains_wake_alias("lunatic moon", &a));
        assert!(!contains_wake_alias("deluna", &a));
    }

    #[test]
    fn strips_wake_word_for_inline_command() {
        let a = aliases();
        assert_eq!(
            strip_wake_word("luna what's the time", &a),
            Some("what's the time".to_string())
        );
        assert_eq!(
            strip_wake_word("hey luna what time is it", &a),
            Some("what time is it".to_string())
        );
        assert_eq!(strip_wake_word("luna", &a), Some(String::new()));
        assert_eq!(
            strip_wake_word("no wake word here", &a),
            None
        );
    }
}
