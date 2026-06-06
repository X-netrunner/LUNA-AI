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
/// is detected. Returns once the wake word is heard.
///
/// `stt` is called on each captured chunk. If the returned text contains
/// the wake word, we return. Otherwise we discard and keep listening.
pub async fn listen_for_wake_word(
    sample_rate: u32,
    silence_ms: u64,
    wake_aliases: &[String],
    stt: &crate::stt::whisper::WhisperStt,
) -> Result<()> {
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

        if wake_aliases.iter().any(|a| lower.contains(&a.to_lowercase())) {
            tracing::info!("Wake word detected in: {:?}", text);
            return Ok(());
        }
        // Not the wake word — discard and keep listening silently
    }
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
