pub mod piper;
pub mod rvc;

use crate::config::{LunaConfig, VoiceMode};
use anyhow::Result;
use std::process::Command;

/// Speak text in the configured voice mode.
/// Config is passed in (loaded once at startup) — piper/rvc never reload it.
/// The microphone is muted for the duration of playback so Luna's own
/// voice never loops back and gets transcribed as a command (echo).
pub async fn speak(text: &str, mode: &VoiceMode, config: &LunaConfig) -> Result<()> {
    if *mode == VoiceMode::Off {
        return Ok(());
    }

    let cleaned = clean_for_speech(text);
    if cleaned.is_empty() {
        return Ok(());
    }

    set_mic_muted(true);
    let _guard = MicUnmuteGuard;

    let result = match mode {
        VoiceMode::Off => Ok(()),
        VoiceMode::Basic => piper::speak(&cleaned, config).await,
        VoiceMode::Jinx => {
            let wav_path = piper::synthesize_to_file(&cleaned, config).await?;
            rvc::convert_and_play(&wav_path, config).await
        }
    };

    result
}

/// Mutes/unmutes the default input source via PulseAudio/PipeWire.
/// No-op if pactl is unavailable.
fn set_mic_muted(muted: bool) {
    let _ = Command::new("pactl")
        .args([
            "set-source-mute",
            "@DEFAULT_SOURCE@",
            if muted { "1" } else { "0" },
        ])
        .status();
}

/// Ensures the mic is unmuted even if playback errors or panics.
struct MicUnmuteGuard;

impl Drop for MicUnmuteGuard {
    fn drop(&mut self) {
        set_mic_muted(false);
    }
}

pub fn clean_for_speech(text: &str) -> String {
    let mut out = text.to_string();

    // Remove code blocks entirely — don't read code out loud
    while let Some(start) = out.find("```") {
        if let Some(end) = out[start + 3..].find("```") {
            let content_end = start + 3 + end + 3;
            out.replace_range(start..content_end, "... (code block) ...");
        } else {
            break;
        }
    }

    out = out.replace('`', "");
    out = out.replace("**", "").replace("__", "").replace('*', "").replace('_', " ");
    out = out
        .replace("\\[", "").replace("\\]", "")
        .replace("\\(", "").replace("\\)", "")
        .replace("\\frac", "fraction")
        .replace("\\int", "integral of")
        .replace("\\,", " ");

    let lines: Vec<&str> = out.lines().map(|l| l.trim_start_matches('#').trim()).collect();
    out = lines.join(". ");

    out = out.replace("- ", "").replace("• ", "");

    while out.contains("  ") {
        out = out.replace("  ", " ");
    }

    out = out.replace('\n', " ").trim().to_string();

    // Strip emojis so TTS doesn't read them out as names/descriptions.
    crate::util::strip_emojis(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_emojis_before_speech() {
        let cleaned = clean_for_speech("Hello! 😊 How can I assist you? 🌍🔥");
        assert!(!cleaned.contains('😊'));
        assert!(!cleaned.contains('🌍'));
        assert!(!cleaned.contains('🔥'));
        assert!(cleaned.contains("Hello!"));
        assert!(cleaned.contains("How can I assist you?"));
    }

    #[test]
    fn leaves_plain_text_alone() {
        let cleaned = clean_for_speech("The temperature is 72 degrees today.");
        assert_eq!(cleaned, "The temperature is 72 degrees today.");
    }
}
