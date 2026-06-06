pub mod piper;
pub mod rvc;

use crate::config::{LunaConfig, VoiceMode};
use anyhow::Result;

/// Speak text in the configured voice mode.
/// Config is loaded once here and passed down — piper/rvc never reload it.
pub async fn speak(text: &str, mode: &VoiceMode) -> Result<()> {
    let cleaned = clean_for_speech(text);
    match mode {
        VoiceMode::Off => Ok(()),
        VoiceMode::Basic => {
            let config = LunaConfig::load()?;
            piper::speak(&cleaned, &config).await
        }
        VoiceMode::Jinx => {
            let config = LunaConfig::load()?;
            let wav_path = piper::synthesize_to_file(&cleaned, &config).await?;
            rvc::convert_and_play(&wav_path, &config).await
        }
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
    out.replace('\n', " ").trim().to_string()
}
