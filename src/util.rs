//! util.rs — Tiny shared helpers

/// Truncate to at most `max` characters without ever splitting a UTF-8
/// character (byte-slicing a string mid-character panics). No allocation.
pub fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Strip emoji so they're never printed or read aloud by TTS
/// (e.g. 😊 spoken as "smiling face with smiling eyes").
pub fn strip_emojis(text: &str) -> String {
    text.chars().filter(|c| !is_emoji(*c)).collect()
}

fn is_emoji(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        // Emoticons, symbols & pictographs, regional indicators (flags),
        // supplemental symbols, transports
        0x1F000..=0x1FAFF |
        // Misc symbols & dingbats (⚠ ☀ ☕ ⭐ ❤ ✈ etc.)
        0x2600..=0x27BF |
        // Variation selectors (emoji presentation)
        0xFE00..=0xFE0F |
        // Zero-width joiner — glues multi-codepoint emoji together
        0x200D |
        // Combining enclosing keycap (1️⃣)
        0x20E3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_never_splits_utf8() {
        let s = "a😊bcd";
        assert_eq!(truncate(s, 2), "a😊");
        assert_eq!(truncate(s, 1), "a");
        assert_eq!(truncate(s, 100), s);
    }

    #[test]
    fn strips_emojis() {
        assert_eq!(strip_emojis("Hello 😊 🌍🔥"), "Hello  ");
        assert_eq!(strip_emojis("plain text"), "plain text");
    }
}
