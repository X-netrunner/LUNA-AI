//! util.rs — Tiny shared helpers

/// Truncate to at most `max` characters without ever splitting a UTF-8
/// character (byte-slicing a string mid-character panics). No allocation.
pub fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}
