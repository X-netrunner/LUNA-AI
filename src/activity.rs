//! Live progress for long-running turns — "what is Luna doing right now".
//!
//! ## Why this exists
//!
//! The gap it fills was visible in a real session on 2026-10-02. Text mode
//! printed `Luna[security]: ` and then went silent for 16s, 22s, then 37s —
//! three consecutive ReAct iterations with no output whatsoever, because
//! `tools::execute` only ever wrote to `tracing`, and text mode does not render
//! `tracing`. From the terminal the turn looked hung. It was not; the model was
//! generating, and one of those iterations ran `sudo pacman -Syu`.
//!
//! That silence is not cosmetic. It is indistinguishable from a crash, and it is
//! indistinguishable from Luna quietly doing something irreversible. A user who
//! cannot see progress will either interrupt work that is fine, or trust output
//! that fabricates — which is what happened next turn.
//!
//! ## Why a broadcast rather than a return value
//!
//! The call chain is `ReactLoop::run` → `tools::execute`, and the surfaces that
//! want this (text mode, the TUI, the daemon, the WhatsApp bridge) each drive the
//! loop from a different place. Threading a callback or a progress value back up
//! through `run_loop` → `enrich` → the three call sites would mean every future
//! surface has to be remembered. A subscriber list means a surface opts in, and a
//! new tool is visible everywhere at once because the hook sits at the single
//! dispatch chokepoint — the same reasoning as the capability gate.
//!
//! ## Ordering and failure
//!
//! Subscribers are called synchronously, in registration order, and a panicking
//! subscriber must not take down the turn it is reporting on. Each call is
//! therefore isolated. A subscriber that blocks delays the turn: it is reporting
//! progress, so it should be cheap — write a line, push an event, nothing more.
//!
//! ## What is deliberately NOT here
//!
//! No timestamps and no durations. The surfaces own their own formatting, and a
//! duration computed here would have to be measured across an `await` boundary
//! that several subscribers would rather handle themselves.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// What a subscriber is told about.
#[derive(Debug, Clone)]
pub enum Event {
    /// A ReAct iteration began — the model is generating. This is the long,
    /// silent part: it was 16–37s per iteration on the 7B, and it is exactly the
    /// window where the user could not tell "thinking" from "hung".
    Iteration { n: u32 },
    /// A tool is about to run. `summary` is a short human-readable form of the
    /// arguments — the command for `run_shell`, the path for file tools.
    ToolStart { name: String, summary: String },
    /// A tool finished. `ok` distinguishes success from a refusal or error, which
    /// matters because a refusal is the user seeing the gate do its job.
    ToolEnd { name: String, ok: bool, elapsed: Instant },
}

type Sink = Box<dyn Fn(&Event) + Send + Sync>;

/// A registered sink. `id` is a monotonic counter, never reused, so a dropped
/// subscription removes exactly its own sink and not an equal-looking one.
struct Registration {
    id: u64,
    sink: Sink,
}

fn subscribers() -> &'static Mutex<Vec<Registration>> {
    static S: OnceLock<Mutex<Vec<Registration>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Vec::new()))
}

/// Subscribe to activity. Returns a guard that unregisters on drop.
///
/// Drop-based cleanup matters because these are process-lifetime surfaces: a
/// TUI that closes must not leave a sink behind holding a dead render target.
pub struct Subscription(u64);

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Ok(mut v) = subscribers().lock() {
            let id = self.0;
            v.retain(|r| r.id != id);
        }
    }
}

/// Register a sink. Keep the returned guard alive for as long as it should run.
pub fn subscribe<F>(f: F) -> Subscription
where
    F: Fn(&Event) + Send + Sync + 'static,
{
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    subscribers()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Registration { id, sink: Box::new(f) });
    Subscription(id)
}

/// Publish to every subscriber. Never panics, whatever a subscriber does.
pub fn publish(event: Event) {
    let guard = subscribers().lock();
    let Ok(v) = guard else { return };
    for reg in v.iter() {
        // Isolated on purpose: a subscriber exists to *report*, so a broken one
        // must not be able to abort the tool call or the turn it describes.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (reg.sink)(&event)));
    }
}

/// Convenience for the common case: a one-line human summary of a tool's
/// arguments.
///
/// Long argument blobs are the reason this exists at all. `run_shell` with a
/// whole script, or `write_file` with a file body, must not dump onto the user's
/// terminal — the point is to see that work is happening, not to read it.
pub fn summarise(name: &str, args: &serde_json::Value) -> String {
    let str_field = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());

    let full = str_field("command").or_else(|| str_field("query")).or_else(|| str_field("text"));

    if let Some(s) = full {
        let one_line = s.split_whitespace().collect::<Vec<_>>().join(" ");
        return clip(&one_line, 88);
    }
    if let Some(p) = str_field("path").or_else(|| str_field("file")) {
        return clip(&p, 88);
    }
    if let Some(u) = str_field("url") {
        return clip(&u, 88);
    }

    // Fall back to the key names, so the user at least sees which tool ran on
    // what shape of input rather than a bare tool name.
    let keys: Vec<&str> = args
        .as_object()
        .map(|o| o.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();
    if keys.is_empty() {
        name.to_string()
    } else {
        clip(&format!("{name}({})", keys.join(", ")), 88)
    }
}

/// Clip to `max` chars on a char boundary, marking that it was cut.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counting sink, to check delivery and ordering.
    fn counter() -> (Subscription, Arc<AtomicUsize>) {
        let n = Arc::new(AtomicUsize::new(0));
        let c = n.clone();
        (subscribe(move |_| { c.fetch_add(1, Ordering::SeqCst); }), n)
    }

    use std::sync::Arc;

    #[test]
    fn events_reach_every_subscriber() {
        let (_s1, a) = counter();
        let (_s2, b) = counter();
        publish(Event::Iteration { n: 1 });
        assert_eq!(a.load(Ordering::SeqCst), 1);
        assert_eq!(b.load(Ordering::SeqCst), 1);
    }

    /// The summary must show the command, because "run_shell" alone tells the
    /// user nothing about what is about to happen to their machine.
    #[test]
    fn a_shell_command_is_summarised_by_what_it_runs() {
        let args = serde_json::json!({"command": "sudo nmap -sS -p- 127.0.0.1"});
        assert_eq!(summarise("run_shell", &args), "sudo nmap -sS -p- 127.0.0.1");
    }

    /// A long script must not flood the terminal. The point is visible progress,
    /// not a second copy of the tool arguments.
    #[test]
    fn a_long_argument_is_clipped_rather_than_dumped() {
        let long = format!("echo {}", "a".repeat(400));
        let args = serde_json::json!({ "command": long });
        let s = summarise("run_shell", &args);
        assert!(s.chars().count() <= 88, "not clipped: {} chars", s.chars().count());
        assert!(s.ends_with('…'), "clipping should be visible: {s}");
    }

    /// Multi-line scripts collapse to one line, or the "line" would wrap and
    /// look like several separate activities.
    #[test]
    fn a_multiline_script_collapses_to_one_line() {
        let args = serde_json::json!({"command": "line one\nline two\n\nline three"});
        assert_eq!(summarise("run_shell", &args), "line one line two line three");
    }

    #[test]
    fn file_and_url_tools_summarise_by_target() {
        assert_eq!(
            summarise("read_file", &serde_json::json!({"path": "/tmp/x.rs"})),
            "/tmp/x.rs"
        );
        assert_eq!(
            summarise("fetch_page", &serde_json::json!({"url": "https://example.com"})),
            "https://example.com"
        );
    }

    /// An argument set with no recognisable field still names its shape, so the
    /// user sees the tool ran on something rather than nothing.
    #[test]
    fn an_unrecognised_argument_set_falls_back_to_its_keys() {
        let s = summarise("todoist_add", &serde_json::json!({"content": "x", "due": "y"}));
        assert!(s.starts_with("todoist_add("), "got {s}");
        assert!(s.contains("content") && s.contains("due"), "got {s}");
    }

    /// A subscriber that panics must not break the turn it is reporting on. This
    /// is the whole reason `publish` isolates each call.
    #[test]
    fn a_panicking_subscriber_cannot_break_the_turn() {
        let _s = subscribe(|_| panic!("subscriber is broken"));
        let (_ok, n) = counter();
        publish(Event::Iteration { n: 2 });
        assert_eq!(
            n.load(Ordering::SeqCst),
            1,
            "a broken subscriber stopped delivery to the others"
        );
    }

    /// Subscribing with no surfaces attached must be free enough to call on every
    /// tool — `publish` runs unconditionally at the dispatch chokepoint.
    #[test]
    fn publishing_with_no_subscribers_is_harmless() {
        // The `counter` subscription above has been dropped by now in most test
        // orders, so this asserts the empty case does not panic or deadlock.
        publish(Event::ToolStart {
            name: "system_info".into(),
            summary: String::new(),
        });
    }
}
