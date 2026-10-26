//! stdio.rs — `luna --stdio [--history FILE]`: Luna as a child process of another program (Halcyon's Luna window).
//!
//! One JSON object per line in each direction, on stdout and stdin only (logs stay on stderr).
//!
//! In:   {"type":"turn","text":"..."}   one question; the reply comes back as one `reply` event
//!       {"type":"clear"}               forget the conversation held in the history file
//!       {"type":"quit"}                exit (end of input does the same)
//! Out:  {"type":"ready","model":"...","messages":N}
//!       {"type":"activity","kind":"step"|"tool_start"|"tool_end","name":"...","summary":"...","ok":true,"secs":1.2}
//!       {"type":"thinking","delta":"..."}                 the model's reasoning, as it streams
//!       {"type":"reply","text":"...","thinking":"..."|null,"model":"...","secs":3.4}
//!       {"type":"error","error":"..."}   {"type":"cleared"}
//!
//! The turn goes through `agent::run_routed_turn`, the same fast / deep / security / full routing the TUI uses, so the
//! answer comes from the same tier it would in a terminal. `--history` points the rolling conversation at a file of the
//! caller's choosing: a caller that keeps one file per conversation can resume any of them by starting Luna with that file.
//! Tools run exactly as they do in the terminal; nothing here adds or removes a permission.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::config::LunaConfig;
use crate::memory::Memory;

fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn activity_event(ev: &crate::activity::Event) -> Value {
    use crate::activity::Event;
    match ev {
        Event::Iteration { n } => json!({"type": "activity", "kind": "step", "name": "", "summary": "", "n": n}),
        Event::ToolStart { name, summary } => {
            json!({"type": "activity", "kind": "tool_start", "name": name, "summary": summary})
        }
        Event::ToolEnd { name, ok, elapsed } => json!({
            "type": "activity", "kind": "tool_end", "name": name, "ok": ok,
            "secs": (elapsed.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        }),
        Event::Thinking { delta } => json!({"type": "thinking", "delta": delta}),
    }
}

/// What one input line asks for. Anything that is not a recognised object is ignored, so a stray line never starts a turn.
#[derive(Debug, PartialEq)]
enum Request {
    Turn(String),
    Clear,
    Quit,
    Ignore,
}

fn parse_request(line: &str) -> Request {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Request::Ignore;
    };
    match v.get("type").and_then(Value::as_str) {
        Some("turn") => match v.get("text").and_then(Value::as_str).map(str::trim) {
            Some(t) if !t.is_empty() => Request::Turn(t.to_string()),
            _ => Request::Ignore,
        },
        Some("clear") => Request::Clear,
        Some("quit") => Request::Quit,
        _ => Request::Ignore,
    }
}

pub async fn run(config: &LunaConfig) -> Result<()> {
    let mut memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
    let _activity = crate::activity::subscribe(|ev| emit(activity_event(ev)));
    emit(json!({"type": "ready", "model": config.llm.model, "messages": memory.len()}));

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        match parse_request(&line) {
            Request::Quit => break,
            Request::Ignore => {}
            Request::Clear => {
                memory.clear()?;
                crate::llm::escalation::reset_security_latch();
                crate::recon::reset_latch();
                emit(json!({"type": "cleared"}));
            }
            Request::Turn(text) => {
                let started = Instant::now();
                match crate::agent::run_routed_turn(&text, &mut memory, config).await {
                    Ok(outcome) => {
                        crate::agent::learning::append_turn("user", &text);
                        crate::agent::learning::append_turn("assistant", &outcome.text);
                        emit(json!({
                            "type": "reply", "text": outcome.text, "thinking": outcome.thinking,
                            "model": outcome.model, "secs": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                        }));
                    }
                    Err(e) => emit(json!({"type": "error", "error": e.to_string()})),
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_needs_text() {
        assert_eq!(parse_request(r#"{"type":"turn","text":"  hi  "}"#), Request::Turn("hi".into()));
        assert_eq!(parse_request(r#"{"type":"turn","text":"   "}"#), Request::Ignore);
        assert_eq!(parse_request(r#"{"type":"turn"}"#), Request::Ignore);
    }

    #[test]
    fn control_lines_and_noise() {
        assert_eq!(parse_request(r#"{"type":"clear"}"#), Request::Clear);
        assert_eq!(parse_request(r#"{"type":"quit"}"#), Request::Quit);
        assert_eq!(parse_request("not json"), Request::Ignore);
        assert_eq!(parse_request(r#"{"type":"rm -rf"}"#), Request::Ignore);
    }

    #[test]
    fn activity_events_carry_their_kind() {
        let e = crate::activity::Event::ToolStart { name: "run_shell".into(), summary: "ls".into() };
        assert_eq!(activity_event(&e)["kind"], "tool_start");
        let t = crate::activity::Event::Thinking { delta: "hm".into() };
        assert_eq!(activity_event(&t)["type"], "thinking");
    }
}
