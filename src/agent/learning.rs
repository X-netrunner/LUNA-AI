//! agent/learning.rs — Luna's self-improvement loop
//!
//! Inspired by the Hermes Agent (Nous Research): a periodic *memory nudge*
//! makes the model deliberately persist user knowledge, repeatable procedures
//! become reusable *skills*, and past conversations get *titles +
//! summaries* so they can be searched later. Everything degrades gracefully —
//! if Ollama or an embedding model is unavailable the blocks simply stay empty.

use crate::config::LunaConfig;
use crate::llm::ollama::{Message, OllamaClient, OllamaResponse};
use crate::memory::permanent::{Fact, PermanentMemory};
use crate::memory::{recall, skills};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

// ── Nudge instructions ────────────────────────────────────────────────────────

/// Injected every `nudge_interval` turns. Teaches the model to capture durable
/// user knowledge on its own initiative (Hermes's `background_review` nudge).
const MEMORY_NUDGE: &str = "\n\n[Memory review — for your own benefit, do not mention this]\n\
    Review this conversation INCLUDING the user's latest message: did the user \
    reveal a durable preference, personal detail, habit, or an expectation of \
    how you should behave? If yes and it's NEW and significant, call remember \
    right away (category user / preference / system / general; one sentence per \
    fact). Skip trivia, one-off requests, and anything already saved. Then \
    answer normally.";

/// The same nudge, extended for loops that have skill tooling (full/deep).
const SKILL_NUDGE: &str = "\n\n[Skill review — for your own benefit, do not mention this]\n\
    If THIS conversation contains a repeatable procedure (a multi-step task you \
    would do again, or something you had to figure out), save it now with \
    create_skill: a short kebab-case name, a one-line description, and a \
    step-by-step procedure. If you just used an existing skill and it was \
    wrong, outdated, or missing a step, re-save it with create_skill under the \
    same name to improve it. Skip it when nothing is procedural. Then answer \
    normally.";

/// Injected into every tier's system prompt so the model can answer questions
/// about its own capabilities without claiming to be a "text-based AI with no
/// voice". Kept in the learning module (alongside the other dynamic prompt
/// blocks) so the ReAct loop can place it correctly relative to them.
pub const SELF_AWARENESS: &str = "\n\n### Capabilities\n\
    - Luna is a voice-capable AI assistant running locally on Arch Linux (powered by Ollama models).\n\
    - Voice input: activate by saying your wake word (e.g. \"luna\"); speech is transcribed via Whisper.\n\
    - Hands-free voice mode: say \"luna voice mode\" to start, \"luna voice mode off\" to stop; auto-ends after configured idle time.\n\
    - Text-to-speech output: replies are spoken aloud when TTS is enabled.\n\
    - Tool use: web search, file and command actions on this machine (for deeper queries).\n\
    - If asked about your capabilities, answer honestly: you have voice wake detection, hands-free voice mode, and text-to-speech.";

// ── Paths ────────────────────────────────────────────────────────────────────

fn data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
}

static STATE_PATH: OnceLock<PathBuf> = OnceLock::new();

fn state_path() -> PathBuf {
    STATE_PATH
        .get()
        .cloned()
        .unwrap_or_else(|| data_dir().join("learning_state.json"))
}

/// Point `note_turn` at a scratch file (tests only, must be called before any
/// turn is counted in this process).
#[cfg(test)]
pub(crate) fn set_test_state_path(path: PathBuf) {
    let _ = STATE_PATH.set(path);
}

fn conversation_path() -> PathBuf {
    data_dir().join("conversations.jsonl")
}

fn sessions_path() -> PathBuf {
    data_dir().join("sessions.jsonl")
}

// ── Session identity ─────────────────────────────────────────────────────────

/// One id per process run — all turns recorded in one Luna session share it,
/// so session summaries group whole conversations correctly.
pub fn current_session_id() -> &'static str {
    static SESSION: OnceLock<String> = OnceLock::new();
    SESSION.get_or_init(|| chrono::Local::now().timestamp_millis().to_string())
}

// ── Memory nudge counter ─────────────────────────────────────────────────────

#[derive(Default, Serialize, Deserialize)]
struct LearningState {
    turn_count: u64,
    last_nudge: u64,
}

fn load_state() -> LearningState {
    std::fs::read_to_string(state_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(state: &LearningState) {
    if let Ok(json) = serde_json::to_string(state) {
        if let Some(dir) = state_path().parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(state_path(), json);
    }
}

/// Count one model turn; returns true when the nudge should fire this turn.
/// With a fresh state the counter starts at zero, so the first nudge lands on
/// turn `interval`, then every `interval` turns after (10, 20, 30, …).
/// `nudge_interval = 0` disables the periodic review entirely.
pub fn note_turn(config: &LunaConfig) -> bool {
    let mut state = load_state();
    state.turn_count = state.turn_count.saturating_add(1);
    let interval = config.memory.nudge_interval as u64;
    if interval == 0 {
        save_state(&state);
        return false;
    }
    let due = state.turn_count >= state.last_nudge + interval;
    if due {
        state.last_nudge = state.turn_count;
        tracing::debug!("Memory nudge due (turn {})", state.turn_count);
    }
    save_state(&state);
    due
}

// ── Conversation log ─────────────────────────────────────────────────────────

/// Append one turn to the append-only conversation log (searchable later).
pub fn append_turn(role: &str, content: &str) {
    let content = content.trim();
    if content.is_empty() {
        return;
    }
    let path = conversation_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Keep the log bounded — rotate to an archive at ~25 MB.
    if path
        .metadata()
        .map(|m| m.len() > 25 * 1024 * 1024)
        .unwrap_or(false)
    {
        let _ = std::fs::rename(&path, data_dir().join("conversations.old.jsonl"));
    }
    let entry = serde_json::json!({
        "ts": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        "session": current_session_id(),
        "role": role,
        "content": content,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{}", entry);
    }
}

/// Deterministic search over past conversations (turns + session summaries).
pub fn search_history(query: &str) -> String {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return "What should I search past conversations for?".to_string();
    }

    let mut results: Vec<String> = Vec::new();

    // 1. Session summaries first — a match there means the whole topic happened.
    if let Ok(raw) = std::fs::read_to_string(sessions_path()) {
        for line in raw.lines() {
            if !line.to_lowercase().contains(&q) {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let title = v["title"].as_str().unwrap_or("");
                let summary = v["summary"].as_str().unwrap_or("");
                results.push(format!("[session] {}: {}", title, summary));
            }
        }
    }

    // 2. Raw turns, newest first.
    if let Ok(raw) = std::fs::read_to_string(conversation_path()) {
        let mut hits: Vec<String> = Vec::new();
        for line in raw.lines().rev() {
            if hits.len() >= 10 {
                break;
            }
            if !line.to_lowercase().contains(&q) {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let role = v["role"].as_str().unwrap_or("");
                if role != "user" && role != "assistant" {
                    continue;
                }
                let ts = v["ts"].as_str().unwrap_or("");
                let content = v["content"].as_str().unwrap_or("");
                let who = if role == "user" { "You" } else { "Luna" };
                hits.push(format!(
                    "[{}] {}: {}",
                    ts,
                    who,
                    crate::util::truncate(content, 200)
                ));
            }
        }
        results.extend(hits);
    }

    if results.is_empty() {
        if q.chars().count() <= 3 {
            return "Too short to match. Ask about a topic, task, or phrase.".to_string();
        }
        return format!("Nothing in past conversations matched '{}'.", query);
    }

    let mut out = format!("Matches for '{}' ({}):\n", query, results.len());
    out.push_str(&results.join("\n"));
    out
}

// ── User profile (distilled from facts) ──────────────────────────────────────

/// Deterministic distill of the user's stable identity: everything permanently
/// categorised as `user` or `preference`, re-rendered as a prompt block. The
/// *curation* is the model's job (via the memory nudge); the render is ours.
pub fn profile_block() -> String {
    let Ok(pm) = PermanentMemory::load() else {
        return String::new();
    };
    let personal: Vec<&Fact> = pm
        .all_facts()
        .iter()
        .filter(|f| f.category == "user" || f.category == "preference")
        .collect();
    if personal.is_empty() {
        return String::new();
    }
    let rows = personal
        .iter()
        .map(|f| format!("- {}", f.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n[Stable facts about the user]\n{}\n", rows)
}

// ── Semantic recall blocks ───────────────────────────────────────────────────

async fn memory_recall_block(input: &str, config: &LunaConfig, k: usize) -> String {
    let Ok(pm) = PermanentMemory::load() else {
        return String::new();
    };
    let facts = pm.all_facts();
    if facts.is_empty() {
        return String::new();
    }
    match recall::relevant_facts(
        &config.llm.base_url,
        &config.llm.embedding_model,
        input,
        facts,
        k,
    )
    .await
    {
        Some(recalled) => recall::format_block(&recalled),
        None => pm.as_prompt_block(),
    }
}

async fn skills_recall_block(input: &str, config: &LunaConfig, k: usize) -> String {
    let Ok(metas) = skills::list() else {
        return String::new();
    };
    if metas.is_empty() {
        return String::new();
    }
    let recalled = skills::recall_top(
        &config.llm.base_url,
        &config.llm.embedding_model,
        input,
        &metas,
        k,
    )
    .await;
    if recalled.is_empty() {
        return String::new();
    }
    let rows = recalled
        .iter()
        .map(|m| format!("- {}: {}", m.name, m.description))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n[Relevant skills Luna can apply on request]\n{}\n", rows)
}

/// Assemble the whole dynamic enrichment block for one turn: recalled facts,
/// recalled skills, the stable user profile, and (when due) the nudges.
/// `has_skills` gates the skill-review nudge (only loops whose toolset actually
/// exposes `create_skill` get it).
pub async fn learning_block_for(
    input: &str,
    config: &LunaConfig,
    k: usize,
    nudge: bool,
    has_skills: bool,
) -> String {
    let mut out = String::new();
    out.push_str(&memory_recall_block(input, config, k).await);
    out.push_str(&skills_recall_block(input, config, k.saturating_sub(4).clamp(1, 3)).await);
    out.push_str(&profile_block());
    if nudge {
        out.push_str(MEMORY_NUDGE);
        if has_skills {
            out.push_str(SKILL_NUDGE);
        }
    }
    out
}

// ── Session summaries (Hermes-style titles + bullet summaries) ───────────────

const SUMMARY_SYSTEM: &str = "You are Luna. Summarize the conversation below for your own \
    long-term memory. Output EXACTLY this shape, nothing else:\n\
    TITLE: <one line, at most 8 words>\n\
    SUMMARY: <3-5 short bullet points, each starting with '-' and at most 12 words>";

fn render_session(turns: &[(String, String)]) -> String {
    let start = turns.len().saturating_sub(30);
    let mut out = String::new();
    for (role, content) in &turns[start..] {
        let label = if role == "user" { "You" } else { "Luna" };
        out.push_str(&format!(
            "{}: {}\n",
            label,
            crate::util::truncate(content, 500)
        ));
        if out.len() > 6000 {
            break;
        }
    }
    out
}

async fn summarize_text(config: &LunaConfig, text: &str) -> Option<(String, String)> {
    let client = OllamaClient::new(&config.llm.base_url, &config.llm.model, 0.2, 512);
    let context = vec![
        Message::system(SUMMARY_SYSTEM.to_string()),
        Message::user(format!("Conversation:\n{}", text)),
    ];
    let response = client.chat(&context, None).await.ok()?;
    let out = match response {
        OllamaResponse::Text { text, .. } => text,
        OllamaResponse::ToolUse(_) => return None,
    };
    let mut title = String::new();
    let mut summary = String::new();
    for line in out.lines() {
        if let Some(t) = line.strip_prefix("TITLE:") {
            title = t.trim().to_string();
        } else if let Some(s) = line.strip_prefix("SUMMARY:") {
            summary = s.trim().to_string();
        }
    }
    if title.is_empty() {
        return None;
    }
    if summary.is_empty() {
        summary = title.clone();
    }
    Some((title, summary))
}

fn append_session(session: &str, title: &str, summary: &str) {
    let path = sessions_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let entry = serde_json::json!({
        "session": session,
        "ts": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        "title": title,
        "summary": summary,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{}", entry);
    }
}

/// Distill previous (already-finished) sessions into titles + summaries, up to
/// three per launch so startup stays fast. Runs detached from the agent loop.
pub async fn summarize_pending_sessions(config: &LunaConfig) {
    let Ok(raw) = std::fs::read_to_string(conversation_path()) else {
        return;
    };
    if raw.trim().is_empty() {
        return;
    }

    let current = current_session_id().to_string();
    let mut by_session: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for line in raw.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let session = v["session"].as_str().unwrap_or("");
            let role = v["role"].as_str().unwrap_or("");
            let content = v["content"].as_str().unwrap_or("");
            if session.is_empty() || role.is_empty() || content.trim().is_empty() {
                continue;
            }
            if let Some(entry) = by_session.iter_mut().find(|(s, _)| s == session) {
                entry.1.push((role.to_string(), content.to_string()));
            } else {
                by_session.push((
                    session.to_string(),
                    vec![(role.to_string(), content.to_string())],
                ));
            }
        }
    }

    let mut done: HashSet<String> = HashSet::new();
    if let Ok(meta) = std::fs::read_to_string(sessions_path()) {
        for line in meta.lines() {
            if let Some(session) = line
                .split("\"session\":\"")
                .nth(1)
                .and_then(|s| s.split('"').next())
            {
                done.insert(session.to_string());
            }
        }
    }

    let pending: Vec<(String, Vec<(String, String)>)> = by_session
        .into_iter()
        .filter(|(s, _)| s != &current && !done.contains(s.as_str()))
        .collect();
    if pending.is_empty() {
        return;
    }

    let mut summarized = 0;
    for (session, turns) in pending {
        if summarized >= 3 {
            break;
        }
        let text = render_session(&turns);
        match summarize_text(config, &text).await {
            Some((title, summary)) => {
                append_session(&session, &title, &summary);
                summarized += 1;
                tracing::info!("Summarized session {}: {}", session, title);
            }
            None => break,
        }
    }
}

/// Fire-and-forget wrapper for startup: wait a moment so the first user turn
/// gets model time, then summarize.
pub fn spawn_session_summarizer(config: LunaConfig) {
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        summarize_pending_sessions(&config).await;
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LunaConfig {
        let mut cfg = LunaConfig::default();
        cfg.memory.nudge_interval = 3;
        cfg.memory.history_path = std::env::temp_dir().join("luna-learning-test-history.json");
        cfg
    }

    fn nudge_multiple_fire(config: &LunaConfig, n: u64) -> Vec<bool> {
        let mut results = Vec::new();
        for _ in 0..n {
            results.push(note_turn(config));
        }
        results
    }

    #[test]
    fn nudge_fires_on_interval() {
        let scratch =
            std::env::temp_dir().join(format!("luna-learning-state-{}", std::process::id()));
        set_test_state_path(scratch.clone());
        let _ = std::fs::remove_file(&scratch);

        let cfg = test_config();
        let results = nudge_multiple_fire(&cfg, 6);
        // Interval 3 from a zero baseline: fires on turns 3 and 6.
        assert_eq!(results, vec![false, false, true, false, false, true]);
    }

    #[test]
    fn session_id_is_stable_per_process() {
        let a = current_session_id();
        let b = current_session_id();
        assert!(!a.is_empty());
        assert_eq!(a, b);
    }

    #[test]
    fn search_history_rejects_empty() {
        assert!(search_history("  ").contains("What should I search"));
    }
}
