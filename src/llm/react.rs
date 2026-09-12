//! llm/react.rs — ReAct agent loop with model escalation
//!
//! Simple queries → fast small model
//! Complex/tool queries → full 7B model
//! Empty responses → automatic retry with nudge

use crate::llm::ollama::{Message, OllamaClient, OllamaResponse};
use crate::memory::Memory;
use crate::tools;
use anyhow::Result;
use serde_json::json;
use std::io::Write;

pub struct ReactLoop<'a> {
    client: &'a OllamaClient,
    max_iterations: u8,
    /// Tools exposed to the model this loop drives (native Ollama tool use).
    /// Empty = no tools: the loop still supports freeform `name {json}` text
    /// calls, but no native function-calling is enabled.
    tools: Vec<crate::llm::ollama::ToolDef>,
    /// Loaded once at startup — tools must NOT reload config per call
    /// (that would redo keyring lookups and log noise every turn).
    config: crate::config::LunaConfig,
}

impl<'a> ReactLoop<'a> {
    pub fn new(
        client: &'a OllamaClient,
        max_iterations: u8,
        tools: Vec<crate::llm::ollama::ToolDef>,
        config: &crate::config::LunaConfig,
    ) -> Self {
        Self { client, max_iterations, tools, config: config.clone() }
    }

    /// True when the TUI is rendering — tool progress must go through
    /// tracing rather than direct print! to avoid corrupting the screen.
    fn tui(&self) -> bool {
        self.config.audio.input_mode == crate::config::InputMode::Tui
    }

    pub async fn run(
        &self,
        user_input: &str,
        memory: &mut Memory,
        system_prompt: &str,
    ) -> Result<(String, bool)> {
        memory.push(Message::user(user_input));

        let tools_arg = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.as_slice())
        };

        let mut iteration = 0;
        let mut turn_messages: Vec<Message> = Vec::new();
        let mut empty_retries = 0;

        loop {
            iteration += 1;
            if iteration > self.max_iterations {
                tracing::warn!("ReAct max iterations ({}) reached", self.max_iterations);
                let fallback = "I hit my iteration limit.".to_string();
                memory.push(Message::assistant(&fallback));
                return Ok((fallback, false));
            }

            let mut context = memory.build_context(system_prompt);
            context.extend(turn_messages.clone());

            tracing::debug!("ReAct iteration {}, context: {}", iteration, context.len());

            let response = self.client.chat(&context, tools_arg).await?;

            tracing::debug!(
                "model output {}: {}",
                if !self.tools.is_empty() { "(tools)" } else { "(text)" },
                crate::util::truncate(&truncate_control(&text_of(&response)), 600)
            );

            match response {
                OllamaResponse::Text { text, streamed } => {
                    // ── Empty response — retry with nudge ─────────────────
                    if text.trim().is_empty() {
                        empty_retries += 1;
                        if empty_retries >= 2 {
                            // Give up after 2 empty retries
                            let fallback = "I couldn't generate a response.".to_string();
                            memory.push(Message::assistant(&fallback));
                            return Ok((fallback, false));
                        }
                        tracing::warn!("Empty response, retrying ({}/2)...", empty_retries);
                        turn_messages.push(Message::user(
                            "Please respond or use a tool to complete the request.",
                        ));
                        continue;
                    }
                    empty_retries = 0;

                    // ── Freeform tool call intercept ──────────────────────
                    if let Some(tool_call) = parse_freeform_tool_call(&text) {
                        let tool_name = tool_call.function.name.clone();
                        tracing::info!("Intercepted freeform tool: {}", tool_name);
                        let tool_result = match tools::execute(&tool_call, &self.config).await {
                            Ok(o) => {
                                if self.tui() {
                                    tracing::info!("Tool {} succeeded: {}", tool_name, crate::util::truncate(&o, 120));
                                    print_sources(&tool_name, &o, true);
                                } else {
                                    print!("\n[Luna → {}] ", tool_name);
                                    std::io::stdout().flush().ok();
                                    println!("✓");
                                    print_sources(&tool_name, &o, false);
                                }
                                o
                            }
                            Err(e) => {
                                if self.tui() {
                                    tracing::warn!("Tool {} failed: {}", tool_name, e);
                                } else {
                                    print!("\n[Luna → {}] ", tool_name);
                                    std::io::stdout().flush().ok();
                                    println!("✗");
                                }
                                format!("Error: {}", e)
                            }
                        };

                        turn_messages.push(Message::assistant(format!(
                            "<|tool_call|>{}<|/tool_call|>",
                            tool_name
                        )));
                        turn_messages.push(Message::tool(tool_result));
                        continue;
                    }

                    // ── Genuine text response ─────────────────────────────
                    memory.push(Message::assistant(&text));
                    if let Err(e) = memory.save() {
                        tracing::warn!("Failed to save memory: {}", e);
                    }
                    return Ok((text, streamed));
                }

                OllamaResponse::ToolUse(tool_calls) => {
                    empty_retries = 0;
                    for tool_call in &tool_calls {
                        let tool_name = tool_call.function.name.clone();
                        tracing::info!("Tool call: {}", tool_name);

                        let tool_result = match tools::execute(tool_call, &self.config).await {
                            Ok(o) => {
                                if self.tui() {
                                    tracing::info!("Tool {} succeeded: {}", tool_name, crate::util::truncate(&o, 120));
                                    print_sources(&tool_name, &o, true);
                                } else {
                                    print!("\n[Luna → {}] ", tool_name);
                                    std::io::stdout().flush().ok();
                                    println!("✓");
                                    print_sources(&tool_name, &o, false);
                                }
                                o
                            }
                            Err(e) => {
                                if self.tui() {
                                    tracing::warn!("Tool {} failed: {}", tool_name, e);
                                } else {
                                    print!("\n[Luna → {}] ", tool_name);
                                    std::io::stdout().flush().ok();
                                    println!("✗");
                                }
                                format!("Error: {}", e)
                            }
                        };

                        tracing::debug!(
                            "Tool '{}' result: {}",
                            tool_name,
                            &crate::util::truncate(&tool_result, 160)
                        );

                        turn_messages.push(Message::assistant(format!(
                            "<|tool_call|>{}<|/tool_call|>",
                            tool_name
                        )));
                        turn_messages.push(Message::tool(tool_result));
                    }
                }
            }
        }
    }
}

/// Print the URLs a research tool referenced, so the user can click through.
/// In TUI mode these go through tracing so they land in the debug panel.
fn print_sources(tool_name: &str, result: &str, tui: bool) {
    let sources = crate::tools::extract_sources(tool_name, result);
    if sources.is_empty() {
        return;
    }
    if tui {
        tracing::info!("Sources: {}", sources.join(" | "));
    } else {
        println!("  Sources:");
        for src in &sources {
            println!("    ↳ {}", src);
        }
    }
}

fn parse_freeform_tool_call(text: &str) -> Option<crate::llm::ollama::ToolCall> {
    use crate::llm::ollama::{ToolCall, ToolCallFunction};
    let text = text.trim();

    // Pattern 1: [tool_call: name(args)]
    if text.starts_with("[tool_call:") {
        let inner = text.trim_start_matches("[tool_call:").trim();
        let paren_pos = inner.find('(')?;
        let name = inner[..paren_pos].trim().to_string();
        let rest = &inner[paren_pos + 1..];
        let close_pos = rest.rfind(')')?;
        let arguments: serde_json::Value = serde_json::from_str(&rest[..close_pos]).ok()?;
        return Some(ToolCall {
            function: ToolCallFunction { name, arguments },
        });
    }

    // Pattern 2: Called tool: name with args {...}
    if let Some(idx) = text.find("Called tool:") {
        let inner = text[idx..].trim_start_matches("Called tool:").trim();
        let parts: Vec<&str> = inner.splitn(2, " with args ").collect();
        if parts.len() == 2 {
            let name = parts[0].trim().to_string();
            let arguments: serde_json::Value = serde_json::from_str(parts[1].trim()).ok()?;
            return Some(ToolCall {
                function: ToolCallFunction { name, arguments },
            });
        }
    }

    // Pattern 3: <|tool_call|>name<|/tool_call|>  (echoed from memory format)
    // May appear multiple times or as the only content; extract the last one.
    if let Some(start) = text.rfind("<|tool_call|>") {
        let inner = &text[start + 13..]; // len("<|tool_call|>") = 13
        if let Some(end) = inner.find("<|/tool_call|>") {
            let name = inner[..end].trim().to_string();
            if !name.is_empty() {
                return Some(ToolCall {
                    function: ToolCallFunction {
                        name,
                        arguments: json!({}),
                    },
                });
            }
        }
    }

    // Pattern 4: known tool name followed by a JSON object — e.g.
    //   run_shell {"command": "ls"}   or   run_shell: {"command": "ls"}
    // This is the shorthand 7B models emit when streaming without native
    // tool bindings (e.g. after the Ollama 500 → chat_streaming fallback).
    if let Some(call) = parse_json_tool_call(text) {
        return Some(call);
    }

    None
}

/// Collapse embedded \n / \r / \t into spaces so tool-result dumps don't
/// turn one log line into dozens of wrapped rows.
fn truncate_control(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\n' | '\r' | '\t' => ' ',
            c => c,
        })
        .collect()
}

fn text_of(response: &OllamaResponse) -> String {
    match response {
        OllamaResponse::Text { text, .. } => text.clone(),
        OllamaResponse::ToolUse(calls) => {
            let names: Vec<String> = calls.iter().map(|c| c.function.name.clone()).collect();
            format!("[tools: {}]", names.join(", "))
        }
    }
}

/// Detect a standalone `ESCALATE` token in the response, or a general
/// refusal/uncertainty ("I don't know", "can't answer", …). The latter is the
/// general safety net: ANY time the fast model confesses it can't answer, the
/// turn re-runs on the full model — so we never need to predict phrasing.
pub(crate) fn is_escalation_response(text: &str) -> bool {
    if text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| w == "ESCALATE")
    {
        return true;
    }

    let t = text.to_lowercase();
    let phrases: &[&str] = &[
        "i don't know",
        "i dont know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "im not sure",
        "i have no idea",
        "i can't answer",
        "i cannot answer",
        "cannot answer",
        "can't answer that",
        "unable to answer",
        "i am unable to",
        "no information",
        "i can't help",
        "i cannot help",
    ];

    // A refusal is inherently short; only treat it as escalation when the
    // whole reply is the refusal (short) or it *starts* with the phrase.
    let begins = phrases.iter().any(|p| t.trim_start().starts_with(p));
    let contained_in_short_reply =
        text.trim().chars().count() < 200 && phrases.iter().any(|p| t.contains(p));
    begins || contained_in_short_reply
}

/// Try to read `tool_name {json}` out of the middle of a response.
fn parse_json_tool_call(text: &str) -> Option<crate::llm::ollama::ToolCall> {
    use crate::llm::ollama::{ToolCall, ToolCallFunction};

    let open = text.find('{')?;
    let head = text[..open].trim();
    // Allow: "run_shell" / "run_shell:" / "Call run_shell" / "tool run_shell"
    let name = head.split_whitespace().last()?.trim_end_matches(':');
    if name.is_empty() {
        return None;
    }
    let defs = crate::tools::tool_definitions();
    let known: Vec<&str> = defs.iter().map(|t| t.function.name.as_str()).collect();
    if !known.contains(&name) {
        return None;
    }

    // Extract the balanced JSON object starting at '{', ignoring trailing text.
    let obj = &text[open..];
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, ch) in obj.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let args: serde_json::Value = serde_json::from_str(&obj[..=i]).ok()?;
                    return Some(ToolCall {
                        function: ToolCallFunction { name: name.to_string(), arguments: args },
                    });
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_escalation_as_standalone_word() {
        assert!(is_escalation_response("ESCALATE"));
        assert!(is_escalation_response("Hi there, built by Netrunner! ESCALATE"));
        assert!(!is_escalation_response("escalate"));
        assert!(!is_escalation_response("I'll escalate this to the team."));
        assert!(!is_escalation_response(""));
        assert!(!is_escalation_response("ESCALATED"));
    }

    #[test]
    fn detects_refusal_and_uncertainty() {
        // The general safety net: a fast model that confesses ignorance
        // must escalate even without the literal ESCALATE token.
        assert!(is_escalation_response("I don't know about *Bleach the Anime*."));
        assert!(is_escalation_response("I don't know"));
        assert!(is_escalation_response("I'm not sure how to answer that."));
        assert!(is_escalation_response("I cannot answer this question."));
        assert!(is_escalation_response(
            "I have no idea about that. Ask me something else."
        ));
    }

    #[test]
    fn does_not_escalate_on_friendly_or_resolving_replies() {
        // Friendly replies with NO refusal phrase, and long substantive
        // answers that merely contain "don't know" mid-sentence, must NOT
        // escalate. (A short reply that mentions ignorance DOES escalate —
        // over-escalation only costs a full-model re-run, which is safe.)
        assert!(!is_escalation_response(
            "I'm Luna, built by Netrunner. Let me know if there's anything I can help with!"
        ));
        let long_answer = "Bleach is a long-running anime adapted from Tite Kubo's manga; \
            it aired from 2004 to 2012, following Ichigo Kurosaki. The Thousand-Year Blood War \
            arc returned in 2022. It is one of the 'big three' shonen series alongside \
            One Piece and Naruto. I don't know your dog's name though.";
        assert!(long_answer.chars().count() > 200);
        assert!(!is_escalation_response(long_answer));
    }

    #[test]
    fn parses_json_shorthand_tool_call() {
        let text = r#"run_shell {"command": "systemctl --user stop bluetooth"}"#;
        let call = parse_freeform_tool_call(text).unwrap();
        assert_eq!(call.function.name, "run_shell");
        assert_eq!(
            call.function.arguments["command"],
            "systemctl --user stop bluetooth"
        );
    }

    #[test]
    fn parses_json_shorthand_with_colon_and_prefix() {
        let call = parse_freeform_tool_call(r#"Call: run_shell: {"command": "ls"}"#).unwrap();
        assert_eq!(call.function.name, "run_shell");
        let call = parse_freeform_tool_call(r#"tool run_shell {"command": "ls"}"#).unwrap();
        assert_eq!(call.function.name, "run_shell");
    }

    #[test]
    fn rejects_non_tool_json_text() {
        assert!(parse_freeform_tool_call("Tell me the time").is_none());
        assert!(parse_freeform_tool_call(r#"{"command": "ls"}"#).is_none());
        assert!(parse_freeform_tool_call(r#"unknown_tool {"a": 1}"#).is_none());
    }

    #[test]
    fn still_parses_classic_patterns() {
        let call = parse_freeform_tool_call(r#"[tool_call: run_shell({"command": "ls"})]"#).unwrap();
        assert_eq!(call.function.name, "run_shell");
        let call =
            parse_freeform_tool_call(r#"Called tool: run_shell with args {"command": "ls"}"#).unwrap();
        assert_eq!(call.function.name, "run_shell");
        let call = parse_freeform_tool_call("<|tool_call|>run_shell<|/tool_call|>").unwrap();
        assert_eq!(call.function.name, "run_shell");
    }
}
