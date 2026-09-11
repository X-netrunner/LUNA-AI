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
    native_tools: bool,
    /// Loaded once at startup — tools must NOT reload config per call
    /// (that would redo keyring lookups and log noise every turn).
    config: crate::config::LunaConfig,
}

impl<'a> ReactLoop<'a> {
    pub fn new(
        client: &'a OllamaClient,
        max_iterations: u8,
        native_tools: bool,
        config: &crate::config::LunaConfig,
    ) -> Self {
        Self { client, max_iterations, native_tools, config: config.clone() }
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

        let tool_defs = tools::tool_definitions();
        let tools_arg = if self.native_tools { Some(tool_defs.as_slice()) } else { None };
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
                            &crate::util::truncate(&tool_result, 400)
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

/// Detect a standalone `ESCALATE` token in the response.
/// The 0.6b fast model sometimes appends "ESCALATE" after a greeting
/// (e.g. "Hi there, built by Netrunner! ESCALATE") — we still want to
/// escalate in that case.
pub(crate) fn is_escalation_response(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| w.eq_ignore_ascii_case("ESCALATE"))
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
