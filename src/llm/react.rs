//! llm/react.rs — ReAct agent loop with model escalation
//!
//! Simple queries → fast small model
//! Complex/tool queries → full 7B model
//! Empty responses → automatic retry with nudge

use crate::llm::ollama::{Message, OllamaClient, OllamaResponse};
use crate::memory::Memory;
use crate::tools;
use anyhow::Result;

pub struct ReactLoop<'a> {
    client: &'a OllamaClient,
    max_iterations: u8,
    native_tools: bool,
}

impl<'a> ReactLoop<'a> {
    pub fn new(client: &'a OllamaClient, max_iterations: u8, native_tools: bool) -> Self {
        Self { client, max_iterations, native_tools }
    }

    pub async fn run(
        &self,
        user_input: &str,
        memory: &mut Memory,
        system_prompt: &str,
    ) -> Result<String> {
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
                return Ok(fallback);
            }

            let mut context = memory.build_context(system_prompt);
            context.extend(turn_messages.clone());

            tracing::debug!("ReAct iteration {}, context: {}", iteration, context.len());

            let response = self.client.chat(&context, tools_arg).await?;

            match response {
                OllamaResponse::Text(text) => {
                    // ── Empty response — retry with nudge ─────────────────
                    if text.trim().is_empty() {
                        empty_retries += 1;
                        if empty_retries >= 2 {
                            // Give up after 2 empty retries
                            let fallback = "I couldn't generate a response.".to_string();
                            memory.push(Message::assistant(&fallback));
                            return Ok(fallback);
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
                        tracing::info!("Intercepted freeform tool: {}", tool_call.function.name);
                        print!("\n[Luna → {}] ", tool_call.function.name);
                        use std::io::Write;
                        std::io::stdout().flush().ok();

                        let tool_result = match tools::execute(&tool_call, &get_config()).await {
                            Ok(o) => {
                                println!("✓");
                                o
                            }
                            Err(e) => {
                                println!("✗");
                                format!("Error: {}", e)
                            }
                        };

                        turn_messages.push(Message::assistant(&format!(
                            "<|tool_call|>{}<|/tool_call|>",
                            tool_call.function.name
                        )));
                        turn_messages.push(Message::tool(tool_result));
                        continue;
                    }

                    // ── Genuine text response ─────────────────────────────
                    memory.push(Message::assistant(&text));
                    if let Err(e) = memory.save() {
                        tracing::warn!("Failed to save memory: {}", e);
                    }
                    return Ok(text);
                }

                OllamaResponse::ToolUse(tool_calls) => {
                    empty_retries = 0;
                    for tool_call in &tool_calls {
                        let tool_name = tool_call.function.name.clone();
                        tracing::info!("Tool call: {}", tool_name);
                        print!("\n[Luna → {}] ", tool_name);
                        use std::io::Write;
                        std::io::stdout().flush().ok();

                        let tool_result = match tools::execute(tool_call, &get_config()).await {
                            Ok(o) => {
                                println!("✓");
                                o
                            }
                            Err(e) => {
                                println!("✗");
                                format!("Error: {}", e)
                            }
                        };

                        tracing::debug!(
                            "Tool '{}' result: {}",
                            tool_name,
                            &tool_result[..tool_result.len().min(200)]
                        );

                        turn_messages.push(Message::assistant(&format!(
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

/// Load config inline — needed for tool execution
fn get_config() -> crate::config::LunaConfig {
    crate::config::LunaConfig::load().unwrap_or_default()
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

    None
}
