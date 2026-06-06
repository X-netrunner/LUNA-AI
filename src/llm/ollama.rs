//! llm/ollama.rs — Ollama API client
//!
//! Two modes:
//!   - No tools → streaming (feels fast, tokens appear as they generate)
//!   - With tools → non-streaming (tool calls come as one complete JSON blob)
//!
//! This split is necessary because Ollama sends tool_calls only in the final
//! response object, which conflicts with chunk-by-chunk stream parsing.

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── Message ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
        }
    }
}

// ── Tool definitions (sent to model so it knows what it can call) ─────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ── Tool call (what the model sends back when it wants to use a tool) ─────────

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ── Request shapes ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    options: ChatOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    temperature: f32,
    num_predict: u32,
    num_ctx: u32,
}

// ── Response shapes ───────────────────────────────────────────────────────────

// Streaming chunk (no tools)
#[derive(Debug, Deserialize)]
struct StreamChunk {
    message: StreamMessage,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct StreamMessage {
    content: Option<String>,
}

// Full response (with tools, non-streaming)
#[derive(Debug, Deserialize)]
struct FullResponse {
    message: FullMessage,
}

#[derive(Debug, Deserialize)]
struct FullMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

// ── What our client returns ───────────────────────────────────────────────────

#[derive(Debug)]
pub enum OllamaResponse {
    Text(String),
    ToolUse(Vec<ToolCall>),
}

// ── The client ────────────────────────────────────────────────────────────────

pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str, temperature: f32, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
        }
    }

    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<OllamaResponse> {
        if tools.is_some() {
            // Tool mode — non-streaming, parse one complete JSON response
            self.chat_with_tools(messages, tools).await
        } else {
            // Plain chat — streaming, print tokens as they arrive
            self.chat_streaming(messages).await
        }
    }

    // ── Streaming chat (no tools) ─────────────────────────────────────────────
    async fn chat_streaming(&self, messages: &[Message]) -> Result<OllamaResponse> {
        let request = ChatRequest {
            model: &self.model,
            messages,
            stream: true,
            options: ChatOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
                num_ctx: 4096, // Ensure enough context for long prompts
            },
            tools: None,
        };

        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to connect to Ollama — is it running?")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {}: {}", status, body);
        }

        let mut stream = response.bytes_stream();
        let mut full_text = String::new();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("Stream error")?;
            let text = std::str::from_utf8(&bytes).context("Non-UTF8 from Ollama")?;
            buffer.push_str(text);

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let chunk: StreamChunk = serde_json::from_str(&line)
                    .with_context(|| format!("Failed to parse chunk: {}", line))?;

                if let Some(token) = chunk.message.content {
                    print!("{}", token);
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                    full_text.push_str(&token);
                }

                if chunk.done {
                    println!();
                    break;
                }
            }
        }

        Ok(OllamaResponse::Text(full_text))
    }

    // ── Non-streaming chat with tools ─────────────────────────────────────────
    async fn chat_with_tools(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<OllamaResponse> {
        let request = ChatRequest {
            model: &self.model,
            messages,
            stream: false, // ← key difference: get one complete response
            options: ChatOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
                num_ctx: 4096, // Ensure enough context for long prompts + tools
            },
            tools,
        };

        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to connect to Ollama")?;


        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if status.as_u16() == 500 {
                tracing::warn!(
                    "Ollama 500 with tools — falling back to plain chat. \
                     Set native_tools=false in luna.toml to disable tool calls entirely. \
                     Error: {}",
                    &body[..body.len().min(200)]
                );
                return self.chat_streaming(messages).await;
            }

            anyhow::bail!("Ollama returned {}: {}", status, body);
        }

        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        // TEMP DEBUG — remove after diagnosis
        /*eprintln!(
            "\n[DEBUG RAW RESPONSE]\n{}\n[END DEBUG]\n",
            &body[..body.len().min(1000)]
        );*/

        let parsed: FullResponse = serde_json::from_str(&body).with_context(|| {
            format!(
                "Failed to parse Ollama response: {}",
                &body[..body.len().min(300)]
            )
        })?;

        // Tool call takes priority — if present, return it immediately
        if !parsed.message.tool_calls.is_empty() {
            return Ok(OllamaResponse::ToolUse(parsed.message.tool_calls));
        }

        // Text response — return it WITHOUT printing.
        // The caller (agent/mod.rs) owns all printing so there's one print site.
        let text = parsed.message.content.unwrap_or_default();
        Ok(OllamaResponse::Text(text))
    }
}
