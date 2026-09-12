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
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
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
    #[serde(default)]
    thinking: Option<String>,
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
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

/// Pull a usable answer out of a qwen3-style thinking block when `content`
/// came back empty. Scans from the END and keeps the composed final answer,
/// discarding the front-loaded deliberation about HOW to answer. Returns ""
/// when the thinking is nothing but meta-monologue, so the caller falls back
/// to a retry nudge instead of echoing the model's instructions back at the
/// user.
fn thinking_as_answer(thinking: &str) -> String {
    let collapsed: String = thinking
        .split('\n')
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.trim().is_empty() {
        return String::new();
    }

    let sentences: Vec<&str> = collapsed
        .split('.')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if sentences.is_empty() {
        return String::new();
    }

    // Walk backwards, keeping up to 3 trailing non-meta sentences (the real
    // answer), and stop as soon as deliberation reappears in the middle.
    let mut kept: Vec<&str> = Vec::new();
    for s in sentences.iter().rev() {
        if is_meta_sentence(s) {
            if kept.is_empty() {
                continue;
            }
            break;
        }
        kept.push(s);
        if kept.len() >= 3 {
            break;
        }
    }
    if kept.is_empty() {
        return String::new();
    }
    kept.reverse();

    let mut answer = crate::util::strip_emojis(&kept.join(". ")).trim().to_string();
    answer = answer
        .trim_end_matches(|c: char| c == '.' || c.is_whitespace())
        .to_string();

    if answer.chars().count() < 8 || answer.chars().count() > 300 {
        return String::new();
    }
    answer
}

/// CoT / meta-monologue tell: the model narrating what it should do or
/// quoting its own instructions, rather than talking to the user. Sentences
/// matching any marker are treated as deliberation, never as the answer.
fn is_meta_sentence(sentence: &str) -> bool {
    let t = sentence.trim().to_lowercase();
    const META: &[&str] = &[
        // Instruction-recitation
        "the response should be",
        "the reply should be",
        "the answer should be",
        "should be 1-2",
        "should be 2-3",
        "1-2 sentenc",
        "2-3 sentenc",
        "sentences max",
        // Self-narration about the user / payload
        "i remember",
        "remember that",
        "the user is",
        "the user asked",
        "the user wants",
        "the user isn't",
        "the user said",
        "wait,",
        "but wait",
        "i should",
        "i need to",
        "i'll answer",
        "i'll respond",
        "i'll search",
        "i'll look",
        "i'll use",
        "i'm going to",
        "i am going to",
        "so i'll",
        "so i should",
        "so let me",
        "let me",
        "but first",
        "i think the",
        "i think it",
        "i guess",
        "let's",
        // Control / formatting deliberation
        "check if",
        "make sure",
        "no tool",
        "no need",
        "don't add",
        "do not add",
        "don't introduce",
        "introduce myself",
        "my introduction",
        "beyond the",
        "the system prompt",
        "the instruction",
        "given the",
        "based on my",
        "just a short",
        "just a quick",
        "as luna",
        "in my role",
        "as the assistant",
        // Drafting the reply ("mention the price, note if it's international")
        "mention",
        "i should mention",
        "also mention",
        "note that",
        "maybe the",
        "reply with",
        "respond with",
        "the price on",
        "so the best",
    ];
    META.iter().any(|m| t.contains(m))
}

// ── What our client returns ───────────────────────────────────────────────────

#[derive(Debug)]
pub enum OllamaResponse {
    /// Plain-text reply. `streamed` is true when tokens were already printed
    /// live to stdout, so the caller must not print the text again.
    Text { text: String, streamed: bool },
    ToolUse(Vec<ToolCall>),
}

// ── The client ────────────────────────────────────────────────────────────────

pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
    enable_thinking: bool,
    debug: bool,
    /// When false, never print tokens/thinking directly to stdout/stderr.
    /// The TUI renders its own screen, so stray print!()/eprintln!() would
    /// corrupt the alternate screen — they go through tracing instead.
    term_output: bool,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str, temperature: f32, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
            enable_thinking: true,
            debug: false,
            term_output: true,
        }
    }

    pub fn enable_thinking(mut self, on: bool) -> Self {
        self.enable_thinking = on;
        self
    }

    pub fn debug(mut self, on: bool) -> Self {
        self.debug = on;
        self
    }

    pub fn term_output(mut self, on: bool) -> Self {
        self.term_output = on;
        self
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
                num_ctx: 8192, // Ensure enough context for long prompts
                think: (!self.enable_thinking).then_some(false),
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
        let mut thinking_buf = String::new();
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

                // Collect thinking tokens — printed as a block when done
                if self.debug {
                    if let Some(ref think) = chunk.message.thinking {
                        thinking_buf.push_str(think);
                    }
                }

                if let Some(token) = chunk.message.content {
                    // Strip emojis as tokens arrive — they'd otherwise be
                    // echoed live to the terminal.
                    let clean = crate::util::strip_emojis(&token);
                    if self.term_output {
                        print!("{}", clean);
                        use std::io::Write;
                        std::io::stdout().flush().ok();
                    }
                    full_text.push_str(&clean);
                }

                if chunk.done {
                    // Log the accumulated thinking block — tracing in TUI mode
                    // routes it to the debug panel instead of raw stderr.
                    if !thinking_buf.is_empty() {
                        let think = crate::util::truncate_marked(&thinking_buf, 160);
                        let think_one_line = think
                            .chars()
                            .map(|c| match c {
                                '\n' | '\r' | '\t' => ' ',
                                c => c,
                            })
                            .collect::<String>();
                        if self.term_output {
                            eprintln!("\n[think] {}", think_one_line);
                        } else {
                            tracing::debug!("[think] {}", think_one_line);
                        }
                    }
                    if self.term_output {
                        println!();
                    }
                    break;
                }
            }
        }

        // A qwen3 thinking model can finish with empty content (answer stuck
        // in the thinking block) — rescue it so the caller never retries.
        if full_text.trim().is_empty() && !thinking_buf.is_empty() {
            let rescued = thinking_as_answer(&thinking_buf);
            if !rescued.is_empty() {
                tracing::debug!("Rescued answer from thinking ({} chars)", rescued.chars().count());
                full_text = rescued;
            }
        }

        Ok(OllamaResponse::Text {
            text: full_text,
            streamed: true,
        })
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
                num_ctx: 8192, // Ensure enough context for long prompts + tools
                think: (!self.enable_thinking).then_some(false),
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
                    crate::util::truncate(&body, 200)
                );
                return self.chat_streaming(messages).await;
            }

            anyhow::bail!("Ollama returned {}: {}", status, body);
        }

        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        let parsed: FullResponse = serde_json::from_str(&body).with_context(|| {
            format!(
                "Failed to parse Ollama response: {}",
                crate::util::truncate(&body, 300)
            )
        })?;

        // Tool call takes priority — if present, return it immediately
        if !parsed.message.tool_calls.is_empty() {
            // Surface thinking tokens in debug mode even when there's a tool call
            if self.debug {
                if let Some(ref think) = parsed.message.thinking {
                    let think = crate::util::truncate_marked(think, 500);
                    if self.term_output {
                        eprintln!("[think] {}", think);
                    } else {
                        tracing::debug!("[think] {}", think);
                    }
                }
            }
            return Ok(OllamaResponse::ToolUse(parsed.message.tool_calls));
        }

        // Text response — return it WITHOUT printing.
        // The caller (agent/mod.rs) owns all printing so there's one print site.
        if self.debug {
            if let Some(ref think) = parsed.message.thinking {
                let think = crate::util::truncate_marked(think, 500);
                if self.term_output {
                    eprintln!("[think] {}", think);
                } else {
                    tracing::debug!("[think] {}", think);
                }
            }
        }
        let mut text = crate::util::strip_emojis(&parsed.message.content.unwrap_or_default());

        // qwen3 thinking models sometimes return an empty `content` with the
        // real answer stuck in `thinking` — rescue it instead of surfacing an
        // empty reply and forcing a retry nudge.
        if text.trim().is_empty() {
            if let Some(think) = parsed.message.thinking.as_deref() {
                let rescued = thinking_as_answer(think);
                if !rescued.is_empty() {
                    tracing::debug!("Rescued answer from thinking ({} chars)", rescued.chars().count());
                    text = rescued;
                }
            }
        }

        Ok(OllamaResponse::Text {
            text,
            streamed: false,
        })
    }
}

impl OllamaClient {
    /// Embed one or more inputs via Ollama's /api/embed endpoint.
    pub async fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(serde::Deserialize)]
        struct EmbedResp {
            embeddings: Vec<Vec<f32>>,
        }

        let url = format!("{}/api/embed", self.base_url);
        let resp = self
            .client
            .post(&url)
            .timeout(std::time::Duration::from_secs(20))
            .json(&serde_json::json!({ "model": model, "input": inputs }))
            .send()
            .await
            .context("Failed to connect to Ollama for embeddings")?;

        anyhow::ensure!(
            resp.status().is_success(),
            "Ollama embed failed ({}): is '{}' pulled?",
            resp.status(),
            model
        );

        let parsed: EmbedResp = resp.json().await.context("Bad embed response")?;
        Ok(parsed.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_as_answer_skips_filler_and_keeps_tail() {
        let t = "The user asked why they are called the Big Three. \
                 I should search the web. \
                 So I'll use the web_search function with the query \"why is it called the big three\". \
                 They're called the Big Three because Naruto, One Piece, and Bleach were \
                 the three most popular shonen manga of their era.";
        let a = thinking_as_answer(t);
        assert!(a.contains("Big Three"), "got: {}", a);
        assert!(!a.contains("I should"), "filler leaked: {}", a);
        assert!(!a.contains("web_search function"), "meta leaked: {}", a);
    }

    #[test]
    fn thinking_as_answer_returns_empty_for_empty() {
        assert!(thinking_as_answer("").is_empty());
        assert!(thinking_as_answer("Okay.").is_empty());
        assert!(thinking_as_answer("   \n  ").is_empty());
    }

    #[test]
    fn thinking_as_answer_drops_instruction_monologue() {
        let t = ". The response should be 1-2 sentences max. First, I remember \
                 that Luna's intro is strictly \"I'm Luna, built by Netrunner\". \
                 Don't add more. The user is greeting me, so I should acknowledge \
                 it briefly. Check if there's any need for tools here. The user \
                 isn't asking for a command or action, just a hello. So no tool \
                 calls needed. Just a short reply. Make sure not to introduce \
                 myself beyond the requirements";
        assert!(
            thinking_as_answer(t).is_empty(),
            "pure meta-monologue leaked through: {:?}",
            thinking_as_answer(t)
        );
    }

    #[test]
    fn thinking_as_answer_skips_leading_deliberation_keeps_final_answer() {
        let t = "I should check the weather for the user's city. Let me use the \
                 weather tool. So I'll fetch current conditions. The weather in \
                 Bengaluru is 28 C, partly cloudy, with a 20% chance of rain";
        let a = thinking_as_answer(t);
        assert!(a.contains("Bengaluru is 28"), "answer lost: {:?}", a);
        assert!(!a.contains("I should"), "deliberation leaked: {}", a);
        assert!(!a.contains("tool"), "tool narration leaked: {}", a);
    }

    #[test]
    fn thinking_as_answer_rejects_drafting_monologue() {
        let t = "Mention the price on Amazon India, the features, and maybe the \
                 AliExpress option as a cheaper alternative but note that it's \
                 international. Wait, the user said \"in India\", so the best local \
                 option is Amazon India";
        assert!(
            thinking_as_answer(t).is_empty(),
            "drafting monologue leaked through: {:?}",
            thinking_as_answer(t)
        );
    }
}
