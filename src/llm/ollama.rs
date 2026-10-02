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

/// Context window requested from Ollama.
///
/// Measured, not guessed: the 9-clause Constitution (6.4 kB) plus the
/// 47-tool schema (29.8 kB) is ~8.0 k tokens before Luna has read a single
/// user message. A `files` + `read` round-trip on src/util.rs measured 9 326
/// tokens. At the old hardcoded 8192, Ollama does NOT return an error when
/// the prompt overflows — it silently context-shifts, and we measured a real
/// request dropping 8 156 -> 4 098 tokens (half the window, tools and file
/// contents included). Bigger targets are worse: util.rs is 60 lines, and
/// react.rs is over 2 000.
///
/// 16384 leaves room for a multi-step propose without asking the user to
/// think about context windows. Cost is KV cache RAM, which is small for a
/// 7B (~0.9 GB) but grows with model size — see README.
const DEFAULT_NUM_CTX: u32 = 16384;

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

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ── Constrained-decoding envelope (fallback path) ──────────────────────────────
//
// Why this exists, measured not guessed:
//
//   Ollama parses native `tool_calls` with a strict JSON reader. When a tool
//   call's `arguments` exceed roughly 100-200 tokens — which every non-trivial
//   self-patch does, because `find` strings contain newlines and quotes — the
//   parse fails and Ollama returns an EMPTY message. Not an error, not a
//   refusal: `done_reason: "stop"` with 139-976 tokens evaluated and no
//   content and no tool call. Observed 0/3 parsed on a 5-line edit and 0/3 on
//   a full test module, against 2/3 on a one-line edit.
//
//   This is what the old "intermittent empty response" notes actually were.
//   The correlation with eval_count (working calls 37-47 tokens, failing ones
//   131+) was never sampling noise — it tracks argument size, so retries
//   could never help.
//
// The fix is to stop depending on Ollama's tool-call parser for these. We ask
// for the same call as constrained JSON in `content` and parse it ourselves.
//
// Only a FALLBACK. Measured: forcing the envelope on every turn makes Luna
// unable to hold a normal conversation — "hey luna, what are you?" came back
// as a `web_search` call 3/3, with or without a "none" option. Native tool
// calls work fine for small calls and preserve prose replies, so we keep them
// as the primary path and only escalate on the empty-response signature.

/// `tool` is a free string, not an enum: the enum variant scored identically
/// (3/3) and a 48-entry enum in the grammar is a lot of context to spend on
/// something the `tools` schemas already describe.
fn tool_envelope_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "reply": {"type": "string"},
            "arguments": {"type": "object"}
        },
        "required": ["tool", "arguments"]
    })
}

#[derive(Debug, Deserialize)]
struct Envelope {
    tool: String,
    #[serde(default)]
    reply: Option<String>,
    #[serde(default)]
    arguments: serde_json::Value,
}

/// Outcome of reading a constrained-decoding envelope.
#[derive(Debug, PartialEq)]
enum EnvelopeRead {
    /// A tool call, ready to dispatch.
    Call(ToolCall),
    /// The model declined to call a tool and gave prose instead.
    Text(String),
    /// Not an envelope we can use.
    Invalid(String),
}

/// Parse the envelope back into a normal `ToolCall` so the ReAct loop, the
/// repeat guard, and the tool dispatcher are all unchanged by this path.
fn read_envelope(text: &str) -> EnvelopeRead {
    let parsed: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return EnvelopeRead::Invalid(format!("not JSON: {e}")),
    };
    let env: Envelope = match serde_json::from_value(parsed) {
        Ok(e) => e,
        Err(e) => return EnvelopeRead::Invalid(format!("wrong shape: {e}")),
    };
    let name = env.tool.trim();
    if name.is_empty() || name.eq_ignore_ascii_case("none") {
        return EnvelopeRead::Text(env.reply.unwrap_or_default());
    }
    EnvelopeRead::Call(ToolCall {
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: env.arguments,
        },
    })
}

/// Recover tool calls the model emitted as raw template markers.
///
/// Observed live: with `tools` present, Ollama returned qwen2.5's special
/// tokens verbatim in `content` and no `tool_calls` at all —
///
///     <|tool_call|>self_patch {"action":"files"}<|/tool_call|>
///
/// so the call was neither dispatched nor treated as an error; Luna just
/// printed the markers. Because the content is non-empty, the empty-response
/// envelope retry does not fire here — this is its own case.
///
/// The markers are perfectly well formed, so parsing them is strictly better
/// than showing them to the user as prose.
fn extract_marked_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;

    // qwen2.5 / hermes style
    while let Some(start) = rest.find("<|tool_call|>") {
        let after = &rest[start + "<|tool_call|>".len()..];
        let Some(end) = after.find("<|/tool_call|>") else {
            break;
        };
        let inner = after[..end].trim();
        rest = &after[end + "<|/tool_call|>".len()..];
        if let Some(c) = parse_marked_one(inner) {
            calls.push(c);
        }
    }

    // qwen3 style: a JSON array in [TOOL_CALLS] ... [/TOOL_CALLS]
    if calls.is_empty() {
        if let Some(open) = text.find("[TOOL_CALLS]") {
            let after = &text[open + "[TOOL_CALLS]".len()..];
            if let Some(end) = after.find("[/TOOL_CALLS]") {
                let inner = after[..end].trim();
                if let Ok(serde_json::Value::Array(items)) = serde_json::from_str(inner) {
                    for it in items {
                        // qwen3 emits [{"name":..,"arguments":{..}}, ..]
                        let name = it.get("name").and_then(|v| v.as_str());
                        let args = it.get("arguments").cloned();
                        if let (Some(n), Some(a)) = (name, args) {
                            calls.push(ToolCall {
                                function: ToolCallFunction { name: n.into(), arguments: a },
                            });
                        }
                    }
                }
            }
        }
    }

    // qwen2.5-coder:14b style: a fenced ```json block. Verified live — this
    // shape is what the deep model actually emits, and without handling it a
    // correctly-diagnosed task stalls forever: the log shows
    //   model output (tools): ```json {"name":"self_patch", ...} ```
    // and then nothing, because the call was never dispatched and the text was
    // printed to the user as if it were the answer.
    if calls.is_empty() {
        calls = extract_fenced_tool_calls(text);
    }

    calls
}

/// Pull tool calls out of ```json fenced blocks. Accepts a single object, an
/// array of them, and a `{"tool_calls": [...]}` wrapper, since the model varies.
///
/// `arguments` must be a JSON object — a call whose arguments are not an object
/// cannot be dispatched, and silently coercing one would be guessing.
fn extract_fenced_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("```json") {
        let after = &rest[start + "```json".len()..];
        let Some(end) = after.find("```") else {
            break;
        };
        let inner = after[..end].trim();
        rest = &after[end + 3..];
        if inner.is_empty() {
            continue;
        }
        if let Ok(value) = parse_lenient_json(inner) {
            collect_tool_calls(&value, &mut calls);
        }
    }
    calls
}

/// Parse JSON that a language model produced, repairing the two malformations
/// it reliably produces: `//` and `/* */` comments, and trailing commas.
///
/// This is not hypothetical. The live 14B emitted
/// ```json
/// {"at_line": 123, // Assuming the function starts at line 123
///  "replace_with": "..."}
/// ```
/// which `serde_json` rejects outright. Because the content is non-empty, no
/// other recovery path ran, so a correct tool call was printed to the user as
/// prose and the task silently did nothing. Stripping comments is what makes
/// fenced-block recovery actually work on this model.
///
/// Repairs are restricted to what is provably not JSON — a comment outside a
/// string, or a comma directly before `}`/`]`. Nothing inside a string literal
/// is touched, so a URL like "http://x" or a path stays intact.
fn parse_lenient_json(src: &str) -> Result<serde_json::Value, serde_json::Error> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(src) {
        return Ok(v);
    }
    let stripped = strip_json_comments(src);
    let trimmed = strip_trailing_commas(&stripped);
    if trimmed != src {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed) {
            return Ok(v);
        }
    }
    // Give the caller the original error, so a genuinely broken block is still
    // reported as unparseable rather than silently reinterpreted.
    serde_json::from_str::<serde_json::Value>(src)
}

/// Remove `//` and `/* */` comments that are OUTSIDE string literals.
fn strip_json_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_string = false;
    let mut escaped = false;
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
                i += 1;
            }
            '/' if bytes.get(i + 1) == Some(&'/') => {
                while i < bytes.len() && bytes[i] != '\n' {
                    i += 1;
                }
            }
            '/' if bytes.get(i + 1) == Some(&'*') => {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == '*' && bytes[i + 1] == '/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// Remove commas that sit immediately before a `}` or `]` (optionally with
/// whitespace in between). Commas elsewhere are preserved — a naive
/// implementation that drops a comma before any non-closer corrupts ordinary
/// JSON like `{"a":1,"b":2}`, so the comma's position is tracked in the output
/// and only retracted when a closer actually follows it.
fn strip_trailing_commas(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_string = false;
    let mut escaped = false;
    // Byte index in `out` of a comma that is not yet known to be trailing.
    let mut pending: Option<usize> = None;
    for c in src.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            ',' => {
                out.push(',');
                pending = Some(out.len() - 1);
            }
            '}' | ']' => {
                if let Some(i) = pending.take() {
                    // Whitespace between the comma and this closer goes too.
                    while out.len() > i + 1 && out.chars().next_back().is_some_and(char::is_whitespace)
                    {
                        out.pop();
                    }
                    out.remove(i);
                }
                out.push(c);
            }
            _ => {
                out.push(c);
                if !c.is_whitespace() {
                    pending = None;
                }
            }
        }
    }
    out
}

/// Recursively gather `{name, arguments}` objects from a parsed JSON value.
fn collect_tool_calls(value: &serde_json::Value, calls: &mut Vec<ToolCall>) {
    use serde_json::Value;
    match value {
        Value::Array(items) => {
            for it in items {
                collect_tool_calls(it, calls);
            }
        }
        Value::Object(map) => {
            // `{"tool_calls":[...]}` wrapper.
            if let Some(inner) = map.get("tool_calls") {
                collect_tool_calls(inner, calls);
                return;
            }
            let name = map.get("name").and_then(|v| v.as_str());
            let args = map
                .get("arguments")
                .or_else(|| map.get("parameters"))
                .or_else(|| map.get("args"));
            if let (Some(n), Some(a @ Value::Object(_))) = (name, args) {
                calls.push(ToolCall {
                    function: ToolCallFunction {
                        name: n.to_string(),
                        arguments: a.clone(),
                    },
                });
            }
        }
        _ => {}
    }
}

/// `"self_patch {\"action\":\"files\"}"` -> a ToolCall. Tolerant of the shapes
/// the model actually produces; returns None rather than guessing when the
/// arguments are not a JSON object.
fn parse_marked_one(inner: &str) -> Option<ToolCall> {
    let inner = inner.trim();
    if inner.is_empty() {
        return None;
    }
    // Prefer the first '{' so leading prose before the JSON is discarded.
    let brace = inner.find('{')?;
    let name = inner[..brace].trim();
    let json = inner[brace..].trim();
    if name.is_empty() {
        return None;
    }
    // The arguments may run to the end or stop at a stray '}'; try the whole
    // tail first, then progressively shorter prefixes.
    for end in (1..=json.len()).rev() {
        if !json.is_char_boundary(end) {
            continue;
        }
        if let Ok(v @ serde_json::Value::Object(_)) = serde_json::from_str::<serde_json::Value>(&json[..end])
        {
            return Some(ToolCall {
                function: ToolCallFunction { name: name.to_string(), arguments: v },
            });
        }
    }
    None
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
    /// JSON Schema for constrained decoding. Ollama constrains sampling to
    /// this grammar, so the reply is valid JSON by construction. Used only
    /// for the tool-call envelope fallback — see `chat_with_tools`.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'a serde_json::Value>,
    /// How long to keep the model loaded after this request, e.g. `"30m"`.
    ///
    /// A top-level Ollama field, not one of `options`. Sending it on every
    /// request is what stops a tier from being evicted between two related
    /// questions — with four tiers sharing one 6 GB card, the default 5-minute
    /// window means a follow-up often pays a full cold load from disk.
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
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
    /// Tokens actually generated. Not used for control flow, but it is the
    /// only way to tell "the model produced nothing" apart from "the model
    /// produced a lot that we failed to parse" when diagnosing an empty turn.
    #[serde(default)]
    eval_count: Option<u32>,
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
    Text { text: String, thinking: Option<String>, streamed: bool },
    /// `thinking` is carried here too, not just on `Text`.
    ///
    /// It used to be dropped on this arm: the tools path returned the calls and
    /// nothing else, logging the thinking block only under `self.debug`
    /// (default false). Every full and security turn goes through here, so Luna
    /// reasoned on every iteration and the reasoning never left the client. The
    /// `Text` arm already returned it, which is exactly why the omission was
    /// invisible — the one path that worked was the one that carries no tools.
    ToolUse {
        calls: Vec<ToolCall>,
        thinking: Option<String>,
    },
}

/// Drop a blank thinking block so callers don't have to distinguish "the model
/// thought nothing" from "the model isn't a thinking model".
///
/// Both arrive as `Some("")` or `Some("   ")` on some builds, and showing an
/// empty reasoning panel is worse than showing none.
fn present_thinking(t: Option<String>) -> Option<String> {
    t.filter(|s| !s.trim().is_empty())
}

// ── The client ────────────────────────────────────────────────────────────────

pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
    /// Ollama context window. This is NOT "room for the answer" — the system
    /// prompt plus the 47-tool schema is ~8k tokens before Luna says anything.
    /// At 8192 a single `self_patch read` pushed the request past the window
    /// and Ollama silently context-shifted (8156 -> 4098 tokens) instead of
    /// erroring, so she lost the file she had just read. See DEFAULT_NUM_CTX.
    num_ctx: u32,
    enable_thinking: bool,
    debug: bool,
    /// When false, never print tokens/thinking directly to stdout/stderr.
    /// The TUI renders its own screen, so stray print!()/eprintln!() would
    /// corrupt the alternate screen — they go through tracing instead.
    term_output: bool,
    /// How long Ollama keeps the model resident after a request, in minutes.
    ///
    /// Without this, Ollama's default is 5 minutes, and with four tiers on one
    /// 6 GB card a cold load is expensive: the 14B coder loads 12 GB from disk,
    /// and that exact cold load once stalled a single iteration for 23 minutes
    /// while the system swapped. Requests that alternate between tiers (an
    /// exploit question, then a code question, then a chat reply) would each
    /// pay that cost. A longer residency turns three reloads into one.
    ///
    /// Cost: the weights stay in RAM/VRAM between calls. That is the point, but
    /// it does mean the GPU is held, which matters on a card this size.
    keep_alive_mins: u32,
}

/// Default residency window, in minutes.
///
/// 30 was chosen to span a realistic working session rather than a single
/// question: long enough that a follow-up question about the same exploit does
/// not trigger a reload, short enough that an idle machine eventually frees the
/// card.
pub const DEFAULT_KEEP_ALIVE_MINS: u32 = 30;

impl OllamaClient {
    pub fn new(base_url: &str, model: &str, temperature: f32, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
            num_ctx: DEFAULT_NUM_CTX,
            enable_thinking: true,
            debug: false,
            term_output: true,
            keep_alive_mins: DEFAULT_KEEP_ALIVE_MINS,
        }
    }

    pub fn enable_thinking(mut self, on: bool) -> Self {
        self.enable_thinking = on;
        self
    }

    pub fn num_ctx(mut self, n: u32) -> Self {
        self.num_ctx = n;
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

    /// The `keep_alive` value as Ollama wants it: a duration string like
    /// `"30m"`, or `-1` to hold the model indefinitely until shutdown.
    fn keep_alive_value(&self) -> String {
        if self.keep_alive_mins == 0 {
            "0".to_string()
        } else {
            format!("{}m", self.keep_alive_mins)
        }
    }

    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<OllamaResponse> {
        self.chat_at(messages, tools, false).await
    }

    /// As `chat`, but `greedy` forces temperature 0 for this one request.
    /// Used to break out of a stochastic sampling failure (see the empty
    /// response handling in the ReAct loop).
    pub async fn chat_at(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
        greedy: bool,
    ) -> Result<OllamaResponse> {
        if tools.is_some() {
            // Tool mode — non-streaming, parse one complete JSON response
            self.chat_with_tools(messages, tools, greedy).await
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
                num_ctx: self.num_ctx,
                think: Some(self.enable_thinking),
            },
            tools: None,
            format: None,
            keep_alive: Some(self.keep_alive_value()),
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

                // Collect thinking tokens — saved and logged when done
                if let Some(ref think) = chunk.message.thinking {
                    thinking_buf.push_str(think);
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

        let thinking = if thinking_buf.trim().is_empty() {
            None
        } else {
            Some(thinking_buf)
        };

        Ok(OllamaResponse::Text {
            text: full_text,
            thinking,
            streamed: true,
        })
    }

    // ── Non-streaming chat with tools ─────────────────────────────────────────
    async fn chat_with_tools(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
        greedy: bool,
    ) -> Result<OllamaResponse> {
        let request = ChatRequest {
            model: &self.model,
            messages,
            stream: false, // ← key difference: get one complete response
            options: ChatOptions {
                temperature: if greedy { 0.0 } else { self.temperature },
                num_predict: self.max_tokens,
                num_ctx: self.num_ctx,
                // Always send the flag explicitly. The previous form was
                // `(!self.enable_thinking).then_some(false)`, which meant
                // enable_thinking = true sent NOTHING and only the `false`
                // setting could ever transmit a value — the config option was
                // silently dead and thinking could never be turned on.
                think: Some(self.enable_thinking),
            },
            tools,
            // Primary path: native tool calling. The constrained-decoding
            // envelope is only used if this comes back empty.
            format: None,
            keep_alive: Some(self.keep_alive_value()),
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
            return Ok(OllamaResponse::ToolUse {
                calls: parsed.message.tool_calls,
                thinking: present_thinking(parsed.message.thinking),
            });
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

        // The model emitted tool calls as raw template markers and Ollama
        // handed them back as prose. They are well formed — dispatch them
        // rather than showing the user `<|tool_call|>` on screen.
        let marked = extract_marked_tool_calls(&text);
        if !marked.is_empty() {
            tracing::info!(
                "Recovered {} tool call(s) from raw template markers",
                marked.len()
            );
            return Ok(OllamaResponse::ToolUse {
                calls: marked,
                thinking: present_thinking(parsed.message.thinking),
            });
        }

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

        // Nothing came back at all — no tool call, no prose, no thinking.
        //
        // This is the signature of Ollama's tool-call parser giving up on a
        // large `arguments` payload (see `tool_envelope_schema`). It used to
        // reach the caller as a blank reply, which surfaced as an unexplained
        // "intermittent" ~15% empty response and drove pointless retries.
        //
        // Re-ask the same turn as constrained JSON. Measured on the same
        // prompt that scored 0/3 natively: 3/3 valid calls. Prose replies are
        // non-empty, so this never fires on ordinary conversation.
        if text.trim().is_empty() && tools.is_some() {
            tracing::warn!(
                "Ollama returned an empty tool response (eval_count={}) — \
                 retrying with the constrained-decoding envelope",
                parsed.eval_count.unwrap_or(0)
            );
            return self.chat_via_envelope(messages, tools, greedy).await;
        }

        let thinking = present_thinking(parsed.message.thinking);
        Ok(OllamaResponse::Text {
            text,
            thinking,
            streamed: false,
        })
    }

    /// Fallback: ask for the tool call as constrained JSON in `content`
    /// instead of relying on Ollama's `tool_calls` parser.
    async fn chat_via_envelope(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
        greedy: bool,
    ) -> Result<OllamaResponse> {
        let schema = tool_envelope_schema();
        let request = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
            options: ChatOptions {
                temperature: if greedy { 0.0 } else { self.temperature },
                num_predict: self.max_tokens,
                num_ctx: self.num_ctx,
                think: Some(self.enable_thinking),
            },
            tools,
            format: Some(&schema),
            keep_alive: Some(self.keep_alive_value()),
        };

        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to connect to Ollama (envelope retry)")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {} on envelope retry: {}", status, body);
        }

        let body = response
            .text()
            .await
            .context("Failed to read envelope response body")?;
        let parsed: FullResponse = serde_json::from_str(&body).with_context(|| {
            format!(
                "Failed to parse Ollama envelope response: {}",
                crate::util::truncate(&body, 300)
            )
        })?;

        match read_envelope(parsed.message.content.unwrap_or_default().trim()) {
            EnvelopeRead::Call(call) => {
                tracing::info!("Recovered tool call '{}' via envelope", call.function.name);
                Ok(OllamaResponse::ToolUse {
                    calls: vec![call],
                    thinking: present_thinking(parsed.message.thinking),
                })
            }
            EnvelopeRead::Text(text) => {
                let text = crate::util::strip_emojis(&text);
                if text.trim().is_empty() {
                    // The envelope is constrained, so this is genuinely rare.
                    tracing::warn!("Envelope retry produced neither a call nor prose");
                }
                let thinking = present_thinking(parsed.message.thinking);
                Ok(OllamaResponse::Text {
                    text,
                    thinking,
                    streamed: false,
                })
            }
            EnvelopeRead::Invalid(why) => {
                anyhow::bail!("Envelope retry unusable: {why}")
            }
        }
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
    /// A blank reasoning block must not become an empty panel. Some builds send
    /// `Some("")` rather than omitting the field, and a visible empty thinking
    /// block is worse than none — it reads as a broken feature.
    #[test]
    fn a_blank_thinking_block_is_treated_as_absent() {
        assert_eq!(present_thinking(None), None);
        assert_eq!(present_thinking(Some(String::new())), None);
        assert_eq!(present_thinking(Some("   \n\t ".into())), None);
        assert_eq!(present_thinking(Some("reasoned".into())), Some("reasoned".into()));
    }

    /// The tool arm must be able to carry reasoning.
    ///
    /// This is the regression that motivated the change: `ToolUse` was a bare
    /// `Vec<ToolCall>`, so a tools turn had nowhere to put thinking even when
    /// Ollama sent it. Verified live against qwen3:8b, which returns a thinking
    /// block alongside `tool_calls` on both a direct tool request (389 chars)
    /// and one that asks for reasoning first (3172 chars).
    #[test]
    fn the_tool_arm_can_carry_thinking_alongside_the_calls() {
        let call = ToolCall {
            function: ToolCallFunction {
                name: "run_shell".into(),
                arguments: serde_json::json!({"command": "echo hi"}),
            },
        };

        let reasoned = OllamaResponse::ToolUse {
            calls: vec![call.clone()],
            thinking: Some("worked out 23*17 first".into()),
        };
        let OllamaResponse::ToolUse { calls, thinking } = &reasoned else {
            panic!("the tool arm changed shape again")
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(thinking.as_deref(), Some("worked out 23*17 first"));

        // A turn that acted without reasoning must still read as None, or the
        // TUI shows an empty panel on every ordinary tool call.
        let plain = OllamaResponse::ToolUse { calls: vec![call], thinking: None };
        let OllamaResponse::ToolUse { thinking, .. } = &plain else {
            panic!("the tool arm changed shape again")
        };
        assert!(thinking.is_none());
    }

    use super::*;

    // ── keep_alive ─────────────────────────────────────────────────────────

    fn client_with_keep_alive(mins: u32) -> OllamaClient {
        OllamaClient {
            keep_alive_mins: mins,
            ..OllamaClient::new("http://localhost:11434", "test-model", 0.7, 2048)
        }
    }

    /// The default must be long enough to span a follow-up question. Ollama's
    /// own default is 5 minutes, which with four tiers on one 6 GB card means a
    /// reload between "now scan the subnet" and "what did you find".
    #[test]
    fn keep_alive_defaults_to_a_working_session_not_ollamas_five_minutes() {
        let c = client_with_keep_alive(DEFAULT_KEEP_ALIVE_MINS);
        assert_eq!(c.keep_alive_value(), "30m");
        assert!(
            DEFAULT_KEEP_ALIVE_MINS >= 30,
            "too short: {DEFAULT_KEEP_ALIVE_MINS}m"
        );
    }

    /// Zero is meaningful to Ollama ("unload immediately"), not "unset" — it has
    /// to be sent as `0`, not omitted, or a caller asking for no residency gets
    /// the 5-minute default instead.
    #[test]
    fn keep_alive_zero_means_unload_not_unset() {
        assert_eq!(client_with_keep_alive(0).keep_alive_value(), "0");
    }

    /// `keep_alive` is a top-level field on /api/chat, NOT a member of
    /// `options`. Nesting it under options is silently accepted by Ollama and
    /// ignored — the model still reloads and nothing looks wrong.
    #[test]
    fn keep_alive_serializes_at_the_top_level_not_inside_options() {
        let req = ChatRequest {
            model: "m",
            messages: &[],
            stream: false,
            options: ChatOptions {
                temperature: 0.7,
                num_predict: 100,
                num_ctx: 8192,
                think: Some(false),
            },
            tools: None,
            format: None,
            keep_alive: Some("30m".to_string()),
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["keep_alive"], serde_json::json!("30m"));
        assert!(
            v["options"].get("keep_alive").is_none(),
            "keep_alive inside options is accepted and then ignored by Ollama"
        );
    }

    /// Every request path has to carry it. A single site missing means that one
    /// tier still evicts, which is invisible in testing and maddening in use.
    #[test]
    fn every_chat_request_site_sends_keep_alive() {
        let src = include_str!("ollama.rs");
        // Only production code — this test's own literal would match too.
        let prod = src.split("#[cfg(test)]\nmod tests {").next().unwrap();
        let built: usize = prod
            .matches("keep_alive: Some(self.keep_alive_value())")
            .count();
        let requests: usize = prod.matches("let request = ChatRequest {").count();
        assert_eq!(
            built, requests,
            "{built} of {requests} /api/chat request sites set keep_alive"
        );
        assert_eq!(requests, 3, "streaming, native tools, envelope fallback");
    }

    // ── Constrained-decoding envelope ──────────────────────────────────────
    //
    // These guard the fix for a bug that failed silently: Ollama's native
    // tool-call parser drops calls whose `arguments` run past ~100-200 tokens,
    // returning an empty message instead of an error.

    #[test]
    fn envelope_yields_a_normal_tool_call() {
        let raw = r#"{"tool":"self_patch","arguments":{"action":"propose",
            "changes":[{"file":"src/util.rs"}]}}"#;
        match read_envelope(raw) {
            EnvelopeRead::Call(c) => {
                assert_eq!(c.function.name, "self_patch");
                assert_eq!(c.function.arguments["action"], "propose");
                assert_eq!(c.function.arguments["changes"][0]["file"], "src/util.rs");
            }
            other => panic!("expected a call, got {other:?}"),
        }
    }

    /// The whole point of constrained decoding: a `find` string full of
    /// newlines and quotes must survive as escaped JSON. This is precisely the
    /// payload that made Ollama's own parser give up.
    #[test]
    fn envelope_survives_multiline_find_strings() {
        let find = "    #[test]\n    fn truncate_never_splits_utf8() {\n        assert_eq!(\n            truncate(\"héllo\", 3),\n            \"hél\"\n        );\n    }";
        let payload = serde_json::json!({
            "tool": "self_patch",
            "arguments": {
                "action": "propose",
                "changes": [{"file": "src/util.rs",
                             "edits": [{"find": find, "replace": ""}]}]
            }
        });
        match read_envelope(&payload.to_string()) {
            EnvelopeRead::Call(c) => {
                let got = &c.function.arguments["changes"][0]["edits"][0]["find"];
                assert_eq!(got.as_str().unwrap(), find);
            }
            other => panic!("expected a call, got {other:?}"),
        }
    }

    #[test]
    fn envelope_treats_none_and_blank_as_prose() {
        assert!(matches!(
            read_envelope(r#"{"tool":"none","reply":"hey, what can I do"}"#),
            EnvelopeRead::Text(t) if t == "hey, what can I do"
        ));
        assert!(matches!(
            read_envelope(r#"{"tool":"","reply":"hi"}"#),
            EnvelopeRead::Text(t) if t == "hi"
        ));
        assert!(matches!(
            read_envelope(r#"{"tool":"NONE"}"#),
            EnvelopeRead::Text(_)
        ));
    }

    #[test]
    fn envelope_rejects_junk_without_panicking() {
        for bad in [
            "",
            "   ",
            "not json at all",
            "{\"reply\":\"no tool field\"}",
            "[1,2,3]",
            "{\"tool\":123}",
        ] {
            assert!(
                matches!(read_envelope(bad), EnvelopeRead::Invalid(_)),
                "expected Invalid for {bad:?}"
            );
        }
    }

    /// `arguments` is optional in practice — a no-arg tool like `system_info`
    /// may omit it. That must not be treated as a parse failure.
    #[test]
    fn envelope_tolerates_missing_arguments() {
        match read_envelope(r#"{"tool":"system_info"}"#) {
            EnvelopeRead::Call(c) => {
                assert_eq!(c.function.name, "system_info");
                assert!(c.function.arguments.is_null() || c.function.arguments == serde_json::json!({}));
            }
            other => panic!("expected a call, got {other:?}"),
        }
    }

    /// The schema must stay a plain object with `tool`/`arguments` — if it
    /// ever grows a `oneOf`/`anyOf`, ollama's grammar builder may choke and
    /// the fallback would fail on every call.
    /// Observed live: with `tools` present, Ollama returned qwen2.5's markers
    /// verbatim in `content` and no `tool_calls`, so the call was silently
    /// shown to the user as prose. This is the exact string from that run —
    /// note it appeared TWICE, which the repeat guard would have caught had it
    /// been parsed as two calls rather than printed.
    #[test]
    fn recovers_qwen25_marker_tool_calls() {
        let text = "<|tool_call|>self_patch {\"action\":\"files\"}<|/tool_call|>\n\
                    <|tool_call|>self_patch {\"action\":\"files\"}<|/tool_call|>";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 2);
        for c in &calls {
            assert_eq!(c.function.name, "self_patch");
            assert_eq!(c.function.arguments["action"], "files");
        }
    }

    #[test]
    fn recovers_fenced_json_tool_call() {
        let text = "```json\n{\n  \"name\": \"self_patch\",\n  \"arguments\": {\"action\": \"read\", \"file\": \"src/tools/selfpatch.rs\"}\n}\n```";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1, "fenced block must dispatch, not print");
        assert_eq!(calls[0].function.name, "self_patch");
        assert_eq!(calls[0].function.arguments["action"], "read");
    }

    #[test]
    fn recovers_fenced_json_array_tool_calls() {
        let text = "```json\n[\n  {\"name\":\"self_patch\",\"arguments\":{\"action\":\"files\"}},\n  {\"name\":\"self_patch\",\"arguments\":{\"action\":\"read\",\"file\":\"src/util.rs\"}}\n]\n```";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn recovers_fenced_json_with_tool_calls_wrapper() {
        let text = "```json\n{\"tool_calls\": [{\"name\":\"self_patch\",\"arguments\":{\"action\":\"files\"}}]}\n```";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
    }

    /// Must not invent a call out of a fenced block that isn't one — e.g. a
    /// JSON example in prose, or arguments that aren't an object.
    #[test]
    fn fenced_non_tool_json_is_ignored() {
        assert!(extract_fenced_tool_calls("```json\n{\"city\": \"Ely\", \"pop\": 20000}\n```").is_empty());
        assert!(extract_fenced_tool_calls("```json\n{\"name\": \"self_patch\", \"arguments\": \"oops\"}\n```").is_empty());
        assert!(extract_fenced_tool_calls("```json\n{ this is not valid json }\n```").is_empty());
    }

    /// Prose around the block must not prevent recovery.
    #[test]
    fn fenced_tool_call_survives_surrounding_prose() {
        let text = "I'll read the file first.\n\n```json\n{\"name\":\"self_patch\",\"arguments\":{\"action\":\"read\",\"file\":\"src/util.rs\"}}\n```\n\nThis is the right move.";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments["file"], "src/util.rs");
    }

    /// The exact failure from the live 14B run: a `//` comment inside the JSON
    /// object made it unparseable, so the call printed as prose and the task
    /// silently did nothing. Reproduced verbatim, comment included.
    #[test]
    fn recovers_fenced_call_containing_a_json_comment() {
        let text = r#"```json
{
  "name": "self_patch",
  "arguments": {
    "action": "propose",
    "changes": [
      {
        "file": "src/tools/selfpatch.rs",
        "line_edits": [
          {
            "at_line": 123, // Assuming the function starts at line 123
            "replace_with": "/// Computes a hash of a file.\n///\n/// A `Result` with the hash."
          }
        ]
      }
    ]
  }
}
```"#;
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1, "a comment must not lose the call");
        assert_eq!(calls[0].function.name, "self_patch");
        let le = &calls[0].function.arguments["changes"][0]["line_edits"][0];
        assert_eq!(le["at_line"], 123);
        assert!(le["replace_with"].as_str().unwrap().contains("Computes a hash"));
    }

    #[test]
    fn recovers_fenced_call_with_trailing_comma() {
        let text = r#"```json
{"name":"self_patch","arguments":{"action":"files",},}
```"#;
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments["action"], "files");
    }

    /// A `//` inside a string is data, not a comment. Stripping it would
    /// corrupt a real argument (a URL, a path), so this must survive intact.
    #[test]
    fn comment_stripping_preserves_slashes_inside_strings() {
        let text = r#"```json
{"name":"self_patch","arguments":{"action":"read","file":"https://example.com/a//b.rs"}}
```"#;
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].function.arguments["file"],
            "https://example.com/a//b.rs",
            "URL must not be truncated at the //"
        );
    }

    /// Escaped quotes inside a string must not be read as string boundaries.
    #[test]
    fn comment_stripping_handles_escaped_quotes() {
        let text = r#"```json
{"name":"self_patch","arguments":{"action":"propose","note":"he said \"hi\" // not a comment"}}
```"#;
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].function.arguments["note"]
            .as_str()
            .unwrap()
            .contains("not a comment"));
    }

    /// Genuinely broken JSON must still fail rather than be half-parsed.
    #[test]
    fn unrepairable_json_still_fails() {
        assert!(parse_lenient_json("{ this is not valid json }").is_err());
        assert!(parse_lenient_json("{\"a\": }").is_err());
    }

    /// Regression guard: stripping trailing commas must not eat ordinary
    /// separating commas. An earlier version dropped the comma before any
    /// non-closer and would have turned {"a":1,"b":2} into {"a":1"b":2}.
    #[test]
    fn comma_stripping_keeps_separating_commas() {
        let v = parse_lenient_json(r#"{"a":1,"b":2,"c":[1,2]}"#).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], 2);
        assert_eq!(v["c"][1], 2);
    }

    #[test]
    fn trailing_comma_with_whitespace_before_closer_is_removed() {
        let v = parse_lenient_json("{\"a\": [1, 2, ],\n  \"b\": 3,\n}").unwrap();
        assert_eq!(v["a"][1], 2);
        assert_eq!(v["b"], 3);
    }

    /// A comma inside a string is content, not structure.
    #[test]
    fn comma_stripping_preserves_commas_inside_strings() {
        let v = parse_lenient_json(r#"{"note":"a, b, }c","x":1}"#).unwrap();
        assert_eq!(v["note"], "a, b, }c");
    }

    /// Already-valid JSON must pass through the fast path untouched.
    #[test]
    fn valid_json_is_not_mangled_by_the_repair() {
        let v = parse_lenient_json(r#"{"a":[1,2,{"b":"x,y"}],"c":"}"}"#).unwrap();
        assert_eq!(v["a"][2]["b"], "x,y");
        assert_eq!(v["c"], "}");
    }

    #[test]
    fn recovers_marker_call_with_multiline_and_quoted_args() {
        let text = "<|tool_call|>self_patch {\"action\":\"propose\",\"changes\":[{\"file\":\
                    \"src/util.rs\",\"edits\":[{\"find\":\"    #[test]\\n    fn t() {\",\
                    \"replace\":\"    }\"}]}]}<|/tool_call|>";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        let f = &calls[0].function.arguments["changes"][0]["edits"][0]["find"];
        assert_eq!(f.as_str().unwrap(), "    #[test]\n    fn t() {");
    }

    #[test]
    fn recovers_qwen3_tool_calls_array() {
        let text = "[TOOL_CALLS][{\"name\":\"self_patch\",\"arguments\":{\"action\":\"files\"}}]\
                    [/TOOL_CALLS]";
        let calls = extract_marked_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "self_patch");
        assert_eq!(calls[0].function.arguments["action"], "files");
    }

    /// Prose that merely mentions a marker must not become a tool call, and
    /// ordinary replies must yield nothing at all.
    #[test]
    fn marker_extraction_ignores_prose_and_junk() {
        for text in [
            "",
            "hey luna, what are you?",
            "here is how you'd call it: <|tool_call|> in prose",
            "<|tool_call|></|/tool_call|>",
            "<|tool_call|>no_json_here<|/tool_call|>",
        ] {
            assert!(
                extract_marked_tool_calls(text).is_empty(),
                "should not have extracted a call from {text:?}"
            );
        }
    }

    #[test]
    fn envelope_schema_stays_simple_enough_for_ollama() {
        let s = tool_envelope_schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["tool"]["type"] == "string");
        assert!(s["properties"]["arguments"]["type"] == "object");
        let ser = s.to_string();
        for bad in ["oneOf", "anyOf", "allOf", "$ref"] {
            assert!(!ser.contains(bad), "schema must not use {bad}");
        }
    }

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
