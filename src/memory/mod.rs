//! memory/mod.rs — Conversation history and context management

pub mod permanent;

use crate::llm::ollama::Message;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct Memory {
    messages: Vec<Message>,
    window_size: usize,
    history_path: PathBuf,
}

impl Memory {
    pub fn new(window_size: usize, history_path: &Path) -> Result<Self> {
        if let Some(parent) = history_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create memory directory")?;
        }

        let messages = if history_path.exists() {
            let raw =
                std::fs::read_to_string(history_path).context("Failed to read history file")?;
            serde_json::from_str::<Vec<Message>>(&raw).unwrap_or_else(|_| {
                tracing::warn!("History file corrupt, starting fresh");
                Vec::new()
            })
        } else {
            Vec::new()
        };

        tracing::info!("Memory loaded: {} previous messages", messages.len());
        Ok(Self {
            messages,
            window_size,
            history_path: history_path.to_path_buf(),
        })
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
        while self.messages.len() > self.window_size {
            self.messages.remove(0);
        }
    }

    /// Build context for LLM — filters out tool call artifacts that
    /// confuse the model into echoing them on the next turn.
    pub fn build_context(&self, system_prompt: &str) -> Vec<Message> {
        let mut context = vec![Message::system(system_prompt)];

        let clean: Vec<Message> = self
            .messages
            .iter()
            .filter(|m| {
                let c = m.content.trim();
                // Drop assistant messages that are just tool call echoes
                !(m.role == "assistant"
                    && (c.starts_with("[tool_call:") ||
                    c.starts_with("Called tool:") ||
                    c.starts_with("<tool_use>") ||
                    c.starts_with("<|tool_call|>") ||
                    c.contains("<brtc>") ||
                    c.starts_with("(run_shell") ||
                    c.starts_with("Opening `") ||
                    // Single-word or punctuation-only responses are artifacts
                    (c.len() < 3 && !c.chars().any(|ch| ch.is_alphabetic()))))
            })
            .cloned()
            .collect();

        context.extend(clean);
        context
    }

    pub fn save(&self) -> Result<()> {
        let json =
            serde_json::to_string_pretty(&self.messages).context("Failed to serialize history")?;
        std::fs::write(&self.history_path, json).context("Failed to write history to disk")?;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.messages.clear();
        self.save()?;
        tracing::info!("Memory cleared");
        Ok(())
    }
}
