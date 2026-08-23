//! memory/permanent.rs — Long-term facts Luna remembers across sessions
//!
//! Stored as a simple JSON array at ~/.local/share/luna/permanent_memory.json
//! Injected into every system prompt so Luna always has this context.
//! Luna can add/remove facts via the remember/forget tools.

use anyhow::{Context, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub content: String,
    pub added: String,    // ISO date string
    pub category: String, // "user", "system", "preference", "general"
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PermanentMemory {
    facts: Vec<Fact>,
    #[serde(skip)]
    path: PathBuf,
}

impl PermanentMemory {
    /// Load from disk, creating the file if it doesn't exist
    pub fn load() -> Result<Self> {
        let path = Self::storage_path();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create memory directory")?;
        }

        let facts = if path.exists() {
            let raw = std::fs::read_to_string(&path).context("Failed to read permanent memory")?;
            serde_json::from_str::<Vec<Fact>>(&raw).unwrap_or_else(|_| {
                tracing::warn!("Permanent memory file corrupt, starting fresh");
                Vec::new()
            })
        } else {
            // Seed with basic facts on first run
            vec![
                Fact {
                    content: "User runs Arch Linux with fish shell".into(),
                    added: Local::now().format("%Y-%m-%d").to_string(),
                    category: "system".into(),
                },
                Fact {
                    content: "User's default browser is zen-browser".into(),
                    added: Local::now().format("%Y-%m-%d").to_string(),
                    category: "system".into(),
                },
                Fact {
                    content: "User's default terminal is kitty".into(),
                    added: Local::now().format("%Y-%m-%d").to_string(),
                    category: "system".into(),
                },
            ]
        };

        let mem = PermanentMemory {
            facts,
            path: path.clone(),
        };
        mem.save()?;
        tracing::info!("Permanent memory loaded: {} facts", mem.facts.len());
        Ok(mem)
    }

    /// Add a new fact — deduplicates by content similarity
    pub fn remember(&mut self, content: &str, category: &str) -> Result<String> {
        let content = content.trim().to_string();
        if content.is_empty() {
            anyhow::bail!("Cannot remember empty fact");
        }

        // Near-duplicate (same first 30 chars) — update it in place
        let prefix = crate::util::truncate(&content, 30);
        if let Some(existing) = self.facts.iter_mut().find(|f| f.content.starts_with(prefix)) {
            existing.content = content.clone();
            existing.added = Local::now().format("%Y-%m-%d").to_string();
            self.save()?;
            return Ok(format!("Updated memory: {}", content));
        }

        // Cap at 100 facts — drop oldest if over limit
        if self.facts.len() >= 100 {
            self.facts.remove(0);
        }

        self.facts.push(Fact {
            content: content.clone(),
            added: Local::now().format("%Y-%m-%d").to_string(),
            category: category.to_string(),
        });

        self.save()?;
        tracing::info!("Remembered: {}", content);
        Ok(format!("Remembered: {}", content))
    }

    /// Remove a fact by keyword match
    pub fn forget(&mut self, keyword: &str) -> Result<String> {
        let keyword = keyword.to_lowercase();
        let before = self.facts.len();
        self.facts
            .retain(|f| !f.content.to_lowercase().contains(&keyword));
        let removed = before - self.facts.len();

        if removed == 0 {
            return Ok(format!("No facts found matching '{}'", keyword));
        }

        self.save()?;
        tracing::info!("Forgot {} fact(s) matching '{}'", removed, keyword);
        Ok(format!("Forgot {} fact(s) about '{}'", removed, keyword))
    }

    /// List all facts as a formatted string
    /// Direct access to all facts (for semantic recall)
    pub fn all_facts(&self) -> &[Fact] {
        &self.facts
    }

    pub fn list(&self) -> String {        if self.facts.is_empty() {
            return "No permanent memories stored.".to_string();
        }
        self.facts
            .iter()
            .map(|f| format!("- [{}] {}", f.category, f.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Build the prompt injection block
    /// This goes at the top of every system prompt
    pub fn as_prompt_block(&self) -> String {
        if self.facts.is_empty() {
            return String::new();
        }

        let facts_text = self
            .facts
            .iter()
            .map(|f| format!("- {}", f.content))
            .collect::<Vec<_>>()
            .join("\n");

        format!("[What Luna knows about the user]\n{}\n", facts_text)
    }

    fn save(&self) -> Result<()> {        let json = serde_json::to_string_pretty(&self.facts)
            .context("Failed to serialize permanent memory")?;
        std::fs::write(&self.path, json).context("Failed to write permanent memory")?;
        Ok(())
    }

    fn storage_path() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna")
            .join("permanent_memory.json")
    }
}
