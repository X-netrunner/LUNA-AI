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
        Self::load_path(Self::storage_path())
    }

    fn load_path(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create memory directory")?;
        }

        let facts = match std::fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str::<Vec<Fact>>(&raw)
                .context("Permanent memory is corrupt; original file preserved")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e).context("Failed to read permanent memory"),
        };

        let mem = PermanentMemory {
            facts,
            path: path.clone(),
        };
        tracing::info!("Permanent memory loaded: {} facts", mem.facts.len());
        Ok(mem)
    }

    /// Add a new fact — deduplicates by content similarity
    pub fn remember(&mut self, content: &str, category: &str) -> Result<String> {
        let content = content.trim().to_string();
        if content.is_empty() {
            anyhow::bail!("Cannot remember empty fact");
        }

        if !["user", "system", "preference", "general"].contains(&category) {
            anyhow::bail!("Unknown memory category: {category}");
        }
        if content.chars().count() > 1000 || content.chars().any(char::is_control) {
            anyhow::bail!(
                "Memory must be one sentence without control characters, at most 1000 characters"
            );
        }
        let before = self.facts.clone();
        let normalized = |s: &str| {
            s.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase()
        };
        if let Some(existing) = self
            .facts
            .iter_mut()
            .find(|f| f.category == category && normalized(&f.content) == normalized(&content))
        {
            existing.content = content.clone();
            existing.added = Local::now().format("%Y-%m-%d").to_string();
            self.commit(before)?;
            return Ok(format!("Updated memory: {}", content));
        }

        if self.facts.len() >= 100 {
            anyhow::bail!("Memory is full; forget an outdated fact before adding another");
        }

        self.facts.push(Fact {
            content: content.clone(),
            added: Local::now().format("%Y-%m-%d").to_string(),
            category: category.to_string(),
        });

        self.commit(before)?;
        tracing::info!("Remembered: {}", content);
        Ok(format!("Remembered: {}", content))
    }

    /// Remove a fact by keyword match
    pub fn forget(&mut self, keyword: &str) -> Result<String> {
        let keyword = keyword.trim().to_lowercase();
        if keyword.is_empty() {
            anyhow::bail!("Specify a nonempty keyword to forget");
        }
        let previous = self.facts.clone();
        let before = self.facts.len();
        self.facts
            .retain(|f| !f.content.to_lowercase().contains(&keyword));
        let removed = before - self.facts.len();

        if removed == 0 {
            return Ok(format!("No facts found matching '{}'", keyword));
        }

        self.commit(previous)?;
        tracing::info!("Forgot {} fact(s) matching '{}'", removed, keyword);
        Ok(format!("Forgot {} fact(s) about '{}'", removed, keyword))
    }

    /// List all facts as a formatted string
    /// Direct access to all facts (for semantic recall)
    pub fn all_facts(&self) -> &[Fact] {
        &self.facts
    }

    pub fn list(&self) -> String {
        if self.facts.is_empty() {
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

    fn commit(&mut self, previous: Vec<Fact>) -> Result<()> {
        if let Err(error) = self.save() {
            self.facts = previous;
            return Err(error);
        }
        Ok(())
    }

    fn save(&self) -> Result<()> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let json = serde_json::to_vec_pretty(&self.facts)?;
        let tmp = self
            .path
            .with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
            std::fs::rename(&tmp, &self.path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result.context("Failed to save permanent memory")
    }

    fn storage_path() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna")
            .join("permanent_memory.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memories_do_not_clobber_shared_prefixes_or_accept_empty_forget() {
        let dir = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("memory.json");
        assert!(PermanentMemory::load_path(path.clone())
            .unwrap()
            .all_facts()
            .is_empty());
        assert!(!path.exists(), "loading must not rewrite memory");
        std::fs::write(&path, "broken").unwrap();
        assert!(PermanentMemory::load_path(path.clone()).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "broken");
        std::fs::remove_file(&path).unwrap();
        let mut memory = PermanentMemory {
            facts: vec![],
            path,
        };
        memory
            .remember(
                "User prefers the following tools for coding: Rust",
                "preference",
            )
            .unwrap();
        memory
            .remember(
                "User prefers the following tools for coding: Python",
                "preference",
            )
            .unwrap();
        assert_eq!(memory.all_facts().len(), 2);
        assert!(memory.forget(" ").is_err());
        assert_eq!(memory.all_facts().len(), 2);
        assert!(memory.remember("abc", "invalid").is_err());
        let original = memory.path.clone();
        memory.path = dir.join("missing").join("memory.json");
        assert!(memory
            .remember("User likes short answers", "preference")
            .is_err());
        assert_eq!(memory.all_facts().len(), 2);
        assert_eq!(
            serde_json::from_str::<Vec<Fact>>(&std::fs::read_to_string(original).unwrap())
                .unwrap()
                .len(),
            2
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
