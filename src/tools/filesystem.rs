use anyhow::{Context, Result};
use std::path::Path;

pub async fn read_file(path: &str) -> Result<String> {
    tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read: {}", path))
}

pub async fn write_file(path: &str, content: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create parent dirs")?;
    }
    tokio::fs::write(path, content)
        .await
        .with_context(|| format!("Failed to write: {}", path))
}
