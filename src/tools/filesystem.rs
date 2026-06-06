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

pub async fn list_dir(path: &str) -> Result<Vec<String>> {
    let mut entries = tokio::fs::read_dir(path)
        .await
        .with_context(|| format!("Failed to read dir: {}", path))?;
    let mut files = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
        files.push(if is_dir { format!("{}/", name) } else { name });
    }
    files.sort();
    Ok(files)
}
