//! memory/recall.rs — Semantic memory recall (RAG-lite)
//!
//! Instead of dumping every permanent fact into each prompt, facts are
//! embedded once via Ollama's /api/embed (nomic-embed-text by default),
//! cached on disk keyed by content hash, and only the top-k most similar
//! to the current user query are injected.
//!
//! Graceful degradation: if Ollama or the embedding model is unavailable,
//! callers fall back to the old full-dump prompt block.

use crate::memory::permanent::Fact;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct EmbedCache(HashMap<String, Vec<f32>>);

fn cache_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
        .join("fact_embeddings.json")
}

fn content_key(content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Return the top-k facts most relevant to `query`, or None when semantic
/// recall is unavailable (caller should fall back to the full block).
pub async fn relevant_facts<'a>(
    base_url: &str,
    embedding_model: &str,
    query: &str,
    facts: &'a [Fact],
    k: usize,
) -> Option<Vec<&'a Fact>> {
    if facts.is_empty() || query.trim().is_empty() {
        return None;
    }

    let client = crate::llm::ollama::OllamaClient::new(base_url, "unused", 0.0, 1);
    let mut cache: EmbedCache = std::fs::read_to_string(cache_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    // Batch-embed query + any uncached facts in one request
    let missing: Vec<String> = facts
        .iter()
        .filter(|f| !cache.0.contains_key(&content_key(&f.content)))
        .map(|f| f.content.clone())
        .collect();

    let mut inputs = vec![query.to_string()];
    inputs.extend(missing.clone());

    let vectors = client
        .embed(embedding_model, &inputs)
        .await
        .inspect_err(|e| tracing::debug!("Recall disabled this turn: {}", e))
        .ok()?;
    if vectors.len() != inputs.len() {
        return None;
    }

    let query_vec = &vectors[0];
    for (content, vec) in missing.iter().zip(vectors[1..].iter()) {
        cache.0.insert(content_key(content), vec.clone());
    }

    // Persist new entries; prune hashes no longer backed by a fact
    let live: std::collections::HashSet<String> =
        facts.iter().map(|f| content_key(&f.content)).collect();
    cache.0.retain(|k, _| live.contains(k));
    if let Ok(json) = serde_json::to_string(&cache) {
        let _ = std::fs::write(cache_path(), json);
    }

    let mut scored: Vec<(f32, &Fact)> = facts
        .iter()
        .filter_map(|f| {
            cache
                .0
                .get(&content_key(&f.content))
                .map(|v| (cosine(query_vec, v), f))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    Some(scored.into_iter().take(k).map(|(_, f)| f).collect())
}

/// Format recalled facts as a prompt block.
pub fn format_block(facts: &[&Fact]) -> String {
    if facts.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n[Relevant things Luna knows]\n");
    for f in facts {
        out.push_str(&format!("- {}\n", f.content));
    }
    out
}
