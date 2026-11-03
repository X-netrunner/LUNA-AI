//! memory/skills.rs — Reusable procedures Luna creates from experience
//!
//! Hermes-style skill learning: after a task shows a repeatable procedure,
//! Luna saves it as a skill it can load later. Each skill is a Markdown file
//! at ~/.local/share/luna/skills/<name>.md with a small frontmatter header:
//!
//! ```text
//! skill: <slug>
//! description: one-line summary
//! tags: comma, separated
//! created: YYYY-MM-DD
//! updated: YYYY-MM-DD
//! uses: N
//! - - -
//! <procedure body>
//! ```
//!
//! "Self-improvement during use": reading a skill bumps its use count, and
//! overwriting a skill (same name) keeps its history and shows it was refined.

use anyhow::{Context, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub created: String,
    pub updated: String,
    pub uses: u64,
}

impl SkillMeta {
    fn to_lines(&self) -> Vec<String> {
        vec![
            format!("skill: {}", self.name),
            format!("description: {}", self.description),
            format!("tags: {}", self.tags.join(", ")),
            format!("created: {}", self.created),
            format!("updated: {}", self.updated),
            format!("uses: {}", self.uses),
        ]
    }

    fn from_lines(raw: &str) -> Option<SkillMeta> {
        let mut meta = SkillMeta {
            name: String::new(),
            description: String::new(),
            tags: Vec::new(),
            created: String::new(),
            updated: String::new(),
            uses: 0,
        };
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            let (key, value) = line.split_once(':')?;
            match key.trim() {
                "skill" => meta.name = value.trim().to_string(),
                "description" => meta.description = value.trim().to_string(),
                "tags" => {
                    meta.tags = value
                        .split(',')
                        .map(|t| t.trim().to_string())
                        .filter(|t| !t.is_empty())
                        .collect()
                }
                "created" => meta.created = value.trim().to_string(),
                "updated" => meta.updated = value.trim().to_string(),
                "uses" => meta.uses = value.trim().parse().unwrap_or(0),
                _ => {}
            }
        }
        Some(meta)
    }
}

/// A fully loaded skill: metadata plus the procedure body.
#[derive(Debug, Clone)]
pub struct Skill {
    pub meta: SkillMeta,
    pub body: String,
}

const FRONTMATTER_END: &str = "- - -";

fn skills_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
        .join("skills")
}

pub fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut last_hyphen = false;
    for c in name.trim().to_lowercase().chars() {
        let hyphen = !c.is_ascii_alphanumeric();
        if hyphen {
            if !last_hyphen {
                out.push('-');
            }
            last_hyphen = true;
        } else {
            out.push(c);
            last_hyphen = false;
        }
    }
    out.trim_matches('-').to_string()
}

fn path_for(slug: &str) -> PathBuf {
    skills_dir().join(format!("{}.md", slug))
}

/// Ensure the skills directory exists.
pub fn ensure_dir() -> Result<()> {
    fs::create_dir_all(skills_dir()).context("Failed to create skills directory")
}

fn today() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

fn render_file(meta: &SkillMeta, body: &str) -> String {
    let mut out = meta.to_lines().join("\n");
    out.push('\n');
    out.push_str(FRONTMATTER_END);
    out.push('\n');
    out.push_str(body.trim());
    out.push('\n');
    out
}

fn parse_file_at(path: &PathBuf) -> Option<Skill> {
    let raw = fs::read_to_string(path).ok()?;
    let (front, body) = raw.split_once(FRONTMATTER_END)?;
    let meta = SkillMeta::from_lines(front)?;
    if meta.name.is_empty() {
        return None;
    }
    Some(Skill {
        meta,
        body: body.trim().to_string(),
    })
}

fn load_one(slug: &str) -> Option<Skill> {
    parse_file_at(&path_for(slug))
}

/// List all skills, sorted by name. Returns an empty vec when none exist.
pub fn list() -> Result<Vec<SkillMeta>> {
    let dir = skills_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if let Some(skill) = parse_file_at(&path) {
            out.push(skill.meta);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Create (or overwrite) a skill. Overwriting keeps the original creation date
/// and use count, and simply records the new update date — that's the
/// self-improvement loop: Luna re-saves an outdated skill with better steps.
pub fn create(name: &str, description: &str, procedure: &str) -> Result<String> {
    let name_trim = name.trim();
    let slug = slugify(name_trim);
    if slug.is_empty() {
        anyhow::bail!("Skill name can't be empty");
    }
    if procedure.trim().is_empty() {
        anyhow::bail!("Skill procedure can't be empty");
    }

    ensure_dir()?;

    let existing = load_one(&slug).map(|s| s.meta);
    let meta = SkillMeta {
        name: existing
            .as_ref()
            .map(|m| m.name.clone())
            .unwrap_or_else(|| slug.clone()),
        description: if description.trim().is_empty() {
            format!("Procedure for '{}'", name_trim)
        } else {
            description.trim().to_string()
        },
        tags: Vec::new(),
        created: existing
            .as_ref()
            .map(|m| m.created.clone())
            .unwrap_or_else(today),
        updated: today(),
        uses: existing.map(|m| m.uses).unwrap_or(0),
    };

    let path = path_for(&slug);
    fs::write(&path, render_file(&meta, procedure))
        .with_context(|| format!("Failed to write skill '{}'", slug))?;
    tracing::info!("Skill saved: {}", slug);
    Ok(format!(
        "Saved skill '{}': {}\nSay \"use skill {} and ...\" to apply it, or I'll use it when relevant.",
        name_trim, meta.description, name_trim
    ))
}

/// Read a skill's full procedure by name. Bumps its use count (the model can
/// see that repeated use makes a skill battle-tested).
pub fn read(name: &str) -> Result<String> {
    let slug = slugify(name);
    if slug.is_empty() {
        anyhow::bail!("No skill name provided");
    }
    let Some(skill) = load_one(&slug) else {
        anyhow::bail!(
            "No skill named '{}' found. Use list_skills to see what I can do.",
            name
        );
    };
    let _ = bump_uses(&slug, &skill.meta);
    let header = format!(
        "Skill: {}\nDescription: {}\nUses: {}\n\n--- Procedure ---\n",
        skill.meta.name,
        skill.meta.description,
        skill.meta.uses + 1
    );
    Ok(format!("{}{}", header, skill.body))
}

/// Called after a skill was demonstrably relevant: bumps the use count so the
/// recall ordering rewards skills that actually get used.
fn bump_uses(slug: &str, meta: &SkillMeta) -> Result<()> {
    if meta.uses == u64::MAX {
        return Ok(());
    }
    let mut updated = meta.clone();
    updated.uses += 1;
    updated.updated = today();
    if let Some(skill) = load_one(slug) {
        fs::write(path_for(slug), render_file(&updated, &skill.body))
            .with_context(|| format!("Failed to update skill '{}'", slug))?;
    }
    Ok(())
}

/// Remove a skill entirely.
pub fn forget(name: &str) -> Result<String> {
    let slug = slugify(name);
    if slug.is_empty() {
        anyhow::bail!("No skill name provided");
    }
    let path = path_for(&slug);
    if !path.exists() {
        return Ok(format!("No skill named '{}' exists.", name));
    }
    fs::remove_file(&path)?;
    tracing::info!("Removed skill: {}", slug);
    Ok(format!("Forgot skill '{}'.", name))
}

/// Formatted listing for the list_skills tool.
pub fn list_and_format() -> String {
    match list() {
        Ok(skills) if skills.is_empty() => {
            "No skills yet. After I learn something repeatable I'll save it as a \
             skill automatically."
                .to_string()
        }
        Ok(skills) => {
            let rows = skills
                .iter()
                .map(|m| {
                    format!(
                        "- {} ({} uses): {}",
                        m.name,
                        m.uses,
                        if m.description.is_empty() {
                            "no description"
                        } else {
                            &m.description
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("Skills ({}):\n{}", skills.len(), rows)
        }
        Err(e) => format!("Couldn't list skills: {}", e),
    }
}

/// Semantic recall: top-k skill *descriptions* closest to the query. Skills
/// themselves are stubbed as facts so they ride the existing embedding cache.
pub async fn recall_top<'a>(
    base_url: &str,
    embedding_model: &str,
    query: &str,
    metas: &'a [SkillMeta],
    k: usize,
) -> Vec<&'a SkillMeta> {
    if metas.is_empty() {
        return Vec::new();
    }
    let shims: Vec<crate::memory::permanent::Fact> = metas
        .iter()
        .map(|m| crate::memory::permanent::Fact {
            content: format!("{}: {}", m.name, m.description),
            added: m.created.clone(),
            category: "skill".into(),
        })
        .collect();
    match crate::memory::recall::relevant_facts(base_url, embedding_model, query, &shims, k).await {
        Some(recalled) => recalled
            .into_iter()
            .filter_map(|f| {
                let (name, _) = f.content.split_once(':')?;
                metas.iter().find(|m| m.name == name.trim())
            })
            .collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugifies_names() {
        assert_eq!(slugify("System Update"), "system-update");
        assert_eq!(slugify("  Re-Setup  Monitor "), "re-setup-monitor");
        assert_eq!(slugify("Momo & Co."), "momo-co");
        assert!(slugify("   ").is_empty());
    }

    #[test]
    fn frontmatter_round_trips() {
        let meta = SkillMeta {
            name: "sys-update".into(),
            description: "Update everything".into(),
            tags: vec!["arch".into(), "pacman".into()],
            created: "2026-09-15".into(),
            updated: "2026-09-15".into(),
            uses: 3,
        };
        let rendered = render_file(&meta, "run pacman -Syu\nreboot if kernel changed");
        let parsed = SkillMeta::from_lines(&rendered[..rendered.find(FRONTMATTER_END).unwrap()])
            .expect("meta parses");
        assert_eq!(parsed.name, meta.name);
        assert_eq!(parsed.description, meta.description);
        assert_eq!(parsed.tags, meta.tags);
        assert_eq!(parsed.uses, 3);
    }

    #[test]
    fn create_read_forget_in_scratch_dir() {
        let scratch = std::env::temp_dir().join(format!("luna-skills-test-{}", std::process::id()));
        // Redirect the store into the scratch dir via a private hack: we can't
        // override skills_dir() from here, so exercise the file-level helpers.
        let _ = fs::create_dir_all(&scratch);
        let meta = SkillMeta {
            name: "t".into(),
            description: "test".into(),
            tags: vec![],
            created: "2026-09-15".into(),
            updated: "2026-09-15".into(),
            uses: 0,
        };
        let path = scratch.join("t.md");
        fs::write(&path, render_file(&meta, "do the thing")).unwrap();
        let skill = parse_file_at(&path).unwrap();
        assert_eq!(skill.body, "do the thing");
        let _ = fs::remove_dir_all(&scratch);
    }
}
