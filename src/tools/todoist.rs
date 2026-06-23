//! tools/todoist.rs — Todoist integration via the unified Todoist API v1
//!
//! Todoist shut down REST API v2 / Sync API v9 in late 2025.
//! Everything now lives under https://api.todoist.com/api/v1/
//! List endpoints are now paginated: { "results": [...], "next_cursor": ... }
//!
//! Get your token at: https://todoist.com/app/settings/integrations

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

const BASE_URL: &str = "https://api.todoist.com/api/v1";

#[derive(Debug, Deserialize)]
struct TodoistTask {
    #[serde(default)]
    id: String,
    content: String,
    #[serde(default)]
    due: Option<TodoistDue>,
    #[serde(default)]
    priority: i32,
}

#[derive(Debug, Deserialize)]
struct TodoistDue {
    #[serde(default)]
    string: String,
}

pub async fn list_tasks(api_token: &str, filter: Option<&str>) -> Result<String> {
    let client = Client::new();
    let mut url = format!("{}/tasks", BASE_URL);
    if let Some(f) = filter {
        url = format!("{}?filter={}", url, urlencoding(f));
    }

    let resp = client
        .get(&url)
        .bearer_auth(api_token)
        .send()
        .await
        .context("Failed to reach Todoist API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Todoist returned {}: {}", status, body);
    }

    let raw: Value = resp
        .json()
        .await
        .context("Failed to parse Todoist response")?;

    let tasks: Vec<TodoistTask> = if let Some(arr) = raw.get("results") {
        serde_json::from_value(arr.clone()).context("Failed to parse task list")?
    } else {
        serde_json::from_value(raw).context("Failed to parse task list")?
    };

    if tasks.is_empty() {
        return Ok("No tasks found in Todoist.".to_string());
    }

    let lines: Vec<String> = tasks
        .iter()
        .map(|t| {
            let due = t
                .due
                .as_ref()
                .map(|d| format!(" (due: {})", d.string))
                .unwrap_or_default();
            let priority = if t.priority > 1 {
                format!(" [P{}]", 5 - t.priority)
            } else {
                String::new()
            };
            format!("- {}{}{}", t.content, due, priority)
        })
        .collect();

    Ok(lines.join("\n"))
}

pub async fn add_task(api_token: &str, content: &str, due_string: Option<&str>) -> Result<String> {
    let client = Client::new();
    let mut body = serde_json::json!({ "content": content });
    if let Some(due) = due_string {
        body["due_string"] = serde_json::json!(due);
    }

    let resp = client
        .post(format!("{}/tasks", BASE_URL))
        .bearer_auth(api_token)
        .json(&body)
        .send()
        .await
        .context("Failed to reach Todoist API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Todoist returned {}: {}", status, body);
    }

    let task: TodoistTask = resp.json().await.context("Failed to parse response")?;
    Ok(format!("Added to Todoist: {}", task.content))
}

pub async fn complete_task(api_token: &str, content_match: &str) -> Result<String> {
    let client = Client::new();

    let resp = client
        .get(format!("{}/tasks", BASE_URL))
        .bearer_auth(api_token)
        .send()
        .await
        .context("Failed to reach Todoist API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Todoist returned {}: {}", status, body);
    }

    let raw: Value = resp.json().await.context("Failed to parse tasks")?;
    let tasks: Vec<TodoistTask> = if let Some(arr) = raw.get("results") {
        serde_json::from_value(arr.clone()).context("Failed to parse task list")?
    } else {
        serde_json::from_value(raw).context("Failed to parse task list")?
    };

    let matched = tasks.iter().find(|t| {
        t.content
            .to_lowercase()
            .contains(&content_match.to_lowercase())
    });

    let task = match matched {
        Some(t) => t,
        None => {
            return Ok(format!(
                "No Todoist task found matching '{}'",
                content_match
            ))
        }
    };

    let close_resp = client
        .post(format!("{}/tasks/{}/close", BASE_URL, task.id))
        .bearer_auth(api_token)
        .send()
        .await
        .context("Failed to close task")?;

    if close_resp.status().is_success() {
        Ok(format!("Completed: {}", task.content))
    } else {
        let status = close_resp.status();
        let body = close_resp.text().await.unwrap_or_default();
        anyhow::bail!("Failed to complete task ({}): {}", status, body)
    }
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '&' => "%26".to_string(),
            '|' => "%7C".to_string(),
            c => c.to_string(),
        })
        .collect()
}
