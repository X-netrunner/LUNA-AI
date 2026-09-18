//! Shared types for the browser automation engine.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ActionResult {
    pub success: bool,
    pub action: String,
    pub step_index: u64,
    pub tab_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ActionResult {
    pub fn fail(action: String, step_index: u64, tab_id: u64, error: String) -> Self {
        Self { success: false, action, step_index, tab_id, error: Some(error) }
    }
    pub fn ok(action: String, step_index: u64, tab_id: u64) -> Self {
        Self { success: true, action, step_index, tab_id, error: None }
    }
}