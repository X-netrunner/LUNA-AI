//! LLM-based browser planner — calls Ollama to break a user goal into steps.

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A single planner step in our internal format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub action: String,           // "navigate" | "click" | "type" | "press" | "scroll" | "search"
    pub target: Option<String>,   // URL for navigate, selector/text for click/type, key for press, direction for scroll, query for search
    pub value: Option<String>,    // Additional value (e.g., text to type)
    pub is_last: bool,
}

impl Step {
    pub fn navigate(url: impl Into<String>) -> Self {
        Self {
            action: "navigate".into(),
            target: Some(url.into()),
            value: None,
            is_last: false,
        }
    }
    pub fn click(target: impl Into<String>) -> Self {
        Self {
            action: "click".into(),
            target: Some(target.into()),
            value: None,
            is_last: false,
        }
    }
    pub fn type_text(text: impl Into<String>) -> Self {
        Self {
            action: "type".into(),
            target: None,
            value: Some(text.into()),
            is_last: false,
        }
    }
    pub fn press(key: impl Into<String>) -> Self {
        Self {
            action: "press".into(),
            target: Some(key.into()),
            value: None,
            is_last: false,
        }
    }
    pub fn scroll(direction: impl Into<String>, amount: i64) -> Self {
        Self {
            action: "scroll".into(),
            target: Some(direction.into()),
            value: Some(amount.to_string()),
            is_last: false,
        }
    }
    pub fn search(query: impl Into<String>) -> Self {
        Self {
            action: "search".into(),
            target: Some(query.into()),
            value: None,
            is_last: false,
        }
    }
}

/// Planner that uses an LLM (via Ollama) to generate step sequences.
pub struct Planner {
    client: Client,
    ollama_url: String,
    model: String,
}

impl Planner {
    pub fn new(ollama_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base = ollama_url.into();
        // Ensure it points to /api/generate endpoint
        let api_url = if base.ends_with("/api/generate") {
            base
        } else {
            base.trim_end_matches('/').to_string() + "/api/generate"
        };
        Self {
            client: Client::new(),
            ollama_url: api_url,
            model: model.into(),
        }
    }

    /// Generate a plan for the given goal.
    pub async fn plan(&self, goal: &str) -> Result<Vec<Step>> {
        let prompt = self.build_prompt(goal);
        let response = self.call_ollama(&prompt).await?;
        self.parse_plan(&response)
    }

    /// Re-plan after a failure, given context.
    pub async fn replan(
        &self,
        goal: &str,
        completed: &[Step],
        failed_step: &Step,
        error: &str,
        remaining: &[Step],
    ) -> Result<Vec<Step>> {
        let prompt = self.build_replan_prompt(goal, completed, failed_step, error, remaining);
        let response = self.call_ollama(&prompt).await?;
        self.parse_plan(&response)
    }

    fn build_prompt(&self, goal: &str) -> String {
        format!(
            r#"You are an expert browser automation planner. Break the user's goal into minimal, correct steps.

CRITICAL RULES:
1. NEVER include browser startup steps (no "Open browser", "Launch Chrome").
2. Analyze the user's ACTUAL intent. If they say "search for the history of ipads", they want a SEARCH RESULTS page, NOT an e-commerce flow. Do NOT add "Add to Cart" steps unless the user explicitly mentions purchasing or adding to cart.
3. If the user says "in flipkart add this to cart" or "add to cart", they are already on the product page and want to add the CURRENTLY VIEWED item. Steps: just scroll down and click Add to Cart. Do NOT search again.
4. E-COMMERCE ADD-TO-CART: the Search Results page grid does NOT reliably show an "Add to Cart" button. The correct steps for "add [X] to cart": navigate: <site>, click: search bar, type: X, press: Enter, click: the first product result (then wait for the product detail page), then click: Add to Cart. NEVER emit a bare "click: Add to Cart" step while still on a search-results grid — always click the product card FIRST to open its detail page. If the user says "first result", the product click should read EXACTLY "click: the first product result".
5. For search queries on Google/Bing ("search for X", "look up X", "find X"), the steps are: navigate to google.com, click search bar, type query, press Enter. Then stop - do NOT click specific results unless the user asked to.
6. Screenshots have faces and PII redacted with blur/black boxes. This is NORMAL privacy protection. Do NOT treat redactions as errors. Plan steps that interact with standard UI elements only.
7. URL PRESERVATION: If the user provides a full URL in their request (e.g. a https://docs.google.com/forms/d/... link), the FIRST action MUST be "navigate: <the EXACT full URL>" with the ENTIRE URL kept intact - do NOT trim it to just the domain. Only shorten to a bare domain when the user gave a domain name like "amazon.com" or "google.com".
8. FORM FILLING: If the user asks to fill a form or provides a form URL, DO NOT produce placeholder steps like "type: [user input]" or "click: Form title". Instead just output a SINGLE step "navigate: <full form url>" (or none if already there). The server handles the actual per-field filling automatically, so your job is ONLY to get to the form page. Do not invent field/type steps for forms.

SUPPORTED ACTION TAGS:
- navigate: [URL or Domain] - only if user needs to go to a new site
- click: [Target element description] - visual element on the page
- type: [Text to type] - into the currently focused input
- press: [Key like Enter/Tab/Escape]
- scroll: [down/up]
- search: [query] - navigate to search engine + type + enter

USER GOAL: "{goal}"

Output ONLY a JSON object:
{{"steps": ["step1", "step2", ...]}}"#
        )
    }

    fn build_replan_prompt(
        &self,
        goal: &str,
        completed: &[Step],
        failed_step: &Step,
        error: &str,
        remaining: &[Step],
    ) -> String {
        format!(
            r#"You are an expert browser automation planner recovering from a failure.

USER GOAL: "{goal}"
COMPLETED STEPS: {}
FAILED STEP: "{}" ({})
ERROR / VISUAL CONTEXT: "{}"
REMAINING STEPS: {}

RETHINKING RULES:
1. DO NOT repeat the exact same failed step without a corrective action first.
2. DO NOT restart the entire flow if previous steps already succeeded.
3. Common failure patterns and fixes:
   - Element not found in viewport -> scroll down first, then retry click
   - Popup/modal/overlay blocking -> press Escape or click close, then continue
   - Search bar not found -> click on the search area first
   - Page not loaded yet -> press Enter or wait, then retry
   - Wrong page/tab -> switch_tab, then continue
   - Product variant needed (size/color) -> click variant option first, then Add to Cart
   - Login redirect -> press Escape to dismiss login popup, continue on main page
   - "Add to Cart" not found while on a SEARCH RESULTS grid -> the product is not open yet:
     first emit "click: the first product result" to open its detail page, THEN "click: Add to Cart".
     Prefer this over endless scroll-down retries.
4. Keep corrective steps minimal (1-3 steps). Don't overcomplicate.

Output ONLY a JSON object:
{{"thought": "Brief explanation", "steps": ["corrective_step1", "corrective_step2"]}}"#,
            serde_json::to_string(completed).unwrap_or_default(),
            serde_json::to_string(failed_step).unwrap_or_default(),
            failed_step.action,
            error,
            serde_json::to_string(remaining).unwrap_or_default(),
        )
    }

    async fn call_ollama(&self, prompt: &str) -> Result<String> {
        let resp = self
            .client
            .post(&self.ollama_url)
            .json(&json!({
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": false,
                "options": { "temperature": 0.1, "num_predict": 512 }
            }))
            .send()
            .await
            .context("call ollama")?;

        let body: Value = resp.json().await.context("parse ollama response")?;
        body["response"]
            .as_str()
            .ok_or_else(|| anyhow!("no response from ollama"))
            .map(|s| s.trim().to_string())
    }

    fn parse_plan(&self, response: &str) -> Result<Vec<Step>> {
        let cleaned = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: Value = serde_json::from_str(cleaned).context("parse plan JSON")?;

        let steps_arr = parsed
            .get("steps")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("no steps array in plan"))?;

        let mut steps = Vec::new();
        for (i, step_val) in steps_arr.iter().enumerate() {
            let step_str = step_val.as_str().ok_or_else(|| anyhow!("step not a string"))?;
            let step = self.parse_step(step_str)?;
            // Mark the last step
            let mut step = step;
            step.is_last = i == steps_arr.len() - 1;
            steps.push(step);
        }
        Ok(steps)
    }

    fn parse_step(&self, step: &str) -> Result<Step> {
        let step = step.trim();

        if let Some(rest) = step.strip_prefix("navigate:").or_else(|| step.strip_prefix("navigate ")) {
            return Ok(Step::navigate(rest.trim()));
        }
        if let Some(rest) = step.strip_prefix("click:").or_else(|| step.strip_prefix("click ")) {
            return Ok(Step::click(rest.trim()));
        }
        if let Some(rest) = step.strip_prefix("type:").or_else(|| step.strip_prefix("type ")) {
            return Ok(Step::type_text(rest.trim()));
        }
        if let Some(rest) = step.strip_prefix("press:").or_else(|| step.strip_prefix("press ")) {
            return Ok(Step::press(rest.trim()));
        }
        if let Some(rest) = step.strip_prefix("scroll:").or_else(|| step.strip_prefix("scroll ")) {
            let parts: Vec<&str> = rest.trim().split_whitespace().collect();
            let dir = parts.first().copied().unwrap_or("down");
            let amt = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
            return Ok(Step::scroll(dir, amt));
        }
        if let Some(rest) = step.strip_prefix("search:").or_else(|| step.strip_prefix("search ")) {
            return Ok(Step::search(rest.trim()));
        }

        Err(anyhow!("unknown step format: {step}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_step_variants() {
        let p = Planner::new("http://localhost", "test");
        assert_eq!(p.parse_step("navigate: https://amazon.com").unwrap().action, "navigate");
        assert_eq!(p.parse_step("navigate https://google.com").unwrap().action, "navigate");
        assert_eq!(p.parse_step("click: search bar").unwrap().action, "click");
        assert_eq!(p.parse_step("click search bar").unwrap().action, "click");
        assert_eq!(p.parse_step("type: hello world").unwrap().action, "type");
        assert_eq!(p.parse_step("press: Enter").unwrap().action, "press");
        assert_eq!(p.parse_step("scroll: down 300").unwrap().action, "scroll");
        assert_eq!(p.parse_step("search: ps5 controller").unwrap().action, "search");
    }
}