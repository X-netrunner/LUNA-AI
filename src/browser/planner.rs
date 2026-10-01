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
    pub fn fill_form(instructions: impl Into<String>) -> Self {
        Self {
            action: "fillform".into(),
            target: Some(instructions.into()),
            value: None,
            is_last: false,
        }
    }
    pub fn pick_best() -> Self {
        Self {
            action: "pick_best".into(),
            target: None,
            value: None,
            is_last: false,
        }
    }
    pub fn wait(ms: u64) -> Self {
        Self {
            action: "wait".into(),
            target: Some(ms.to_string()),
            value: None,
            is_last: false,
        }
    }
    pub fn extract(selector: impl Into<String>) -> Self {
        Self {
            action: "extract".into(),
            target: Some(selector.into()),
            value: None,
            is_last: false,
        }
    }
    pub fn back() -> Self {
        Self {
            action: "back".into(),
            target: None,
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
        let mut steps = self.parse_plan(&response)?;
        // Post-process for "best product" goals to ensure correct step order
        steps = Self::fix_best_product_plan(goal, steps);
        Ok(steps)
    }

    /// Fix plan for "best product" queries: ensure pick_best is used correctly.
    fn fix_best_product_plan(goal: &str, steps: Vec<Step>) -> Vec<Step> {
        let goal_lower = goal.to_lowercase();
        let is_best_product = goal_lower.contains("best product") || goal_lower.contains("best value");
        if !is_best_product {
            return steps;
        }
        // Find if pick_best exists
        let has_pick_best = steps.iter().any(|s| s.action == "pick_best");
        if !has_pick_best {
            return steps;
        }
        // Remove any "click: the first product result" or similar product-click steps before pick_best
        // pick_best should be the product selection step, immediately after search/navigation
        let mut fixed = Vec::new();
        let mut pick_best_seen = false;
        for step in steps {
            if step.action == "pick_best" {
                pick_best_seen = true;
                fixed.push(step);
            } else if step.action == "click" {
                let target = step.target.as_deref().unwrap_or("").to_lowercase();
                // Skip product selection clicks if we have pick_best
                if pick_best_seen || target.contains("first product") || target.contains("product result") || target.contains("product card") {
                    continue; // skip this step
                }
                fixed.push(step);
            } else {
                fixed.push(step);
            }
        }
        // Ensure pick_best is followed by "click: Add to Cart"
        // If the last step is pick_best, add Add to Cart
        if let Some(last) = fixed.last() {
            if last.action == "pick_best" {
                fixed.push(Step::click("Add to Cart"));
            }
        }
        fixed
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
8. FORM FILLING: If the user asks to fill a form or provides a form URL, output TWO steps: "navigate: <full form url>" (or none if already there), THEN "fillform: <brief instructions, e.g. 'use sensible sample values' or 'use my email john@example.com'>". The server will enumerate fields and fill them automatically. Do not invent per-field type/click steps.
9. ONE TAB ONLY: The automation browser has a single tab. Navigating to a new site REPLACES the current page — everything on the previous site is gone. When comparing prices across sites (e.g. "amazon or flipkart"), COMPLETE all actions on the FIRST site, including Add to Cart, BEFORE navigating to the second site. Never abandon a half-finished purchase to open another site.
10. LOGIN WALLS: Some sites (e.g. Flipkart) require an account before Add to Cart works — clicking it redirects to a sign-in page. You cannot log the user in. When you KNOW a site blocks anonymous buying: do NOT plan add-to-cart steps on it, do NOT retry. Add to cart on the site that allows it, or just search and report the price difference to the user.
11. BEST PRODUCT (OVERRIDES RULE 4): If the user says "add the best product" or "buy the best value", on a search results page output EXACTLY TWO steps: "pick_best:" (selects and opens the best product by stars*ln(reviews+1)/price), THEN "click: Add to Cart" on the product detail page. DO NOT emit "click: the first product result" or any other product-click step — pick_best IS the product selection.

SUPPORTED ACTION TAGS:
- navigate: [URL or Domain] - only if user needs to go to a new site
- click: [Target element description] - visual element on the page
- type: [Text to type] - into the currently focused input
- press: [Key like Enter/Tab/Escape]
- scroll: [down/up]
- search: [query] - navigate to search engine + type + enter
- fillform: [instructions] - auto-fill form fields after navigating to a form URL
- pick_best: - on a search results page, score products by stars*ln(reviews+1)/price and click best Add to Cart

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
   - Login redirect -> if it is a DISMISSIBLE sign-in popup (Amazon lets you keep browsing),
     press Escape and continue. If the error says the purchase is BLOCKED by a login wall
     (Flipkart-style): DO NOT retry, DO NOT dismiss and re-click. The site requires an
     account the user hasn't given us — skip this site and continue on the other site,
     or stop and report the login requirement to the user.
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
        if let Some(rest) = step.strip_prefix("fillform:").or_else(|| step.strip_prefix("fillform ")) {
            return Ok(Step::fill_form(rest.trim()));
        }
        if step == "pick_best:" || step == "pick_best" {
            return Ok(Step::pick_best());
        }
        if let Some(rest) = step.strip_prefix("wait:").or_else(|| step.strip_prefix("wait ")) {
            let ms = rest.trim().parse::<u64>().unwrap_or(2000);
            return Ok(Step::wait(ms));
        }
        if let Some(rest) = step.strip_prefix("extract:").or_else(|| step.strip_prefix("extract ")) {
            return Ok(Step::extract(rest.trim()));
        }
        if step == "back:" || step == "back" {
            return Ok(Step::back());
        }

        Err(anyhow!("unknown step format: {step}"))
    }

    /// Ask the planner LLM to generate values for form fields given the user's goal.
    /// Returns a JSON array: [{"index": 0, "value": "..."}, {"index": 3, "choose": "Option 1"}, ...]
    pub async fn form_values(&self, goal: &str, schema: &Value) -> Result<Value> {
        let prompt = format!(
            r#"You are filling a web form. User goal: "{}".
Form fields (JSON): {}

CRITICAL: You MUST provide a value for EVERY field in the schema. Do not skip any field.
For each field, choose a value based on its kind and question:

- kind "text" / "textarea": ALWAYS provide a "value" string.
  * If question mentions "email" / "e-mail" / "mail" -> valid email like "user@example.com"
  * If question mentions "phone" / "mobile" / "contact" / "cell" -> phone like "9876543210"
  * If question mentions "name" -> "John Doe"
  * Otherwise -> short relevant text like "Test answer"
- kind "radio": provide "choose" with exactly ONE option string from the field's options (or the field's "option" property)
- kind "checkbox": provide "choose" as array of option strings (if question says "tick all" / "check all", include all options; otherwise pick 1-2)
- kind "select": provide "value" matching one of the options

Output ONLY JSON: {{"values":[{{"index":0,"value":"..."}}, {{"index":3,"choose":"Option 1"}}, {{"index":5,"choose":["Option 1","Option 2"]}}]}}"#,
            goal, serde_json::to_string_pretty(schema).unwrap_or_default()
        );

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
            .context("call ollama for form values")?;

        let body: Value = resp.json().await.context("parse ollama form values response")?;
        let text = body["response"]
            .as_str()
            .ok_or_else(|| anyhow!("no response from ollama for form values"))?
            .trim();

        let cleaned = text
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: Value = serde_json::from_str(cleaned).context("parse form values JSON")?;
        Ok(parsed.get("values").cloned().unwrap_or(json!([])))
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