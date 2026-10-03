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

/// Is `needle` a whole word inside `text`?
///
/// Substring matching is what makes a loose test loose: "keyboard" hides inside
/// "keyboard shortcuts", and "best" inside "bestest". Boundaries are the whole
/// point.
fn word_in(text: &str, needle: &str) -> bool {
    let tb = text.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(needle) {
        let at = from + rel;
        let end = at + needle.len();
        let left_ok = at == 0 || !tb[at - 1].is_ascii_alphanumeric();
        let right_ok = end >= tb.len() || !tb[end].is_ascii_alphanumeric();
        if left_ok && right_ok {
            return true;
        }
        from = at + needle.len();
        if from >= text.len() {
            return false;
        }
    }
    false
}

/// First `http(s)://` URL in free text, if any.
fn url_in_text(text: &str) -> Option<String> {
    let start = text.find("http://").or_else(|| text.find("https://"))?;
    let rest = &text[start..];
    // Stop at whitespace or a quote/bracket. Trailing sentence punctuation is
    // trimmed so "visit https://x.com/y." yields "https://x.com/y".
    let end = rest
        .find(|c: char| {
            c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | ')' | ']' | '}' | '`')
        })
        .unwrap_or(rest.len());
    let mut url = rest[..end]
        .trim_end_matches(['.', ',', ';', ':', '!', '?'])
        .to_string();
    if url.is_empty() {
        return None;
    }
    Some(url)
}

/// Does this URL point at a specific page, rather than a bare homepage?
///
/// A bare host is not a destination the user chose: "search on duckduckgo.com"
/// names a *site*, and planner rule 7 explicitly allows shortening those to a
/// domain. A path, query or fragment is a page they pointed at, and truncating
/// it is what turns a request into the wrong page.
fn is_specific_url(url: &str) -> bool {
    let Some(rest) = url.split_once("://").map(|(_, r)| r) else {
        return false;
    };
    match rest.find('/') {
        None => false,          // "example.com"
        Some(i) => &rest[i..] != "/", // "example.com/" is still the homepage
    }
}

/// Scheme, `www.` and a trailing `/` are not distinctions the user cares about.
fn norm_url(u: &str) -> &str {
    let s = u.trim();
    let s = s.split_once("://").map(|(_, r)| r).unwrap_or(s);
    let s = s.strip_prefix("www.").unwrap_or(s);
    s.trim_end_matches('/')
}

/// Do two URLs land on the same page?
fn same_destination(a: &str, b: &str) -> bool {
    norm_url(a) == norm_url(b)
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
        // A URL the user gave is not the planner's to shorten.
        steps = Self::enforce_goal_url(goal, steps);
        Ok(steps)
    }

    /// Fix plan for "best product" queries: ensure pick_best is used correctly.
    fn fix_best_product_plan(goal: &str, steps: Vec<Step>) -> Vec<Step> {
        if !Self::wants_best_product(goal) {
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

    /// Does the goal ask for a *chosen* product rather than a named one?
    ///
    /// The old check was `contains("best product") || contains("best value")`,
    /// which is a phrase match against wording nobody types. Measured
    /// 2026-10-03: the user asked for "the best laptop", so the predicate was
    /// false, the whole post-processor was skipped, and the plan kept its
    /// `click: the first product result` step — the exact step rule 11 exists
    /// to forbid. The model was told to omit it and did not.
    ///
    /// The signal is "best" applied to something buyable, not one fixed phrase.
    /// "the best laptop", "best value", "best rated", "best deal" are the same
    /// request.
    ///
    /// The noun has to be *near* "best", and the goal has to read as a purchase.
    ///
    /// A flat `noun in goal` test is not enough: "what's the best way to clean a
    /// keyboard" mentions a keyboard and is not a shopping request. Proximity
    /// alone is not enough either — "find the best keyboard shortcuts" is a
    /// lookup, and "best keyboard" is a purchase, and the two are the same four
    /// words. The discriminator is the buying: cart, buy, order, price.
    ///
    /// Both conditions are required because the cost is asymmetric. This
    /// function gates a rewrite that appends `click: Add to Cart`, so a false
    /// positive puts something in the user's basket that they never asked for —
    /// a side effect on the world, not a wrong answer in the chat. A false
    /// negative only leaves a suboptimal step in a plan the model already chose.
    /// When unsure, do nothing.
    fn wants_best_product(goal: &str) -> bool {
        const PURCHASE: &[&str] = &[
            "cart", "buy", "purchase", "order", "checkout", "price", "cost",
            "deal", "cheapest", "budget",
        ];
        const BUYABLE: &[&str] = &[
            "laptop", "phone", "tablet", "monitor", "headphone", "keyboard", "mouse",
            "camera", "watch", "tv", "console", "gpu", "ssd", "product", "value",
            "deal", "option", "choice", "book", "router", "printer", "speaker",
            "vacuum", "fridge", "oven", "laptop", "smartwatch", "earbuds", "tablet",
        ];
        let g = goal.to_lowercase();
        let bytes = g.as_bytes();
        let mut from = 0usize;
        while let Some(rel) = g[from..].find("best") {
            let at = from + rel;
            // Whole word only: "bestest" or a name containing it is not a
            // superlative.
            let before_ok = at == 0 || !bytes[at - 1].is_ascii_alphanumeric();
            let after = at + 4;
            let after_ok =
                after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
            if before_ok && after_ok {
                // Four words is enough for "best rated 27 inch monitor" and
                // short enough to exclude "best way to clean a keyboard".
                let tail_end = g[after..]
                    .char_indices()
                    .filter(|(_, c)| c.is_whitespace())
                    .nth(4)
                    .map(|(i, _)| after + i)
                    .unwrap_or(g.len());
                let window = &g[after..tail_end];
                if BUYABLE.iter().any(|n| word_in(window, n))
                    && PURCHASE.iter().any(|p| word_in(&g, p))
                {
                    return true;
                }
            }
            from = at + 4;
            if from >= g.len() {
                break;
            }
        }
        false
    }

    /// Force a user-supplied URL to be the first navigate step, verbatim.
    ///
    /// Planner rule 7 asks the model to keep a full URL intact. Asking is not
    /// the same as guaranteeing: the same 7B model that ignored rule 11 also
    /// trimmed a DuckDuckGo AI URL down to the bare domain, which is the
    /// "wrong url" confusion reported on 2026-10-03. A trimmed
    /// `https://duckduckgo.com/?q=...&ia=chat` becomes the DDG homepage — the
    /// page loads, looks plausible, and contains none of what was asked for.
    ///
    /// So the URL comes from the goal, not the plan. If the goal names a URL
    /// with a path, a query or a fragment, that URL is what gets navigated to,
    /// whatever the planner emitted. A goal with only a bare domain is left
    /// alone: rule 7 allows shortening those, and `duckduckgo.com` in prose is
    /// often shorthand for a search the planner should build.
    fn enforce_goal_url(goal: &str, steps: Vec<Step>) -> Vec<Step> {
        let Some(url) = url_in_text(goal) else {
            return steps;
        };
        // A bare domain is not evidence of a specific destination.
        if !is_specific_url(&url) {
            return steps;
        }

        let first_nav = steps.iter().position(|s| s.action == "navigate");
        match first_nav {
            // Replace only if the planner navigated somewhere else first.
            Some(i) => {
                if steps[i].target.as_deref().map(|t| same_destination(t, &url)).unwrap_or(false) {
                    return steps;
                }
                let mut fixed = steps;
                fixed[i] = Step::navigate(&url);
                fixed
            }
            // No navigate at all: prepend, rather than trusting the model to
            // have meant the page it was already on.
            None => {
                let mut fixed = vec![Step::navigate(&url)];
                fixed.extend(steps);
                fixed
            }
        }
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
        let steps = self.parse_plan(&response)?;
        // The same guarantee has to hold on the repair path, and it matters more
        // there: a re-plan that trims the URL navigates away from the page the
        // failing step was working on, so the retry starts from the homepage.
        Ok(Self::enforce_goal_url(goal, steps))
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

    /// The wording from the 2026-10-03 session. The old predicate matched
    /// "best product" and "best value"; the user said "the best laptop", so the
    /// post-processor returned early and left the forbidden
    /// `click: the first product result` step in place.
    #[test]
    fn the_best_laptop_counts_as_a_best_product_goal() {
        assert!(Planner::wants_best_product("open amazon and add the best laptop to my cart"));
        assert!(Planner::wants_best_product("add the best value tablet to cart"));
        assert!(Planner::wants_best_product("buy the best rated monitor"));
    }

    /// Widening the predicate must not hijack non-purchase "best" requests. A
    /// plan for "what's the best way to clean a keyboard" must be left alone,
    /// or pick_best gets inserted into a plan that has nothing to pick.
    #[test]
    fn best_without_a_product_is_not_a_purchase() {
        assert!(!Planner::wants_best_product("what's the best way to clean a keyboard"));
        assert!(!Planner::wants_best_product("find the best keyboard shortcuts"));
        assert!(!Planner::wants_best_product("tell me a joke"));
        // The case that forced the purchase requirement. "best keyboard" is a
        // purchase; "best keyboard shortcuts" is a lookup, and the two differ
        // only in the words after the noun. Proximity to "best" cannot tell
        // them apart, so the buying verb is what decides.
        assert!(!Planner::wants_best_product("find the best keyboard shortcuts for vscode"));
        assert!(
            Planner::wants_best_product("find the best keyboard and buy it"),
            "a purchase verb anywhere in the goal is enough"
        );
    }

    /// The full post-processor, on the plan the model actually produced.
    #[test]
    fn the_forbidden_product_click_is_removed_for_a_best_laptop_goal() {
        let steps = vec![
            Step::navigate("https://www.amazon.com"),
            Step::search("best laptop"),
            Step::click("the first product result"),
            Step::pick_best(),
            Step::click("Add to Cart"),
        ];
        // The goal string exactly as Luna passed it in the 2026-10-03 log.
        let fixed = Planner::fix_best_product_plan(
            "open amazon and add the best laptop to cart",
            steps,
        );
        assert!(
            !fixed.iter().any(|s| {
                s.target.as_deref().is_some_and(|t| t.contains("first product"))
            }),
            "the product-click step rule 11 forbids survived: {:?}",
            fixed.iter().map(|s| (&s.action, &s.target)).collect::<Vec<_>>()
        );
        assert!(fixed.iter().any(|s| s.action == "pick_best"));
    }

    /// The DuckDuckGo case. The model emitted the bare domain, so the browser
    /// landed on the homepage: a page that loads fine and contains none of what
    /// was asked for, which is what read as "wrong url".
    #[test]
    fn a_query_string_url_in_the_goal_survives_the_planner() {
        let goal = "check https://duckduckgo.com/?q=best+laptop+2026&ia=chat and tell me the answer";
        let steps = vec![
            Step::navigate("duckduckgo.com"),
            Step::search("best laptop 2026"),
        ];
        let fixed = Planner::enforce_goal_url(goal, steps);
        assert_eq!(
            fixed[0].action, "navigate",
            "first step should still be the navigation"
        );
        assert_eq!(
            fixed[0].target.as_deref(),
            Some("https://duckduckgo.com/?q=best+laptop+2026&ia=chat"),
            "the planner's trimmed domain replaced the user's URL"
        );
    }

    /// A bare domain is not a specific destination. Rule 7 allows shortening
    /// those, and hijacking them would break ordinary searches.
    #[test]
    fn a_bare_domain_is_left_to_the_planner() {
        let steps = vec![Step::navigate("duckduckgo.com"), Step::search("best laptop")];
        let fixed = Planner::enforce_goal_url("search on duckduckgo.com for a laptop", steps);
        assert_eq!(
            fixed[0].target.as_deref(),
            Some("duckduckgo.com"),
            "a bare host was rewritten"
        );
    }

    /// When the planner got it right, leave it right. A fix that rewrites
    /// correct plans is its own bug.
    #[test]
    fn an_already_correct_navigation_is_untouched() {
        let url = "https://docs.google.com/forms/d/e/xyz/viewform";
        let steps = vec![Step::navigate(url), Step::fill_form("use my email")];
        let fixed = Planner::enforce_goal_url(&format!("fill out {url} please"), steps.clone());
        assert_eq!(fixed.len(), steps.len());
        assert_eq!(fixed[0].target.as_deref(), Some(url));
    }

    /// A plan with no navigation at all still gets the user's URL, rather than
    /// being left to act on a page nobody chose.
    #[test]
    fn a_url_in_the_goal_is_prepended_when_the_plan_never_navigates() {
        let url = "https://example.com/reports/42";
        let steps = vec![Step::click("Download"), Step::click("Add to Cart")];
        let n = steps.len();
        let fixed = Planner::enforce_goal_url(&format!("download the report from {url}"), steps);
        assert_eq!(fixed[0].action, "navigate");
        assert_eq!(fixed[0].target.as_deref(), Some(url));
        assert_eq!(fixed.len(), n + 1);
    }

    #[test]
    fn urls_are_read_out_of_running_text() {
        assert_eq!(
            url_in_text("go to https://example.com/a/b?c=1 and read it"),
            Some("https://example.com/a/b?c=1".into())
        );
        assert_eq!(
            url_in_text("see (https://example.com/x)."),
            Some("https://example.com/x".into())
        );
        assert_eq!(url_in_text("no url here"), None);
        assert!(is_specific_url("https://example.com/a"));
        assert!(!is_specific_url("https://example.com"));
        assert!(!is_specific_url("https://example.com/"));
    }

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