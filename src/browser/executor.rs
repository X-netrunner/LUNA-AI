//! Plan executor — runs steps in the browser, handles retries and re-planning.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::time::Duration;
use tokio::time::timeout;

use crate::browser::{cdp::CdpBrowser, dom, planner::Planner, Step, types::ActionResult};

const MAX_RETRIES: u32 = 3;
const STEP_TIMEOUT_SECS: u64 = 30;

/// Execute a single step in the browser.
async fn execute_step(browser: &CdpBrowser, step: &Step, step_index: u64) -> Result<ActionResult> {
    let action_name = step.action.clone();

    match step.action.as_str() {
        "navigate" => {
            let url = step.target.as_deref().unwrap_or("");
            if url.is_empty() {
                return Ok(ActionResult::fail(action_name, step_index, 1, "navigate needs a url".into()));
            }
            browser.navigate(url).await?;
            Ok(ActionResult::ok(action_name, step_index, 1))
        }
        "click" => {
            let target = step.target.as_deref().unwrap_or("");
            let expr = find_and_click_expr(target);
            let result = browser.eval(&expr).await?;
            // Click might trigger navigation - wait a moment for potential page load
            tokio::time::sleep(Duration::from_millis(1000)).await;
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "type" => {
            let text = step.value.as_deref().unwrap_or("");
            if text.is_empty() {
                return Ok(ActionResult::fail(action_name, step_index, 1, "type needs text".into()));
            }
            let result = browser.eval(&dom::dom_type_expr(text)).await?;
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "press" => {
            let key = step.target.as_deref().unwrap_or("Enter");
            let result = browser.eval(&dom::dom_press_expr(key)).await?;
            // Press Enter might trigger navigation
            if key == "Enter" {
                tokio::time::sleep(Duration::from_millis(1000)).await;
            }
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "scroll" => {
            let direction = step.target.as_deref().unwrap_or("down");
            let amount = step.value.as_deref().and_then(|s| s.parse().ok()).unwrap_or(500);
            let result = browser.eval(&dom::dom_scroll_expr(direction, amount)).await?;
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "search" => {
            let query = step.target.as_deref().unwrap_or("");
            if query.is_empty() {
                return Ok(ActionResult::fail(action_name, step_index, 1, "search needs a query".into()));
            }
            let url = format!("https://duckduckgo.com/?q={}", percent_encode(query));
            browser.navigate(&url).await?;
            Ok(ActionResult::ok(action_name, step_index, 1))
        }
        other => Ok(ActionResult::fail(action_name, step_index, 1, format!("unsupported action '{other}'"))),
    }
}

/// Generate JS to find an element by description and click it.
fn find_and_click_expr(description: &str) -> String {
    let desc = serde_json::to_string(description).unwrap_or_else(|_| "\"\"".into());

    format!(
        r#"(() => {{
  const desc = {desc}.toLowerCase();

  // Try to find element by various matching strategies
  const candidates = Array.from(document.querySelectorAll('button,a,input,textarea,select,[role="button"],[role="link"],[role="radio"],[role="checkbox"],[role="option"],[onclick],[tabindex]'));

  // Score each candidate by how well it matches the description
  let best = null;
  let bestScore = -1;

  for (const el of candidates) {{
    let score = 0;
    const text = (el.innerText || el.textContent || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').toLowerCase();
    const id = (el.id || '').toLowerCase();
    const cls = (el.className || '').toLowerCase();
    const aria = (el.getAttribute('aria-label') || '').toLowerCase();

    // Exact substring match
    if (text.includes(desc) || id.includes(desc) || cls.includes(desc) || aria.includes(desc)) score += 10;
    // Word overlap
    const descWords = desc.split(/\\s+/);
    for (const w of descWords) {{
      if (w.length > 2 && (text.includes(w) || id.includes(w) || cls.includes(w))) score += 3;
    }}
    // Prefer buttons/links for "click" actions
    if (el.tagName === 'BUTTON' || el.tagName === 'A') score += 2;
    // Prefer visible elements
    const rect = el.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0 && rect.top < window.innerHeight && rect.bottom > 0) score += 1;

    if (score > bestScore) {{
      bestScore = score;
      best = el;
    }}
  }}

  if (!best || bestScore === 0) {{
    return {{ ok: false, error: "no matching element found for: " + desc }};
  }}

  // Click the best match
  const rect = best.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  const ev = {{ bubbles: true, cancelable: true, view: window, clientX: x, clientY: y }};
  best.dispatchEvent(new PointerEvent("pointerdown", ev));
  best.dispatchEvent(new MouseEvent("mousedown", ev));
  best.focus();
  best.dispatchEvent(new PointerEvent("pointerup", ev));
  best.dispatchEvent(new MouseEvent("mouseup", ev));
  best.click();
  return {{ ok: true }};
}})()"#
    )
}

/// Execute a full plan with retry/rethink logic.
pub async fn execute_plan(
    browser: &CdpBrowser,
    planner: &Planner,
    goal: &str,
    mut steps: Vec<Step>,
) -> Result<Vec<String>> {
    let mut history = Vec::new();
    let mut current_step = 0;
    let mut retry_count = 0;
    let total_steps = steps.len();

    while current_step < steps.len() {
        let step = steps[current_step].clone();

        // Execute with timeout
        let result = timeout(Duration::from_secs(STEP_TIMEOUT_SECS), execute_step(browser, &step, current_step as u64))
            .await
            .context("step timed out")??;

        let desc = describe_step(&step, &result);
        history.push(desc.clone());

        if result.success {
            current_step += 1;
            retry_count = 0;
            tracing::info!("Step {}/{}: {}", current_step, total_steps, desc);
        } else {
            let error = result.error.unwrap_or_else(|| "unknown error".into());
            tracing::warn!("Step {}/{} failed: {}", current_step + 1, total_steps, error);

            if retry_count < MAX_RETRIES {
                let completed: Vec<Step> = steps.iter().take(current_step).cloned().collect();
                let remaining: Vec<Step> = steps.iter().skip(current_step + 1).cloned().collect();

                match planner.replan(goal, &completed, &step, &error, &remaining).await {
                    Ok(corrective_steps) if !corrective_steps.is_empty() => {
                        tracing::info!("Replan: inserting {} corrective step(s)", corrective_steps.len());
                        steps.splice(current_step..current_step, corrective_steps);
                        retry_count += 1;
                        continue;
                    }
                    _ => {
                        retry_count += 1;
                        if retry_count >= MAX_RETRIES {
                            tracing::error!("Max retries reached for step, skipping");
                            current_step += 1;
                            retry_count = 0;
                        }
                    }
                }
            } else {
                tracing::error!("Max retries reached for step, skipping");
                current_step += 1;
                retry_count = 0;
            }
        }
    }

    Ok(history)
}

fn describe_step(step: &Step, result: &ActionResult) -> String {
    let base = match step.action.as_str() {
        "navigate" => format!("open {}", step.target.as_deref().unwrap_or("")),
        "click" => format!("click {}", step.target.as_deref().unwrap_or("")),
        "type" => format!("type: {}", step.value.as_deref().unwrap_or("")),
        "press" => format!("press {}", step.target.as_deref().unwrap_or("Enter")),
        "scroll" => format!("scroll {}", step.target.as_deref().unwrap_or("down")),
        "search" => format!("search for \"{}\"", step.target.as_deref().unwrap_or("")),
        other => other.to_string(),
    };
    if result.success {
        format!("  ✔ {base}")
    } else {
        format!(
            "  ✘ {base} — {}",
            result.error.as_deref().unwrap_or("failed")
        )
    }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'!' | b'*' | b'\'' | b'(' | b')' | b';' | b':' | b'@' | b'&' | b'=' | b'+'
            | b'$' | b',' | b'/' | b'?' | b'#' | b'[' | b']' | b'%' => {
                out.push_str(&format!("%{b:02X}"));
            }
            _ if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("ps4 on amazon.com"), "ps4%20on%20amazon.com");
        assert_eq!(percent_encode("add to cart & buy 'now'"), "add%20to%20cart%20%26%20buy%20%27now%27");
    }
}