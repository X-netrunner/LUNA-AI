//! Plan executor — runs steps in the browser, handles retries and re-planning.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::time::timeout;

use crate::browser::{cdp::CdpBrowser, dom, planner::Planner, Step, types::ActionResult};

const MAX_RETRIES: u32 = 3;
const STEP_TIMEOUT_SECS: u64 = 30;

/// Hard ceiling on steps executed in one plan, replans included.
///
/// The loop this bounds had no ceiling at all. Measured 2026-10-03: a five-step
/// "add the best laptop to cart" plan reached step 16 and was still going when
/// the user hit Ctrl-C, with `steps` growing by two on every failure.
const MAX_TOTAL_STEPS: usize = 30;

/// Hard ceiling on re-plans across the whole plan, not per step.
const MAX_REPLANS: u32 = 6;

/// Why a plan stopped, so the caller can report a reason instead of a hang.
///
/// Every variant here is a case that used to be indistinguishable from "still
/// working". `PlanFinished` is the only one that means the goal was met.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    PlanFinished,
    RetriesExhausted,
    ReplansExhausted,
    StepBudgetExhausted,
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Stop::PlanFinished => "the plan finished",
            Stop::RetriesExhausted => {
                "a step failed three times in a row and was skipped"
            }
            Stop::ReplansExhausted => "it re-planned six times without converging",
            Stop::StepBudgetExhausted => {
                "it ran out of steps before finishing the plan"
            }
        };
        f.write_str(s)
    }
}

/// Bookkeeping for the re-plan loop, with no browser and no planner in it.
///
/// Extracted so the loop's *decisions* are testable. The bug this exists for
/// was not reachable from a test: `execute_plan` needs a live `CdpBrowser` and
/// a model call, so a loop that provably never terminates sat behind both.
///
/// # The loop, and why it never ended
///
/// `retry_count` was reset to zero by every successful step, and every re-plan
/// *inserted* a step that then succeeded. So the counter never accumulated past
/// one, `MAX_RETRIES` was never reached, and the `while current_step <
/// steps.len()` condition kept being satisfied by the very steps being added.
/// The plan grew without bound.
///
/// The fix is not a bigger `MAX_RETRIES`. It is that a step the re-plan itself
/// inserted is not evidence of recovery: it is the repair, running. Only a
/// success on a step from the original plan may clear the failure count.
#[derive(Debug)]
pub struct LoopControl {
    /// `(step, inserted_by_replan)`.
    steps: Vec<(Step, bool)>,
    cursor: usize,
    consecutive_failures: u32,
    replans: u32,
    executed: usize,
}

impl LoopControl {
    pub fn new(steps: Vec<Step>) -> Self {
        Self {
            steps: steps.into_iter().map(|s| (s, false)).collect(),
            cursor: 0,
            consecutive_failures: 0,
            replans: 0,
            executed: 0,
        }
    }

    /// The step to run now, or `None` if the plan is done or out of budget.
    pub fn current(&self) -> Option<&Step> {
        if self.stop_reason().is_some() {
            return None;
        }
        self.steps.get(self.cursor).map(|(s, _)| s)
    }

    /// Why the plan will not run another step, if it will not.
    pub fn stop_reason(&self) -> Option<Stop> {
        if self.cursor >= self.steps.len() {
            return Some(Stop::PlanFinished);
        }
        if self.executed >= MAX_TOTAL_STEPS {
            return Some(Stop::StepBudgetExhausted);
        }
        if self.replans >= MAX_REPLANS {
            return Some(Stop::ReplansExhausted);
        }
        None
    }

    /// A step ran and worked.
    ///
    /// Returns `true` if it was an original-plan step, which is the only thing
    /// permitted to clear the failure count.
    pub fn record_success(&mut self) -> bool {
        let from_plan = self.steps.get(self.cursor).map(|(_, r)| !r).unwrap_or(false);
        self.executed += 1;
        self.cursor += 1;
        if from_plan {
            self.consecutive_failures = 0;
        }
        from_plan
    }

    /// A step ran and failed. Returns whether a re-plan is still permitted.
    pub fn record_failure(&mut self) -> bool {
        self.executed += 1;
        self.consecutive_failures += 1;
        self.consecutive_failures < MAX_RETRIES && self.replans < MAX_REPLANS
    }

    /// Splice corrective steps in at the cursor and account for the re-plan.
    pub fn insert_replan(&mut self, corrective: Vec<Step>) {
        self.replans += 1;
        self.steps.splice(
            self.cursor..self.cursor,
            corrective.into_iter().map(|s| (s, true)),
        );
    }

    /// Give up on the failing step and move past it.
    pub fn skip(&mut self) {
        self.executed += 1;
        self.cursor += 1;
        self.consecutive_failures = 0;
    }

    #[cfg(test)]
    pub fn executed(&self) -> usize {
        self.executed
    }

    /// The plan length, which grows as re-plans insert steps. The old loop
    /// used this as its only termination condition, so a growing plan was a
    /// reason to keep going rather than a reason to stop.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn cursor_index(&self) -> usize {
        self.cursor
    }

    pub fn completed_steps(&self) -> Vec<Step> {
        self.steps[..self.cursor].iter().map(|(s, _)| s.clone()).collect()
    }

    pub fn remaining_steps(&self) -> Vec<Step> {
        self.steps
            .iter()
            .skip(self.cursor + 1)
            .map(|(s, _)| s.clone())
            .collect()
    }

    pub fn replans(&self) -> u32 {
        self.replans
    }
}

/// Execute a single step in the browser.
async fn execute_step(browser: &CdpBrowser, planner: &Planner, goal: &str, step: &Step, step_index: u64) -> Result<ActionResult> {
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
            // Click might trigger navigation - wait a moment for potential page
            // load, then check whether a purchase click ran into a login wall.
            tokio::time::sleep(Duration::from_millis(1000)).await;
            if result.get("ok").and_then(Value::as_bool).unwrap_or(false)
                && is_purchase_click(target)
            {
                if let Some(reason) = browser.login_wall_reason().await {
                    let err = format!(
                        "purchase blocked — {reason}. Do NOT retry clicking; this site \
                         needs the user logged in. Continue elsewhere or report back."
                    );
                    return Ok(ActionResult::fail(
                        action_name,
                        step_index,
                        1,
                        err,
                    ));
                }
            }
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
        "fillform" => {
            let _instructions = step.target.as_deref().unwrap_or("use sensible sample values");
            // 1. Extract form schema
            let schema = browser.eval(dom::dom_form_schema_expr()).await?;
            let fields = schema.get("fields").cloned().unwrap_or(json!([]));
            // 2. Ask planner LLM for values
            let values = planner.form_values(goal, &fields).await?;
            // 3. Fill the form
            let fill_expr = dom::dom_form_fill_expr(&serde_json::to_string(&values)?);
            let result = browser.eval(&fill_expr).await?;
            if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                let filled = result.get("filled").cloned().unwrap_or(json!([]));
                let note = format!("filled {} field(s): {}", filled.as_array().map(|a| a.len()).unwrap_or(0),
                    serde_json::to_string(&filled).unwrap_or_default());
                Ok(ActionResult::ok_note(action_name, step_index, 1, note))
            } else {
                Ok(dom::result_from(result, action_name, step_index, 1))
            }
        }
        "pick_best" => {
            let result = browser.eval(dom::dom_pick_best_expr()).await?;
            if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                let clicked = result.get("clicked").cloned().unwrap_or(json!({}));
                let note = format!("clicked best product: {}", serde_json::to_string(&clicked).unwrap_or_default());
                Ok(ActionResult::ok_note(action_name, step_index, 1, note))
            } else {
                Ok(dom::result_from(result, action_name, step_index, 1))
            }
        }
        "wait" | "pause" => {
            let ms = step.value.as_deref().and_then(|s| s.parse::<u64>().ok())
                .or_else(|| step.target.as_deref().and_then(|s| s.parse::<u64>().ok()))
                .unwrap_or(2000);
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(ActionResult::ok(action_name, step_index, 1))
        }
        "extract" | "read" => {
            let target = step.target.as_deref().unwrap_or("body");
            let expr = dom::dom_extract_text_expr(target);
            let result = browser.eval(&expr).await?;
            if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                let text = result.get("text").and_then(Value::as_str).unwrap_or("");
                let note = format!("extracted text: {}", text);
                Ok(ActionResult::ok_note(action_name, step_index, 1, note))
            } else {
                Ok(dom::result_from(result, action_name, step_index, 1))
            }
        }
        "back" => {
            let result = browser.eval("(() => { window.history.back(); return { ok: true }; })()").await?;
            tokio::time::sleep(Duration::from_millis(1000)).await;
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        other => Ok(ActionResult::fail(action_name, step_index, 1, format!("unsupported action '{other}'"))),
    }
}

/// Whether a click description targets a purchase action (where a login wall
/// would block completion).
fn is_purchase_click(desc: &str) -> bool {
    let lower = desc.to_lowercase();
    [
        "add to cart", "add to bag", "buy now", "checkout", "place order",
        "proceed to", "continue to checkout", "add to wishlist", "add to bucket",
    ]
    .iter()
    .any(|k| lower.contains(k))
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
    steps: Vec<Step>,
) -> Result<Vec<String>> {
    let mut history = Vec::new();
    let mut plan = LoopControl::new(steps);
    let total_steps = plan_len(&plan);

    loop {
        let Some(step) = plan.current().cloned() else {
            break;
        };
        let index = plan.cursor_index();

        // Execute with timeout
        let result = timeout(Duration::from_secs(STEP_TIMEOUT_SECS), execute_step(browser, planner, goal, &step, index as u64))
            .await
            .context("step timed out")??;

        let desc = describe_step(&step, &result);
        history.push(desc.clone());

        if result.success {
            let from_plan = plan.record_success();
            tracing::info!("Step {}/{}: {}", plan.cursor_index(), total_steps, desc);
            if !from_plan {
                // Worth saying out loud, because it is the whole reason this
                // loop used to be infinite: the repair running is not recovery.
                tracing::debug!("  (re-plan step succeeded; retry count stands)");
            }
            continue;
        }

        let error = result.error.unwrap_or_else(|| "unknown error".into());
        tracing::warn!(
            "Step {}/{} failed: {}",
            plan.cursor_index() + 1,
            total_steps,
            error
        );

        if !plan.record_failure() {
            tracing::error!(
                "Giving up on this step after {} attempts: {}",
                MAX_RETRIES,
                error
            );
            plan.skip();
            continue;
        }

        let completed = plan.completed_steps();
        let remaining = plan.remaining_steps();
        match planner.replan(goal, &completed, &step, &error, &remaining).await {
            Ok(corrective) if !corrective.is_empty() => {
                tracing::info!(
                    "Replan {}/{}: inserting {} corrective step(s) — {}",
                    plan.replans() + 1,
                    MAX_REPLANS,
                    corrective.len(),
                    error
                );
                plan.insert_replan(corrective);
            }
            _ => {
                tracing::warn!("Re-plan returned nothing usable; skipping this step");
                plan.skip();
            }
        }
    }

    if let Some(stop) = plan.stop_reason() {
        if stop != Stop::PlanFinished {
            tracing::warn!("Plan stopped early: {stop}");
            history.push(format!("  ✘ stopped: {stop}"));
        }
    }

    Ok(history)
}

fn plan_len(plan: &LoopControl) -> usize {
    plan.len()
}

fn describe_step(step: &Step, result: &ActionResult) -> String {
    let base = match step.action.as_str() {
        "navigate" => format!("open {}", step.target.as_deref().unwrap_or("")),
        "click" => format!("click {}", step.target.as_deref().unwrap_or("")),
        "type" => format!("type: {}", step.value.as_deref().unwrap_or("")),
        "press" => format!("press {}", step.target.as_deref().unwrap_or("Enter")),
        "scroll" => format!("scroll {}", step.target.as_deref().unwrap_or("down")),
        "search" => format!("search for \"{}\"", step.target.as_deref().unwrap_or("")),
        "fillform" => format!("fill form ({})", step.target.as_deref().unwrap_or("auto")),
        "pick_best" => "pick best product".to_string(),
        other => other.to_string(),
    };
    if result.success {
        if let Some(note) = &result.note {
            format!("  ✔ {base} — {note}")
        } else {
            format!("  ✔ {base}")
        }
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

    fn step(action: &str, target: &str) -> Step {
        Step {
            action: action.into(),
            target: Some(target.into()),
            value: None,
            is_last: false,
        }
    }

    /// The loop from the 2026-10-03 session, driven exactly as the log drove it.
    ///
    /// The plan was: navigate, search, click the first product, add to cart.
    /// `pick_best` failed with "no scorable product cards found", the re-plan
    /// inserted a step, that step succeeded, the retry count reset to zero, and
    /// the next step failed — forever. The log reached step 16 and the user
    /// killed it.
    ///
    /// The shape that matters is the alternation: fail, repair-succeeds,
    /// fail, repair-succeeds. Against the old bookkeeping this never ends.
    #[test]
    fn the_replan_loop_terminates_where_it_used_to_run_forever() {
        let mut plan = LoopControl::new(vec![
            step("navigate", "amazon.com"),
            step("search", "best laptop"),
            step("pick_best", ""),
            step("click", "Add to Cart"),
        ]);

        let mut iterations = 0;
        // The old loop's only stop condition was `cursor < steps.len()`, which
        // the re-plans kept satisfying. Bound the harness too, so this test
        // fails loudly rather than hanging if the bug returns.
        while plan.current().is_some() && iterations < 500 {
            iterations += 1;
            let s = plan.current().unwrap().clone();
            if s.action == "pick_best" || s.action == "click" {
                if plan.record_failure() {
                    plan.insert_replan(vec![step("click", "the first product result")]);
                } else {
                    plan.skip();
                }
            } else {
                plan.record_success();
            }
        }

        assert!(
            iterations < 100,
            "the loop did not converge in {iterations} iterations"
        );
        assert!(
            plan.current().is_none(),
            "still running after {iterations} iterations"
        );
        assert_eq!(
            plan.stop_reason(),
            Some(Stop::ReplansExhausted),
            "should stop because re-planning never converged, not because the plan finished"
        );
        assert!(
            plan.len() > 4,
            "the test did not actually exercise re-planning (len {})",
            plan.len()
        );
    }

    /// The specific mechanism. A step the re-plan inserted is the repair
    /// running; it is not evidence that the plan recovered. When it was allowed
    /// to clear the failure count, the count could never reach `MAX_RETRIES`.
    #[test]
    fn a_successful_repair_step_does_not_clear_the_failure_count() {
        let mut plan = LoopControl::new(vec![step("click", "Add to Cart")]);

        assert!(plan.record_failure(), "first failure may re-plan");
        plan.insert_replan(vec![step("click", "the first product result")]);

        // The repair succeeds. This must NOT hand the plan a clean slate.
        let from_plan = plan.record_success();
        assert!(!from_plan, "a re-plan step must not count as plan progress");

        // So the next failure is the second consecutive one, and only the third
        // may skip the step.
        assert!(plan.record_failure(), "second failure may still re-plan");
        assert!(!plan.record_failure(), "third failure must skip");
        // Refusing a re-plan is not itself "stopped" — the caller skips the
        // step, which is what ends a one-step plan.
        plan.skip();
        assert_eq!(plan.stop_reason(), Some(Stop::PlanFinished));
        assert_eq!(plan.executed(), 5, "3 failures + 1 repair + the skip");
    }

    /// The absolute bound. Even a plan where every step succeeds and every
    /// failure re-plans must stop.
    #[test]
    fn the_step_budget_bounds_a_plan_that_never_settles() {
        let mut plan = LoopControl::new(vec![step("navigate", "amazon.com")]);
        let mut n = 0;
        while plan.current().is_some() && n < 1000 {
            n += 1;
            if plan.record_failure() {
                plan.insert_replan(vec![step("click", "Add to Cart")]);
            } else {
                plan.skip();
            }
        }
        assert!(plan.current().is_none(), "still running after {n} iterations");
        assert!(plan.executed() <= MAX_TOTAL_STEPS + MAX_RETRIES as usize);
        assert!(matches!(
            plan.stop_reason(),
            Some(Stop::ReplansExhausted) | Some(Stop::StepBudgetExhausted)
        ));
    }

    /// A plan that just works is unaffected, and reports `PlanFinished` rather
    /// than one of the failure stops.
    #[test]
    fn a_working_plan_finishes_normally() {
        let mut plan = LoopControl::new(vec![
            step("navigate", "amazon.com"),
            step("click", "Add to Cart"),
        ]);
        assert!(plan.record_success());
        assert!(plan.record_success());
        assert_eq!(plan.stop_reason(), Some(Stop::PlanFinished));
        assert!(plan.current().is_none());
        assert_eq!(plan.replans(), 0);
    }

    /// Two original steps failing in a row *should* count. The fix must not
    /// have broken the ordinary retry case it was protecting.
    #[test]
    fn two_real_failures_still_count_as_two() {
        let mut plan = LoopControl::new(vec![
            step("click", "a"),
            step("click", "b"),
        ]);
        assert!(plan.record_failure());
        plan.insert_replan(vec![step("wait", "")]);
        // The repair runs and the ORIGINAL step is retried.
        plan.record_success(); // the wait
        let at_b = plan.current().unwrap().clone();
        assert_eq!(at_b.action, "click");
        assert!(plan.record_failure(), "b's first failure may re-plan");
        assert!(!plan.record_failure(), "b's second failure must skip");
    }

    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("ps4 on amazon.com"), "ps4%20on%20amazon.com");
        assert_eq!(percent_encode("add to cart & buy 'now'"), "add%20to%20cart%20%26%20buy%20%27now%27");
    }
}