//! Plan executor — runs steps in the browser, handles retries and re-planning.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::time::timeout;

use crate::browser::{cdp::CdpBrowser, dom, planner::Planner, Step, types::ActionResult};

const MAX_RETRIES: u32 = 3;
const STEP_TIMEOUT_SECS: u64 = 30;

/// How long `pick_best` waits for a results page to render before giving up.
const PICK_BEST_WAIT: Duration = Duration::from_secs(8);
const PICK_BEST_POLL: Duration = Duration::from_millis(300);

/// Should `pick_best` look at the page again?
///
/// Yes only when there were no cards at all. Cards present but unscorable means
/// the selectors or the markup changed, and waiting will not fix that; cards
/// absent means the page has not finished rendering, and re-reading it is the
/// whole difference between a working pick and a false "no products here".
fn pick_best_should_retry(result: &Value) -> bool {
    if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        return false;
    }
    result.get("cards_seen").and_then(Value::as_u64).unwrap_or(0) == 0
}

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
/// working". `PlanFinished` is the only one that means every step ran — but see
/// `RetriesExhausted`, which is also "finished" and is not a success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    PlanFinished,
    /// Every step was attempted, but at least one was tried `MAX_RETRIES` times
    /// and given up on. This is reported rather than folded into `PlanFinished`
    /// because "the plan ran to the end" and "the plan did everything" are
    /// different claims, and the tool result said the first while the user
    /// needed to hear the second.
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
    /// Indices of steps that were attempted and then given up on.
    abandoned: Vec<usize>,
}

impl LoopControl {
    pub fn new(steps: Vec<Step>) -> Self {
        Self {
            steps: steps.into_iter().map(|s| (s, false)).collect(),
            cursor: 0,
            consecutive_failures: 0,
            replans: 0,
            executed: 0,
            abandoned: Vec::new(),
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
            // Reaching the end is not the same as completing the plan. A step
            // that was abandoned still counts as un-done.
            return Some(if self.abandoned.is_empty() {
                Stop::PlanFinished
            } else {
                Stop::RetriesExhausted
            });
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
    ///
    /// Records that a step was abandoned, so the caller can say so. Skipping
    /// used to be silent: the plan carried on and the history showed the next
    /// step's ✔ with nothing in between, so a step that was tried three times
    /// and dropped read the same as one that never existed.
    pub fn skip(&mut self) {
        self.executed += 1;
        self.cursor += 1;
        self.consecutive_failures = 0;
        self.abandoned.push(self.cursor - 1);
    }

    /// Steps that were tried and given up on, as indices into the plan.
    pub fn abandoned(&self) -> &[usize] {
        &self.abandoned
    }

    /// A short name for the step at `idx`, for reporting.
    pub fn step_label(&self, idx: usize) -> Option<String> {
        let (step, _) = self.steps.get(idx)?;
        Some(match step.action.as_str() {
            "navigate" => format!("open {}", step.target.as_deref().unwrap_or("")),
            "click" => format!("click {}", step.target.as_deref().unwrap_or("")),
            "type" => format!("type: {}", step.value.as_deref().unwrap_or("")),
            "search" => format!("search for \"{}\"", step.target.as_deref().unwrap_or("")),
            "pick_best" => "pick best product".to_string(),
            other => other.to_string(),
        })
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
            let result = browser.eval_resilient(&expr).await?;

            if let Err(why) = purchase_click_verdict(target, &result) {
                return Ok(ActionResult::fail(action_name, step_index, 1, why));
            }
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
            // Say what was clicked, not just that something was clicked. A bare
            // "✔ click Add to Cart" is indistinguishable from having clicked
            // Add to Wishlist; the element's own text is the evidence.
            if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                let clicked = result.get("clicked").and_then(Value::as_str).unwrap_or("");
                if !clicked.is_empty() {
                    let exact = result.get("exact").and_then(Value::as_bool).unwrap_or(false);
                    let note = if exact {
                        format!("clicked \"{clicked}\"")
                    } else {
                        format!("clicked \"{clicked}\" — closest match, not an exact one")
                    };
                    return Ok(ActionResult::ok_note(action_name, step_index, 1, note));
                }
            }
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "type" => {
            let text = step.value.as_deref().unwrap_or("");
            if text.is_empty() {
                return Ok(ActionResult::fail(action_name, step_index, 1, "type needs text".into()));
            }
            let result = browser.eval_resilient(&dom::dom_type_expr(text)).await?;
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "press" => {
            let key = step.target.as_deref().unwrap_or("Enter");
            let result = browser.eval_resilient(&dom::dom_press_expr(key)).await?;
            // Press Enter might trigger navigation
            if key == "Enter" {
                tokio::time::sleep(Duration::from_millis(1000)).await;
            }
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "scroll" => {
            let direction = step.target.as_deref().unwrap_or("down");
            let amount = step.value.as_deref().and_then(|s| s.parse().ok()).unwrap_or(500);
            let result = browser.eval_resilient(&dom::dom_scroll_expr(direction, amount)).await?;
            Ok(dom::result_from(result, action_name, step_index, 1))
        }
        "search" => {
            let query = step.target.as_deref().unwrap_or("").trim();
            if query.is_empty() {
                return Ok(ActionResult::fail(action_name, step_index, 1, "search needs a query".into()));
            }
            // Search the site that is open, not DuckDuckGo. Leaving the site
            // mid-plan is what made a search-results scrape find no products.
            let here = browser.current_url().await.unwrap_or_default();
            let Some(url) = site_search_url(&here, query) else {
                return Ok(ActionResult::fail(
                    action_name,
                    step_index,
                    1,
                    format!(
                        "cannot search: no usable site is open (page is '{here}'). \
                         Use the site's own search box — click: search, type: {query}, press: Enter"
                    ),
                ));
            };
            browser.navigate(&url).await?;
            // Prove it arrived. A redirect to a login wall or a search engine
            // is a different page, and reporting success would hand the next
            // step a page nobody asked for.
            let landed = browser.current_url().await.unwrap_or_default();
            if !arrived_at(&url, &landed) {
                return Ok(ActionResult::fail(
                    action_name,
                    step_index,
                    1,
                    format!("searched '{query}' but ended up on '{landed}', not '{url}'"),
                ));
            }
            Ok(ActionResult::ok(action_name, step_index, 1))
        }
        "fillform" => {
            let _instructions = step.target.as_deref().unwrap_or("use sensible sample values");
            // 1. Extract form schema
            let schema = browser.eval_resilient(dom::dom_form_schema_expr()).await?;
            let fields = schema.get("fields").cloned().unwrap_or(json!([]));
            // 2. Ask planner LLM for values
            let values = planner.form_values(goal, &fields).await?;
            // 3. Fill the form
            let fill_expr = dom::dom_form_fill_expr(&serde_json::to_string(&values)?);
            let result = browser.eval_resilient(&fill_expr).await?;
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
            // A results page is not there the instant navigation returns.
            // Sampling once meant pick_best ran against a half-rendered page and
            // reported "no product cards matched" for a page that was about to
            // have forty of them. Poll, but only while the page has no cards at
            // all: cards present but unscorable is a selector problem, and
            // waiting cannot fix it.
            let started = std::time::Instant::now();
            loop {
                let result = browser.eval_resilient(dom::dom_pick_best_expr()).await?;
                if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                    let clicked = result.get("clicked").cloned().unwrap_or(json!({}));
                    // This step clicked through to a product page, so the next
                    // step starts against a document that does not exist yet.
                    // Wait for the page rather than letting the next click fail
                    // on a dead execution context.
                    browser.wait_until_ready(Duration::from_millis(8000)).await?;
                    let note = format!("clicked best product: {}", serde_json::to_string(&clicked).unwrap_or_default());
                    return Ok(ActionResult::ok_note(action_name, step_index, 1, note));
                }
                let waited = started.elapsed();
                if !pick_best_should_retry(&result) || waited >= PICK_BEST_WAIT {
                    let mut v = result.clone();
                    if waited >= PICK_BEST_WAIT {
                        // Say the wait happened, so "no cards" is not read as
                        // "the page has no products".
                        let secs = waited.as_secs_f64();
                        let err = v
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("page script failed")
                            .to_string();
                        v["error"] = Value::String(format!(
                            "{err} (waited {secs:.1}s for the results to render)"
                        ));
                    }
                    return Ok(dom::result_from(v, action_name, step_index, 1));
                }
                tokio::time::sleep(PICK_BEST_POLL).await;
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
            let result = browser.eval_resilient(&expr).await?;
            if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                let text = result.get("text").and_then(Value::as_str).unwrap_or("");
                let note = format!("extracted text: {}", text);
                Ok(ActionResult::ok_note(action_name, step_index, 1, note))
            } else {
                Ok(dom::result_from(result, action_name, step_index, 1))
            }
        }
        "back" => {
            let result = browser.eval_resilient("(() => { window.history.back(); return { ok: true }; })()").await?;
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

/// Whether the click the expression actually performed may go ahead.
///
/// A purchase click that only matched *part* of the request is not the action
/// that was asked for, and it has side effects. The old matcher scored any
/// element containing "add" as a hit for "Add to Cart" — `+3` per word over
/// length 2 — and separately handed every visible `BUTTON`/`A` a flat `+2+1`
/// whether or not any word matched, so on any page with a button the "no
/// matching element" branch was unreachable. The net effect on 2026-10-03 was
/// that "Add to Wishlist" was clicked and logged as `✔ click Add to Cart`.
///
/// So: `Ok(())` to proceed, `Err(reason)` to refuse. Split out from the
/// executor so the rule can be tested without a browser, which is the only
/// reason it is a function.
///
/// Non-purchase clicks are allowed to be fuzzy. Missing a button labelled
/// "first product result" costs a wrong-but-recoverable click; buying the wrong
/// thing does not come back.
fn purchase_click_verdict(target: &str, result: &Value) -> Result<(), String> {
    if !result.get("ok").and_then(Value::as_bool).unwrap_or(false)
        || !is_purchase_click(target)
    {
        return Ok(());
    }
    if result.get("exact").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(());
    }
    let missing: Vec<&str> = result
        .get("missing")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let clicked = result
        .get("clicked")
        .and_then(Value::as_str)
        .unwrap_or("something else");
    Err(format!(
        "refused to click: asked for \"{target}\", the closest element says \
         \"{clicked}\"{} — that is a different action. Do not retry this click; \
         report what is actually on the page.",
        if missing.is_empty() {
            String::new()
        } else {
            format!(" (missing: {})", missing.join(", "))
        }
    ))
}

/// Build a search URL *for the site currently open*.
///
/// The `search` action used to be hardcoded to DuckDuckGo, which meant "search
/// for best laptop" on an Amazon plan navigated off Amazon to a web search.
/// Measured 2026-10-03: the plan was navigate-amazon, search, pick_best, and
/// pick_best correctly found zero product cards on a DuckDuckGo results page.
/// The failure looked like a scraping bug and was a navigation bug.
///
/// So `search` means "search where I already am". The search path is per-site,
/// so the current host decides the URL, and an unknown site is reported rather
/// than guessed at — guessing produces a 404 page that looks like a result page.
///
/// Pure, so it is testable: this is the decision that broke, and it must not be
/// reachable only through a live browser.
fn site_search_url(current: &str, query: &str) -> Option<String> {
    // Check the raw query: percent_encode("  ") is "%20%20", which is not
    // empty but is not a search either.
    if query.trim().is_empty() {
        return None;
    }
    let q = percent_encode(query.trim());
    let host = host_of(current)?;
    // Order matters: these are checked most-specific first, and the fallbacks
    // are deliberately last.
    let path: &str = if host.contains("amazon.") {
        "/s?k="
    } else if host.contains("ebay.") {
        "/sch/i.html?_nkw="
    } else if host.contains("flipkart.") {
        "/search?q="
    } else if host.contains("etsy.") {
        "/search?q="
    } else if host.contains("walmart.") {
        "/search?q="
    } else if host.contains("bestbuy.") {
        "/site/searchpage.jsp?st="
    } else if host.contains("target.") {
        "/s?searchTerm="
    } else if host.contains("newegg.") {
        "/search?d="
    } else if host.contains("bhphoto.") {
        "/c/search?q="
    } else {
        // Generic guess: a site with a `?q=` search is more likely than one
        // without, and the caller verifies the host afterwards.
        "/?q="
    };
    Some(format!("https://{host}{path}{q}"))
}

/// The host of a URL, lowercased, without `www.`.
fn host_of(url: &str) -> Option<String> {
    let rest = url
        .trim()
        .split_once("://")
        .map(|(_, r)| r)
        .unwrap_or_else(|| url.trim());
    let host = rest.split(['/', '?', '#']).next()?;
    let host = host.strip_prefix("www.").unwrap_or(host);
    if host.is_empty() || !host.contains('.') {
        return None;
    }
    Some(host.to_lowercase())
}

/// Did we arrive where we asked to go?
///
/// Compares hosts, not full URLs: a redirect from `amazon.com` to
/// `www.amazon.com` or a locale subdomain is still arrival, while a login wall
/// or a search engine is not.
fn arrived_at(requested: &str, actual: &str) -> bool {
    match (host_of(requested), host_of(actual)) {
        (Some(want), Some(got)) => {
            want == got || want.trim_start_matches("www.") == got.trim_start_matches("www.")
        }
        _ => false,
    }
}

/// Generate JS to find an element by description and click it.
fn find_and_click_expr(description: &str) -> String {
    let desc = serde_json::to_string(description).unwrap_or_else(|_| "\"\"".into());

    format!(
        r#"(() => {{
  const DESC = {desc};

  const candidates = Array.from(document.querySelectorAll('button,a,input,textarea,select,[role="button"],[role="link"],[role="radio"],[role="checkbox"],[role="option"],[onclick],[tabindex]'));

  const textOf = el => (el.innerText || el.textContent || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '');
  const haystack = el => [textOf(el), el.id || '', String(el.className || ''), el.getAttribute('aria-label') || ''].join(' ').toLowerCase();

  // Words that carry the request. "to", "the", "a" are length <= 2 and drop out
  // on their own; these are dropped because they appear in almost every label.
  const STOP = new Set(['the', 'and', 'for', 'with', 'button', 'link', 'page']);
  const words = DESC.toLowerCase().split(/\s+/).filter(w => w.length > 2 && !STOP.has(w));

  const hitsFor = el => {{
    const hay = haystack(el);
    return words.filter(w => hay.includes(w));
  }};

  // Only visible, on-screen elements are clickable. This ranks; it must never
  // decide that something matched.
  const visible = el => {{
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.top < window.innerHeight && r.bottom > 0;
  }};

  // Pass 1: every significant word present. Only these may be clicked without
  // disclosure, because only these are known to be the thing asked for.
  let exact = null, exactScore = -1;
  // Pass 2: the closest partial, kept for an explicitly labelled fallback.
  let partial = null, partialScore = -1, partialHits = 0;

  for (const el of candidates) {{
    const hits = hitsFor(el);
    if (!hits.length) continue;
    const rect = el.getBoundingClientRect();
    let score = 0;
    if (textOf(el).toLowerCase().includes(DESC)) score += 10;
    if (el.tagName === 'BUTTON' || el.tagName === 'A') score += 2;
    if (visible(el)) score += 1;
    if (hits.length > partialHits || (hits.length === partialHits && score > partialScore)) {{
      partial = el; partialScore = score; partialHits = hits.length;
    }}
    if (hits.length === words.length && score > exactScore) {{
      exact = el; exactScore = score;
    }}
  }}

  const best = exact || partial;
  if (!best) return {{ ok: false, error: 'no element mentions ' + words.join(' or ') + ' (asked for "' + DESC + '")' }};

  const missing = words.filter(w => !hitsFor(best).includes(w));
  const label = (textOf(best) || best.getAttribute('aria-label') || best.id || best.tagName).replace(/\s+/g, ' ').trim().slice(0, 80);

  const r = best.getBoundingClientRect();
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  const ev = {{ bubbles: true, cancelable: true, view: window, clientX: x, clientY: y }};
  best.dispatchEvent(new PointerEvent('pointerdown', ev));
  best.dispatchEvent(new MouseEvent('mousedown', ev));
  best.focus && best.focus();
  best.dispatchEvent(new PointerEvent('pointerup', ev));
  best.dispatchEvent(new MouseEvent('mouseup', ev));
  best.click && best.click();

  return {{
    ok: true,
    clicked: label,
    exact: !!exact,
    missing,
    // A purchase click that only matched some of the words is not the action
    // that was asked for, and it has side effects. The caller decides.
    purchase: /add to cart|add to bag|buy now|checkout|place order|proceed to/i.test(DESC)
  }};
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

        // Read the plan length live. Capturing it once produced logs reading
        // "Step 8/4" — re-plans splice steps in, so the denominator was stale
        // from the first repair onwards and the numbering looked broken.
        if result.success {
            let from_plan = plan.record_success();
            tracing::info!(
                "Step {}/{}: {}",
                plan.cursor_index(),
                plan.len(),
                desc
            );
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
            plan.len(),
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

    // Report the steps that were attempted and dropped. This used to be
    // silent, so a plan that gave up on "click: Add to Cart" and carried on
    // read exactly like one that had never tried it — and the tool reported
    // success. Naming the abandoned steps is the difference between "the plan
    // ran" and "the goal was met".
    for &idx in plan.abandoned() {
        let label = plan
            .step_label(idx)
            .unwrap_or_else(|| format!("step {}", idx + 1));
        tracing::warn!("gave up on {label} after {MAX_RETRIES} attempts");
        history.push(format!(
            "  ✘ gave up on {label} after {MAX_RETRIES} attempts"
        ));
    }

    if let Some(stop) = plan.stop_reason() {
        if stop != Stop::PlanFinished {
            tracing::warn!("Plan stopped early: {stop}");
            history.push(format!("  ✘ stopped: {stop}"));
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
        // step, which is what ends a one-step plan. Ending the plan by giving up
        // on its only step is not `PlanFinished`: that word means everything ran.
        plan.skip();
        assert_eq!(plan.stop_reason(), Some(Stop::RetriesExhausted));
        // Index 1, not 0: the repair step was spliced in at 0, so the abandoned
        // step is the original "Add to Cart" now sitting after it.
        assert_eq!(plan.abandoned(), &[1]);
        assert_eq!(plan.step_label(1).as_deref(), Some("click Add to Cart"));
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
        assert!(plan.abandoned().is_empty());
    }

    /// A plan that gave up on a step did not finish, whatever it reached.
    ///
    /// Both the run and the report said "Done. The browser automation (10 steps)
    /// executed" on 2026-10-03, while `click: Add to Cart` had been tried three
    /// times and dropped. The tool reported the plan completing; the user needed
    /// to hear that a step did not happen.
    #[test]
    fn reaching_the_end_after_giving_up_is_not_finishing() {
        let mut plan = LoopControl::new(vec![
            step("navigate", "amazon.com"),
            step("click", "Add to Cart"),
        ]);
        plan.record_success();
        // Three failures, because MAX_RETRIES is three. One is not a give-up.
        assert!(plan.record_failure());
        assert!(plan.record_failure());
        assert!(!plan.record_failure(), "the third failure must end the attempts");
        plan.skip();

        assert_eq!(
            plan.stop_reason(),
            Some(Stop::RetriesExhausted),
            "a plan that dropped a step must not report PlanFinished"
        );
        assert_eq!(plan.abandoned(), &[1]);
        // And the step is nameable, so the report can say which one.
        assert_eq!(
            plan.step_label(1).as_deref(),
            Some("click Add to Cart"),
            "the abandoned step cannot be named in the report"
        );
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

    /// The 2026-10-03 failure, as a URL mapping. The plan navigated to Amazon and
    /// then ran `search: best laptop`, which went to DuckDuckGo; `pick_best` then
    /// correctly found no product cards on a web search results page.
    #[test]
    fn search_stays_on_the_site_that_is_open() {
        assert_eq!(
            site_search_url("https://www.amazon.com/", "best laptop").as_deref(),
            Some("https://amazon.com/s?k=best%20laptop")
        );
        assert_eq!(
            site_search_url("https://www.amazon.co.uk/", "gaming laptop").as_deref(),
            Some("https://amazon.co.uk/s?k=gaming%20laptop")
        );
        // Whatever site is open, the search stays on that site. Searching *on*
        // google.com yields a google.com URL; the bug was that an amazon.com
        // plan yielded a duckduckgo.com one.
        for engine in [
            "https://duckduckgo.com/",
            "https://www.google.com/",
            "https://www.bing.com/",
        ] {
            let url = site_search_url(engine, "best laptop").unwrap();
            assert!(
                arrived_at(engine, &url),
                "a search on {engine} wandered off to {url}"
            );
        }
    }

    /// An unknown site still gets a search, because refusing outright would
    /// break every site not in the table.
    #[test]
    fn an_unknown_site_falls_back_to_a_query_parameter() {
        assert_eq!(
            site_search_url("https://books.tomlewand.com/catalog", "rust").as_deref(),
            Some("https://books.tomlewand.com/?q=rust")
        );
        assert_eq!(
            site_search_url("https://www.newegg.com/p/N82E16819113877", "ssd").as_deref(),
            Some("https://newegg.com/search?d=ssd")
        );
    }

    /// No site open means there is nothing to search, and the model is told how
    /// to search instead of being sent to a search engine.
    #[test]
    fn no_site_open_is_an_error_not_a_web_search() {
        assert_eq!(site_search_url("about:blank", "best laptop"), None);
        assert_eq!(site_search_url("", "best laptop"), None);
        assert_eq!(site_search_url("https://www.amazon.com/", "  "), None);
    }

    /// Arrival is checked by host, so a redirect to a login wall or a different
    /// search engine is a failure to report rather than a success to claim.
    #[test]
    fn arrival_is_verified_by_host() {
        assert!(arrived_at("https://amazon.com/s?k=x", "https://www.amazon.com/s?k=x"));
        assert!(arrived_at("https://amazon.com/s?k=x", "https://amazon.com/other/page"));
        assert!(!arrived_at("https://amazon.com/s?k=x", "https://duckduckgo.com/?q=x"));
        assert!(!arrived_at("https://amazon.com/s?k=x", "https://www.amazon.co.uk/s?k=x"));
        assert!(!arrived_at("https://amazon.com/s?k=x", "about:blank"));
    }

    /// The wait/retry decision. Retrying on "cards present but unscorable" would
    /// burn eight seconds re-reading a page that will never score.
    #[test]
    fn pick_best_retries_only_while_the_page_is_empty() {
        let empty = serde_json::json!({"ok": false, "cards_seen": 0});
        let present = serde_json::json!({"ok": false, "cards_seen": 12});
        let ok = serde_json::json!({"ok": true, "clicked": {}});
        assert!(pick_best_should_retry(&empty), "an empty page may still be rendering");
        assert!(!pick_best_should_retry(&present), "cards are there; waiting cannot help");
        assert!(!pick_best_should_retry(&ok));
    }

    /// A missing `cards_seen` must not read as "cards present" — an older or
    /// errored expression has no count, and the safe assumption is that the page
    /// might not have loaded. The deadline still bounds this.
    #[test]
    fn an_uncounted_failure_is_treated_as_maybe_still_loading() {
        let legacy = serde_json::json!({"ok": false, "error": "no scorable product cards found"});
        assert!(pick_best_should_retry(&legacy));
    }

    /// The generated expression must still be the JS that was tested.
    ///
    /// The click expression is a `format!` raw string, so every literal brace is
    /// doubled to survive. That is a silent-failure shape: a mis-escaped brace
    /// compiles, and the expression is then wrong at runtime in the browser,
    /// which is exactly where it cannot be unit tested. So it is checked here by
    /// round-tripping through the same formatter the real call uses.
    #[test]
    fn the_click_expression_survives_brace_escaping() {
        let expr = find_and_click_expr("Add to Cart");
        assert!(!expr.contains("{{"), "unescaped brace leaked into the output");
        assert!(!expr.contains("}}"), "unescaped brace leaked into the output");
        assert!(expr.contains("const DESC = \"Add to Cart\";"), "{expr}");
        // The three fields Rust reads back to decide what actually happened.
        for field in ["clicked:", "exact:", "missing,", "purchase:"] {
            assert!(expr.contains(field), "result lost `{field}`: {expr}");
        }
        // And the refusal path.
        assert!(expr.contains("ok: false"), "the refusal path is gone: {expr}");
        // The scoring gate: bonuses must rank, never create a match.
        assert!(
            expr.contains("if (!hits.length) continue;"),
            "an element with no word hit can still be chosen: {expr}"
        );
    }

    /// A description with a quote in it must not break out of the JS string.
    #[test]
    fn a_description_containing_a_quote_is_escaped() {
        let expr = find_and_click_expr(r#"the "best" laptop button"#);
        assert!(expr.contains(r#"\"best\""#), "{expr}");
        assert!(!expr.contains(r#""the "best""#), "unescaped quote: {expr}");
    }

    // --- what a click is allowed to get away with -------------------------

    fn clicked(el: &str, exact: bool, missing: &[&str]) -> Value {
        serde_json::json!({
            "ok": true, "clicked": el, "exact": exact, "missing": missing
        })
    }

    /// The 2026-10-03 outcome, pinned.
    ///
    /// "Add to Cart" on a page whose only "add" element was "Add to Wishlist".
    /// It ran, and it was logged as a success, and nothing entered any cart.
    #[test]
    fn a_near_miss_is_not_a_purchase() {
        let r = clicked("Add to Wishlist", false, &["cart"]);
        let why = purchase_click_verdict("Add to Cart", &r).unwrap_err();
        assert!(why.contains("Add to Wishlist"), "{why}");
        assert!(why.contains("missing: cart"), "{why}");
        // It must say "refused", or a retry loop reads it as a soft failure.
        assert!(why.to_lowercase().contains("refused"), "{why}");
    }

    /// The exact match is allowed through. A rule that refuses this is useless.
    #[test]
    fn the_real_button_is_allowed_through() {
        assert!(purchase_click_verdict("Add to Cart", &clicked("Add to Cart", true, &[])).is_ok());
    }

    /// A fuzzy *non*-purchase click is allowed, and the asymmetry is deliberate.
    #[test]
    fn a_fuzzy_non_purchase_click_is_allowed() {
        let r = clicked("Product result 1 of 48", false, &["first"]);
        assert!(purchase_click_verdict("the first product result", &r).is_ok());
    }

    /// "Add to Wishlist" is a purchase by this table, so a near-miss on it is
    /// refused too — it changes the user's account, not just their basket.
    #[test]
    fn wishlist_is_treated_as_a_purchase() {
        let r = clicked("Add to List", false, &["wishlist"]);
        assert!(purchase_click_verdict("Add to Wishlist", &r).is_err());
    }

    /// A failed expression is the click path's own error, not a fuzzy match.
    #[test]
    fn a_refused_expression_is_not_also_a_fuzzy_purchase() {
        let r = serde_json::json!({"ok": false, "error": "no element mentions cart"});
        // Otherwise the caller would report "refused to click" over an error
        // that already explains itself, hiding the real cause.
        assert!(purchase_click_verdict("Add to Cart", &r).is_ok());
    }

    /// A result with no `exact` field is an *older* expression, not an exact one.
    ///
    /// `unwrap_or(false)` means an unfamiliar shape is treated as inexact, which
    /// for a purchase means refused. That is the right direction: the cost of
    /// wrongly refusing is a retry loop against a page that will not resolve,
    /// and the cost of wrongly allowing is an irreversible purchase.
    #[test]
    fn a_result_with_no_match_fields_is_refused_not_assumed_exact() {
        let r = serde_json::json!({"ok": true});
        assert!(purchase_click_verdict("Add to Cart", &r).is_err());
    }

    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("ps4 on amazon.com"), "ps4%20on%20amazon.com");
        assert_eq!(percent_encode("add to cart & buy 'now'"), "add%20to%20cart%20%26%20buy%20%27now%27");
    }
}