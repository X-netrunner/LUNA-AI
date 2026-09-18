//! DOM interaction expressions — mirrors the extension's content script logic.

use crate::browser::types::ActionResult;
use serde_json::Value;

/// Type text into the currently focused or best-guess input element.
pub fn dom_type_expr(text: &str) -> String {
    let t = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into());
    format!(
        r#"(() => {{
  const isInput = (e) =>
    e instanceof HTMLInputElement || e instanceof HTMLTextAreaElement ||
    (e instanceof HTMLElement && e.isContentEditable);
  let el = document.activeElement;
  if (!(el && isInput(el))) {{
    el = document.querySelector(
      'input[type="search"],input[type="text"],textarea[name="q"],textarea[aria-label*="Search" i],input[name="q"],input[aria-label*="Search" i],textarea,input:not([type="hidden"])'
    );
  }}
  if (!el || !isInput(el)) return {{ ok: false, error: "no supported input found" }};
  el.focus();
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {{
    const d = Object.getOwnPropertyDescriptor(el.constructor.prototype, "value");
    const setter = d && d.set ? d.set : null;
    if (setter) setter.call(el, {t}); else el.value = {t};
    el.dispatchEvent(new Event("input", {{ bubbles: true, composed: true }}));
    el.dispatchEvent(new Event("change", {{ bubbles: true, composed: true }}));
    return {{ ok: true }};
  }}
  el.textContent = {t};
  el.dispatchEvent(new InputEvent("input", {{ bubbles: true, inputType: "insertText", data: {t} }}));
  return {{ ok: true }};
}})()"#
    )
}

/// Press a key (Enter, Tab, Escape, etc.) on the active element.
pub fn dom_press_expr(key: &str) -> String {
    let k = serde_json::to_string(key).unwrap_or_else(|_| "\"Enter\"".into());
    format!(
        r#"(() => {{
  const K = {k};
  const t = document.activeElement instanceof HTMLElement ? document.activeElement : document.body;
  t.dispatchEvent(new KeyboardEvent("keydown", {{ key: K, bubbles: true, cancelable: true }}));
  t.dispatchEvent(new KeyboardEvent("keyup", {{ key: K, bubbles: true, cancelable: true }}));
  if (K === "Enter" && document.activeElement) {{
    const form = document.activeElement.closest ? document.activeElement.closest("form") : null;
    if (form) {{ if (typeof form.requestSubmit === "function") form.requestSubmit(); else form.submit(); }}
  }}
  return {{ ok: true }};
}})()"#
    )
}

/// Scroll the page up or down.
pub fn dom_scroll_expr(direction: &str, amount: i64) -> String {
    let delta = if direction == "down" { amount } else { -amount };
    format!(
        r#"(() => {{ window.scrollBy({{ top: {delta}, behavior: "auto" }}); return {{ ok: true }}; }})()"#
    )
}

/// Execute a DOM action and return a normalized result.
pub fn result_from(v: Value, action: String, step_index: u64, tab_id: u64) -> ActionResult {
    match v.get("ok").and_then(Value::as_bool) {
        Some(true) => ActionResult::ok(action, step_index, tab_id),
        _ => ActionResult::fail(action, step_index, tab_id, v.get("error").and_then(Value::as_str).unwrap_or("page script failed").to_string()),
    }
}