//! DOM interaction expressions — mirrors the extension's content script logic.

use crate::browser::types::ActionResult;
use serde_json::Value;

/// Extract visible form fields as JSON for the planner to generate values.
/// Returns: { ok: true, fields: [{ index, kind, label, question, option, options }] }
/// kind ∈ { "text", "textarea", "select", "radio", "checkbox" }
pub fn dom_form_schema_expr() -> &'static str {
    r#"(() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim().slice(0, 140);
  const visible = el => { const r = el.getBoundingClientRect(); return r.width > 1 && r.height > 1; };
  const questionOf = el => {
    // 1. Try aria-labelledby (Google Forms uses this to point to question title)
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const ids = lb.split(/\s+/);
      for (const id of ids) {
        const target = document.getElementById(id);
        if (target && target.innerText) return clean(target.innerText);
      }
    }
    // 2. Climb to Google Forms item container and find question title
    let p = el.parentElement, hops = 0;
    while (p && hops < 12) {
      // Google Forms item container
      if (p.matches && (p.matches('.freebirdFormviewerViewItemsItemItem') || p.matches('[role="listitem"]'))) {
        const q = p.querySelector('.freebirdFormviewerViewItemsItemItemTitle, .M7eMe, [role="heading"], .Qr7Oae');
        if (q && q.innerText) return clean(q.innerText);
      }
      // Generic heading
      const h = p.querySelector('[role="heading"], .M7eMe, .freebirdFormviewerViewItemsItemItemTitle, .Qr7Oae [role="heading"]');
      if (h && h.innerText && h !== el) return clean(h.innerText);
      p = p.parentElement; hops++;
    }
    // 3. Fallback: nearest preceding heading-like element
    let prev = el.previousElementSibling;
    while (prev) {
      if (prev.matches && (prev.matches('[role="heading"]') || prev.matches('.M7eMe') || prev.matches('.freebirdFormviewerViewItemsItemItemTitle'))) {
        if (prev.innerText) return clean(prev.innerText);
      }
      prev = prev.previousElementSibling;
    }
    return '';
  };
  const labelOf = el => clean(el.getAttribute('aria-label') || '');
  const out = [];
  document.querySelectorAll('input,textarea,select').forEach(el => {
    const t = (el.type || '').toLowerCase();
    if (['hidden','submit','button','image','reset','file','password','search'].includes(t)) return;
    if (!visible(el)) return;
    out.push({ kind: el.tagName === 'TEXTAREA' ? 'textarea' : (el.tagName === 'SELECT' ? 'select' : (t || 'text')),
               question: questionOf(el), option: null, label: labelOf(el),
               options: el.tagName === 'SELECT' ? [...el.options].map(o => clean(o.text)) : null });
  });
  document.querySelectorAll('[role="radio"]').forEach(el => {
    if (!visible(el)) return;
    out.push({ kind: 'radio', question: questionOf(el), option: labelOf(el), label: labelOf(el), options: null });
  });
  document.querySelectorAll('[role="checkbox"]').forEach(el => {
    if (!visible(el)) return;
    out.push({ kind: 'checkbox', question: questionOf(el), option: labelOf(el), label: labelOf(el), options: null });
  });
  return { ok: true, fields: out.map((f,i) => ({ ...f, index: i })) };
})()"#
}

/// Fill form fields given a values array from the planner.
/// values = [{index, value?, choose?}] where choose is string|string[] for radio/checkbox
pub fn dom_form_fill_expr(values_json: &str) -> String {
    let template = r#"(() => {
  const values = __VALUES__;
  const clean = s => (s || '').replace(/\s+/g, ' ').trim().slice(0, 140);
  const visible = el => { const r = el.getBoundingClientRect(); return r.width > 1 && r.height > 1; };
  const questionOf = el => {
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const ids = lb.split(/\s+/);
      for (const id of ids) {
        const target = document.getElementById(id);
        if (target && target.innerText) return clean(target.innerText);
      }
    }
    let p = el.parentElement, hops = 0;
    while (p && hops < 12) {
      if (p.matches && (p.matches('.freebirdFormviewerViewItemsItemItem') || p.matches('[role="listitem"]'))) {
        const q = p.querySelector('.freebirdFormviewerViewItemsItemItemTitle, .M7eMe, [role="heading"], .Qr7Oae');
        if (q && q.innerText) return clean(q.innerText);
      }
      const h = p.querySelector('[role="heading"], .M7eMe, .freebirdFormviewerViewItemsItemItemTitle, .Qr7Oae [role="heading"]');
      if (h && h.innerText && h !== el) return clean(h.innerText);
      p = p.parentElement; hops++;
    }
    let prev = el.previousElementSibling;
    while (prev) {
      if (prev.matches && (prev.matches('[role="heading"]') || prev.matches('.M7eMe') || prev.matches('.freebirdFormviewerViewItemsItemItemTitle'))) {
        if (prev.innerText) return clean(prev.innerText);
      }
      prev = prev.previousElementSibling;
    }
    return '';
  };
  const labelOf = el => clean(el.getAttribute('aria-label') || '');
  const controls = [];
  document.querySelectorAll('input,textarea,select').forEach(el => {
    const t = (el.type || '').toLowerCase();
    if (['hidden','submit','button','image','reset','file','password','search'].includes(t)) return;
    if (!visible(el)) return;
    controls.push({ el, kind: el.tagName === 'TEXTAREA' ? 'textarea' : (el.tagName === 'SELECT' ? 'select' : (t || 'text')), question: questionOf(el), option: null, label: labelOf(el) });
  });
  document.querySelectorAll('[role="radio"]').forEach(el => { if (visible(el)) controls.push({ el, kind: 'radio', question: questionOf(el), option: labelOf(el), label: labelOf(el) }); });
  document.querySelectorAll('[role="checkbox"]').forEach(el => { if (visible(el)) controls.push({ el, kind: 'checkbox', question: questionOf(el), option: labelOf(el), label: labelOf(el) }); });
  const setNative = (el, val) => {
    const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : (el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype);
    const d = Object.getOwnPropertyDescriptor(proto, 'value');
    const setter = d && d.set ? d.set : null;
    if (setter) setter.call(el, val); else el.value = val;
    el.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    el.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
  };
  const fireClick = (el) => {
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const ev = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
    el.dispatchEvent(new PointerEvent('pointerdown', ev));
    el.dispatchEvent(new MouseEvent('mousedown', ev));
    el.focus && el.focus();
    el.dispatchEvent(new PointerEvent('pointerup', ev));
    el.dispatchEvent(new MouseEvent('mouseup', ev));
    el.click && el.click();
  };
  const report = [];
  values.forEach(v => {
    const ctrl = controls[v.index];
    if (!ctrl) return;
    if (ctrl.kind === 'radio' || ctrl.kind === 'checkbox') {
      const want = Array.isArray(v.choose) ? v.choose : [v.choose].filter(Boolean);
      if (want.length === 0 || want.some(w => String(ctrl.option).toLowerCase().includes(String(w).toLowerCase()))) {
        fireClick(ctrl.el);
        report.push({ index: v.index, kind: ctrl.kind, question: ctrl.question, option: ctrl.option });
      }
    } else if (ctrl.kind === 'select') {
      const opt = [...ctrl.el.options].find(o => o.text.trim().toLowerCase() === String(v.value || '').toLowerCase())
                 || [...ctrl.el.options].find(o => o.text.toLowerCase().includes(String(v.value || '').toLowerCase()));
      if (opt) { setNative(ctrl.el, opt.value); report.push({ index: v.index, kind: 'select', question: ctrl.question, value: opt.text }); }
    } else {
      setNative(ctrl.el, v.value ?? '');
      report.push({ index: v.index, kind: ctrl.kind, question: ctrl.question, value: v.value ?? '' });
    }
  });
  return { ok: true, filled: report };
})()"#;
    template.replace("__VALUES__", values_json)
}

/// Scrape product cards, pick the best by score = stars * ln(reviews+1) / price, and click its title/link to open the product detail page.
/// Returns { ok: true, clicked: {title, stars, reviews, price, score} } or { ok: false, error }
pub fn dom_pick_best_expr() -> &'static str {
    r#"(() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const parseNum = s => {
    const m = (s || '').match(/([\d,]+\.?\d*)/);
    return m ? parseFloat(m[1].replace(/,/g, '')) : null;
  };
  const cards = [...document.querySelectorAll('[data-component-type="s-search-result"], .sg-col-20of24 .a-cardui, .product-card, [data-asin], .a-section.a-spacing-base')];
  let best = null, bestScore = -Infinity, bestLink = null;
  for (const card of cards) {
    // Title element (usually a link)
    const titleEl = card.querySelector('h2 a, h3 a, .a-text-bold a, [data-cy="title-recipe"] a, h2, h3, .a-text-bold, [data-cy="title-recipe"]');
    let title = clean(titleEl?.textContent || '');
    if (!title) continue;
    // Stars (aria-label like "4.5 out of 5 stars")
    const starsEl = card.querySelector('[aria-label*="out of 5 stars" i], [aria-label*="star" i], .a-icon-alt');
    const starsText = starsEl?.getAttribute('aria-label') || starsEl?.textContent || '';
    const stars = parseNum(starsText);
    // Reviews (aria-label like "12,345 ratings")
    const revEl = card.querySelector('[aria-label*="rating" i], [aria-label*="review" i], .a-size-base.a-link-normal');
    const revText = revEl?.getAttribute('aria-label') || revEl?.textContent || '';
    const reviews = parseNum(revText);
    // Price (whole + fraction)
    const priceWhole = card.querySelector('.a-price-whole, [data-a-color="price"] .a-offscreen')?.textContent || '';
    const priceFrac = card.querySelector('.a-price-fraction')?.textContent || '';
    const priceText = priceWhole + (priceFrac ? '.' + priceFrac : '');
    const price = parseNum(priceText) || 1;
    if (stars == null || reviews == null) continue;
    const score = stars * Math.log(reviews + 1) / price;
    // Find clickable link for the product
    const link = titleEl?.tagName === 'A' ? titleEl : card.querySelector('a[href*="/dp/"], a[href*="/gp/product/"], a.a-link-normal');
    if (!link) continue;
    if (score > bestScore) { bestScore = score; best = { title, stars, reviews, price, score }; bestLink = link; }
  }
  if (!bestLink) return { ok: false, error: "no scorable product cards found" };
  // Click the product link to open detail page
  const r = bestLink.getBoundingClientRect();
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  const ev = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
  bestLink.dispatchEvent(new PointerEvent('pointerdown', ev));
  bestLink.dispatchEvent(new MouseEvent('mousedown', ev));
  bestLink.focus && bestLink.focus();
  bestLink.dispatchEvent(new PointerEvent('pointerup', ev));
  bestLink.dispatchEvent(new MouseEvent('mouseup', ev));
  bestLink.click && bestLink.click();
  return { ok: true, clicked: best };
})()"#
}

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

/// Extract visible text from a specific selector or the body.
pub fn dom_extract_text_expr(selector: &str) -> String {
    let s = serde_json::to_string(selector).unwrap_or_else(|_| "\"body\"".into());
    format!(
        r#"(() => {{
  const sel = {s};
  let el = document.querySelector(sel);
  if (!el && sel !== "body") el = document.body;
  if (!el) return {{ ok: false, error: "element not found" }};
  const text = (el.innerText || el.textContent || "").replace(/\s+/g, ' ').trim().slice(0, 1000);
  return {{ ok: true, text }};
}})()"#
    )
}

/// Execute a DOM action and return a normalized result.
pub fn result_from(v: Value, action: String, step_index: u64, tab_id: u64) -> ActionResult {
    match v.get("ok").and_then(Value::as_bool) {
        Some(true) => ActionResult::ok(action, step_index, tab_id),
        _ => ActionResult::fail(action, step_index, tab_id, v.get("error").and_then(Value::as_str).unwrap_or("page script failed").to_string()),
    }
}