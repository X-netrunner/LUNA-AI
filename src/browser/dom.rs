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

  // A field is only read from a SHORT string. Every misparse that motivated
  // this came from parsing a long blob: the star rating came out as 314 because
  // it was read out of a product name ("Chromebook 314"), and the review count
  // came out as 4 because it was read out of a spec line ("4GB LPDDR4X"). A
  // rating is at most a dozen characters, so a long string is not a rating.
  const SHORT = 40;
  const brief = el => {
    if (!el) return '';
    const t = clean(el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '');
    return t.length <= SHORT ? t : '';
  };

  // Stars, anchored to Amazon's own wording and range-checked.
  const starsOf = card => {
    const el = card.querySelector('[aria-label*="out of 5 stars" i], [title*="out of 5 stars" i]');
    const t = clean(el?.getAttribute('aria-label') || el?.getAttribute('title') || el?.textContent || '');
    const m = t.match(/([\d.]+)\s*out of\s*5/i);
    if (!m) return null;
    const v = parseFloat(m[1]);
    // A rating outside 0..5 was never a rating. Accepting it is how 314 got in.
    return Number.isFinite(v) && v >= 0 && v <= 5 ? v : null;
  };

  // Reviews. Deliberately no bare `.a-size-base` fallback: that class is used
  // for every piece of small text on the card, which is how "4GB" became a
  // review count.
  const reviewsOf = card => {
    const el = card.querySelector('[aria-label*="rating" i], [aria-label*="review" i], a[href*="customerReviews"]');
    const t = brief(el);
    if (!t) return null;
    const m = t.match(/([\d][\d,]*)/);
    if (!m) return null;
    const v = parseFloat(m[1].replace(/,/g, ''));
    return Number.isFinite(v) && v >= 0 && v < 1e8 ? Math.round(v) : null;
  };

  // Price, preferring the offscreen text, which is the full formatted price.
  const priceOf = card => {
    const off = card.querySelector('.a-price .a-offscreen, [data-a-color="base"] .a-offscreen, .a-offscreen');
    let t = brief(off);
    if (!t) {
      const whole = brief(card.querySelector('.a-price-whole'));
      const frac = brief(card.querySelector('.a-price-fraction'));
      t = whole + (frac ? '.' + frac : '');
    }
    if (!t) return null;
    const v = parseNum(t);
    return Number.isFinite(v) && v > 0 && v < 1e7 ? v : null;
  };

  const cards = [...document.querySelectorAll('[data-component-type="s-search-result"], .sg-col-20of24 .a-cardui, .product-card, [data-asin], [data-testid="product-card"], .a-section.a-spacing-base')];
  let best = null, bestScore = -Infinity, bestLink = null;
  let withStars = 0, withReviews = 0, withPrice = 0, sponsored = 0, rejected = 0;

  for (const card of cards) {
    const titleEl = card.querySelector('h2 a, h3 a, .a-text-bold a, [data-cy="title-recipe"] a, h2, h3, .a-text-bold, [data-cy="title-recipe"]');
    const title = clean(titleEl?.textContent || '');
    if (!title) continue;

    // A sponsored card is an ad, not the best product. It is also where the
    // 2026-10-03 run went wrong: it clicked a "Sponsored" Celeron Chromebook
    // and called it the best laptop.
    const marker = brief(card.querySelector('[aria-label*="Sponsored" i], .puis-sponsored-label-text, .s-sponsored-label-text')) || clean(card.textContent || '').slice(0, 40);
    if (/^\s*sponsored/i.test(marker)) { sponsored++; continue; }

    const stars = starsOf(card);
    if (stars != null) withStars++;
    const reviews = reviewsOf(card);
    if (reviews != null) withReviews++;
    const price = priceOf(card);
    if (price != null) withPrice++;

    if (stars == null || reviews == null || price == null) { rejected++; continue; }

    // An unreadable price must NOT become 1: that scores above every real price
    // on the page, so a single unpriced card would win outright.
    const score = stars * Math.log(reviews + 1) / price;
    const link = titleEl?.tagName === 'A' ? titleEl : card.querySelector('a[href*="/dp/"], a[href*="/gp/product/"]');
    if (!link) { rejected++; continue; }
    if (score > bestScore) {
      bestScore = score;
      best = { title, stars, reviews, price, score: Math.round(score * 1000) / 1000 };
      bestLink = link;
    }
  }

  if (!bestLink) {
    if (!cards.length) {
      return { ok: false, error: 'no product cards matched the selectors on this page - the layout may have changed, or the page has not finished loading', cards_seen: 0, with_stars: 0, with_reviews: 0, with_prices: 0 };
    }
    const why = [];
    if (withStars < cards.length) why.push((cards.length - withStars) + ' had no readable star rating');
    if (withReviews < cards.length) why.push((cards.length - withReviews) + ' had no review count');
    if (withPrice < cards.length) why.push((cards.length - withPrice) + ' had no readable price');
    if (sponsored) why.push(sponsored + ' were sponsored');
    return { ok: false, error: 'found ' + cards.length + ' product cards but could not score any: ' + why.join(', '), cards_seen: cards.length, with_stars: withStars, with_reviews: withReviews, with_prices: withPrice, sponsored };
  }

  const r = bestLink.getBoundingClientRect();
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  const ev = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
  bestLink.dispatchEvent(new PointerEvent('pointerdown', ev));
  bestLink.dispatchEvent(new MouseEvent('mousedown', ev));
  bestLink.focus && bestLink.focus();
  bestLink.dispatchEvent(new PointerEvent('pointerup', ev));
  bestLink.dispatchEvent(new MouseEvent('mouseup', ev));
  bestLink.click && bestLink.click();
  return { ok: true, clicked: best, considered: cards.length, sponsored };
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