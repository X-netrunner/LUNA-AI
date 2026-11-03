# Model tiers for a 6 GB VRAM / 16 GB RAM laptop (RTX 4050)

Rule that decides everything: weights that do not fit in VRAM cross the PCIe bus on every token (6-10 tok/s instead of 45-55). Luna's tool loop pays that cost several times per request.

| Tier (luna.toml key) | Model | Size | Fits VRAM | Status |
|---|---|---|---|---|
| fast_model | phi4-mini | ~2.5 GB | yes (~80 t/s) | applied (chat only, never sees tools) |
| model (orchestrator) | ToolACE-8B Q4_K_M | ~4.9 GB | yes | candidate, verify `tools` capability first |
| deep_model | qwen3.5:9b | ~6 GB | borderline | unchanged, measured best in this repo |
| deep_model_unrestricted | qwen2.5-coder 14B abliterated | ~9 GB | no, hybrid | unchanged (already set to the huihui_ai build) |
| security_model | whiterabbitneo-coder-tools | - | - | unchanged |
| security_model_abliterated (no-refusal gate) | qwythos-9b-abliterated or Foundation-Sec-8B | 5.6 / 4.9 GB | tight / yes | candidate, MUST list `tools` |

## Why some suggestions are candidates, not defaults
- The security tier needs a model that can call tools. A completion-only model makes Luna describe scans she never ran. Check with `ollama show <model> | grep -A3 Capabilities`.
- The repo's own measurement (2026-10-04, same card) had qwen2.5-coder:14b at 4/8 correct scripts, 6 tok/s, versus qwen3.5:9b at 5/8, 12.8 tok/s. A 14B coder does not beat it here.
- Raw GGUFs from Hugging Face often ship without a tool-calling template, so Ollama reports no `tools` capability even if the base model supports it.

## Hybrid offload note
Only models that do not need tool loops (one-shot code generation) are reasonable to run hybrid.


## The real fix for routing misses (not done here, worth doing next)

`escalation.rs` routes by keyword match, so it will always miss some phrasing — "my widgets don't show up"
had to be added by hand once it was noticed, and the newer batch (frozen, laggy, missing, etc.) is still a
hand-picked list of the obvious ones, not a general fix. The sturdier design is to invert the default: send
everything except greetings and short chit-chat to the tool model, and only use the fast model for a small,
explicit allow-list (greetings, simple factual questions with no system/file content). That would end the
"small model says I can't" pattern at its root instead of patching individual phrases, at the cost of the fast
model handling fewer turns (more turns pay its latency to reach the tool model). Flagging this rather than doing
it silently: it's a bigger behavioral change than a keyword addition and changes how often the fast tier runs
at all.
