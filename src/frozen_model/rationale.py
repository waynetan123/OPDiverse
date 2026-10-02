"""distill-self: surface validity, training targets, and the substitution trigger.

A rationale is surface-valid when all of these hold (pinned in docs/decisions/step5_decision_record.md):
- it finished on a stop token (not cut off at RATIONALE_MAX_TOKENS);
- its last non-empty line, read by the item's lenient parser, is exactly the gold answer;
- there is reasoning above that line;
- the reasoning does not restate the hint (pinned.HINT_LEAK).
The training target is that reasoning followed by the canonical gold target. Invalid attempt-1
rationales are regenerated once; a second failure falls back to the gold target alone.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from etl import pinned, verifiers

SOURCES = ("self_a1", "self_a2", "gold_only")
REASONS = ("truncated", "empty", "no_final_answer", "wrong_final_answer", "no_reasoning", "hint_leak")


def split_final_line(reply: str) -> tuple[str, str]:
    """(everything above the last non-empty line, that line), both stripped."""
    lines = reply.rstrip().split("\n")
    return "\n".join(lines[:-1]).strip(), lines[-1].strip()


def hint_leak(text: str) -> bool:
    return any(p.search(text) for p in pinned.HINT_LEAK)


def validate(item: dict, reply: str, finish_reason: str, graph: pinned.CweLookup) -> tuple[str | None, str | None]:
    """(training target, None) for a surface-valid rationale, else (None, reason)."""
    if finish_reason != "stop":
        return None, "truncated"
    body, last = split_final_line(reply)
    if not last:
        return None, "empty"
    verdict = verifiers.verify_item(item["type"], last, item["gold"], graph)
    if not verdict.parse_ok:
        return None, "no_final_answer"
    if verdict.parsed != verifiers.verify_item(item["type"], item["target"], item["gold"], graph).parsed:
        return None, "wrong_final_answer"
    if not body:
        return None, "no_reasoning"
    if hint_leak(body):
        return None, "hint_leak"
    return f"{body}\n\n{item['target']}", None


def decide(item: dict, first: dict, retry: dict | None, graph: pinned.CweLookup) -> dict:
    """One distill_self.jsonl row. `first` / `retry` are the attempts' first outputs ({text, finish_reason})."""
    target, reason1 = validate(item, first["text"], first["finish_reason"], graph)
    row = {"item_id": item["item_id"], "cve_id": item["cve_id"], "type": item["type"], "index": item["index"],
           "a1_reason": reason1, "a1_hint_leak": hint_leak(first["text"]), "a2_reason": None}
    if target is not None:
        return {**row, "source": "self_a1", "target": target}
    if retry is None:
        raise SystemExit(f"{item['item_id']}: attempt 1 is invalid ({reason1}) and has no regeneration; "
                         "run `python -m frozen_model prepare-retry` and the GPU runner on the retry file")
    target, reason2 = validate(item, retry["text"], retry["finish_reason"], graph)
    if target is not None:
        return {**row, "source": "self_a2", "target": target}
    return {**row, "a2_reason": reason2, "source": "gold_only", "target": item["target"]}


def band(rate: Fraction) -> str:
    """The plan's per-arm decision bands, applied to the first-attempt pass rate."""
    if rate >= pinned.AUDIT_LIVE:
        return "as_specified"
    return "report_quality" if rate >= pinned.AUDIT_FLOOR else "floored"


def summarise(rows: list[dict]) -> dict:
    """Per type: pass rate (first attempt), fallback rate, hint-leak rate, reasons, and the trigger."""
    types = {}
    for t in pinned.BANK_TYPES:
        chosen = [r for r in rows if r["type"] == t]
        n = len(chosen)
        if not n:
            continue
        sources = Counter(r["source"] for r in chosen)
        passed = Fraction(sources["self_a1"], n)
        fallback = Fraction(sources["gold_only"], n)
        fires = passed < pinned.SUBSTITUTION_MAX or fallback > pinned.SUBSTITUTION_MAX
        types[t] = {
            "n": n,
            "sources": {s: sources[s] for s in SOURCES},
            "pass_rate_a1": str(passed),
            "fallback_rate": str(fallback),
            "hint_leak_rate_a1": str(Fraction(sum(r["a1_hint_leak"] for r in chosen), n)),
            "a1_reasons": dict(sorted(Counter(r["a1_reason"] for r in chosen if r["a1_reason"]).items())),
            "a2_reasons": dict(sorted(Counter(r["a2_reason"] for r in chosen if r["a2_reason"]).items())),
            "band": band(passed),
            "substitution_fires": fires,
        }
    substituted = [t for t, s in types.items() if s["substitution_fires"]]
    return {
        "types": types,
        "substituted_types": substituted,
        "drop_distill_self_from_primary_test": len(substituted) >= pinned.SUBSTITUTION_DROP_AT,
        "rule": {"pass_rate_below": str(pinned.SUBSTITUTION_MAX), "or_fallback_rate_above": str(pinned.SUBSTITUTION_MAX),
                 "drop_at_types": pinned.SUBSTITUTION_DROP_AT, "pass_rate_reading": "first attempt"},
    }
