"""distill-external: surface validity of the external model's written-out reasoning, and training targets.

A trace is valid when all of these hold, checked in this order (pinned in docs/decisions/step6_decision_record.md):
- the reply ended on its own (stop_reason end_turn: not refused, not cut off);
- it has a last non-empty line;
- that line parses under the item's lenient verifier, and its canonical form is a complete
  target-format answer (strict verifier): CVSS with all eight metrics, "VULNERABLE: yes" with a CWE;
- there is reasoning above that line;
- reasoning, a blank line and the canonical answer fit TRACE_MAX_TOKENS with the stop token.
The answer may be wrong: no verifier selects distill data, so correctness is recorded, never used.
An invalid first attempt is regenerated once; a second failure falls back to the gold target alone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from fractions import Fraction

from etl import pinned, verifiers
from etl.census import quantiles
from frozen_model.rationale import band, split_final_line

from .results import stop_status

SOURCES = ("ext_a1", "ext_a2", "gold_only")
REASONS = ("refusal", "max_tokens", "empty", "no_final_answer", "incomplete_answer", "no_reasoning", "too_long")


def canonical_answer(item_type: str, parsed: str) -> str:
    """Verdict.parsed -> the target format. Only MCQ differs: the verifier parses a bare letter."""
    return f"ANSWER: {parsed}" if item_type == "mcq" else parsed


def validate(item: dict, row: dict, count: Callable[[str], int], graph: pinned.CweLookup) -> tuple[dict | None, str | None]:
    """({target, answer, correct, dense, target_tokens}, None) for a valid trace, else (None, reason)."""
    status = stop_status(row)
    if status != "ok":
        return None, status
    body, last = split_final_line(row["text"] or "")
    if not last:
        return None, "empty"
    verdict = verifiers.verify_item(item["type"], last, item["gold"], graph)
    if not verdict.parse_ok:
        return None, "no_final_answer"
    answer = canonical_answer(item["type"], verdict.parsed)
    scored = verifiers.verify_item(item["type"], answer, item["gold"], graph)
    if not scored.strict_ok:
        return None, "incomplete_answer"
    if not body:
        return None, "no_reasoning"
    target = f"{body}\n\n{answer}"
    n = count(target)
    if n + 1 > pinned.TRACE_MAX_TOKENS:
        return None, "too_long"
    return {"target": target, "answer": answer, "correct": scored.metric == 1, "dense": str(scored.dense),
            "target_tokens": n}, None


def decide(item: dict, first: dict, retry: dict | None, count: Callable[[str], int], graph: pinned.CweLookup) -> dict:
    """One distill_external.jsonl row. `first` / `retry` are the runner's generation rows for attempts 1 and 2."""
    row = {"item_id": item["item_id"], "cve_id": item["cve_id"], "type": item["type"], "index": item["index"],
           "a1_stop_category": first.get("stop_category"), "a2_reason": None, "a2_stop_category": None}
    valid, reason1 = validate(item, first, count, graph)
    row["a1_reason"] = reason1
    if valid is not None:
        return {**row, "source": "ext_a1", **valid}
    if retry is None:
        raise SystemExit(f"{item['item_id']}: the first trace is invalid ({reason1}) and has no regeneration; "
                         "run `python -m generators teacher prepare-retry` and the external runner on the retry file")
    valid, reason2 = validate(item, retry, count, graph)
    row["a2_stop_category"] = retry.get("stop_category")
    if valid is not None:
        return {**row, "source": "ext_a2", **valid}
    return {**row, "a2_reason": reason2, "source": "gold_only", "target": item["target"], "answer": None,
            "correct": None, "dense": None, "target_tokens": None}


def summarise(rows: list[dict]) -> dict:
    """Per type: sources, first-attempt validity, reasons, refusal categories, how often the teacher reaches
    the correct answer (the plan's distill-external audit line, banded), its dense scores and target lengths."""
    types = {}
    for t in pinned.BANK_TYPES:
        chosen = [r for r in rows if r["type"] == t]
        n = len(chosen)
        if not n:
            continue
        sources = Counter(r["source"] for r in chosen)
        correct = Fraction(sum(bool(r["correct"]) for r in chosen), n)
        answered = [r for r in chosen if r["answer"] is not None]
        types[t] = {
            "n": n,
            "sources": {s: sources[s] for s in SOURCES},
            "valid_rate_a1": str(Fraction(sources["ext_a1"], n)),
            "fallback_rate": str(Fraction(sources["gold_only"], n)),
            "correct_rate": str(correct),
            "band": band(correct),
            "a1_reasons": dict(sorted(Counter(r["a1_reason"] for r in chosen if r["a1_reason"]).items())),
            "a2_reasons": dict(sorted(Counter(r["a2_reason"] for r in chosen if r["a2_reason"]).items())),
            "refusal_categories": dict(sorted(Counter(
                str(r[f"{a}_stop_category"]) for r in chosen for a in ("a1", "a2")
                if r[f"{a}_reason"] == "refusal").items())),
            "dense_quantiles": quantiles(float(Fraction(r["dense"])) for r in answered),
            "target_tokens_quantiles": quantiles(r["target_tokens"] for r in answered),
        }
    return {"types": types, "rule": {"max_target_tokens": pinned.TRACE_MAX_TOKENS - 1, "words": pinned.TRACE_WORDS,
                                     "correct_is": "metric == 1 on a valid trace; gold-only rows count as not correct"}}
