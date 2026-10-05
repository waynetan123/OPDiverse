"""DPO rejected answers: the external model picks a near miss, the rules admit it, a rule-built near miss fills in.

Chosen is the item's gold target. Rejected is one unit of error from gold, in the same canonical format
(pinned in docs/decisions/step6_decision_record.md):
- exact_id: a live weakness one ChildOf step from gold (a parent or a child);
- cvss: gold with exactly one of the eight components changed;
- find_error 0 (vulnerable): "VULNERABLE: yes" with a CWE one ChildOf step from gold;
- find_error 1 (patched): "VULNERABLE: yes" with any live weakness;
- line_loc: gold with one line left out (only if gold has two or more), or one line replaced by a code
  line at least 2 away from it; F1 under the pinned matcher must stay below 1;
- mcq: no request. The distractor closest to gold by undirected ChildOf distance, ties to the external
  model's step-4 order.
Every rejection must also parse strictly and score below 1 under its verifier. An invalid first proposal
is requested once more; a second failure is replaced by the rule-built near miss.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction

from etl import pinned, verifiers
from etl.cwe_graph import CweGraph
from frozen_model.rationale import band

from ..bank.items import line_target
from ..bank.report import distance
from .results import stop_status

SOURCES = ("model_a1", "model_a2", "rule")
_CWE = re.compile(r"CWE-0*(\d+)")
_UNREACHABLE = float("inf")


def rule_key(item: dict) -> str:
    return f"find_error_{item['index']}" if item["type"] == "find_error" else item["type"]


def neighbours(gold: str, graph: CweGraph) -> list[str]:
    """Gold's parents and children in view 1000, 'CWE-n', sorted numerically."""
    g = gold[4:]
    return [f"CWE-{c}" for c in sorted(graph.parents(g) | graph.children(g), key=int)]


def _neighbour_list(gold: str, graph: CweGraph) -> str:
    parents = graph.parents(gold[4:])
    return "\n".join(f"- {c}: {graph.weaknesses[c[4:]].name} ({'more general' if c[4:] in parents else 'more specific'})"
                     for c in neighbours(gold, graph))


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def request_user(item: dict, graph: CweGraph) -> str:
    key = rule_key(item)
    gold = item["gold"].get("cwe")
    fields: dict = {}
    if key in ("exact_id", "find_error_0"):
        fields["neighbours"] = _neighbour_list(gold, graph)
    elif key == "find_error_1":
        fields = {"cwe": gold, "name": graph.weaknesses[gold[4:]].name, "release": pinned.CWE_RELEASE}
    rule = pinned.DPO_RULES[key].format(**fields)
    return pinned.DPO_PROMPT.format(target=item["target"], rule=rule, user=item["user"])


def schema_for(item: dict) -> dict:
    return pinned.DPO_SCHEMAS[pinned.DPO_FIELDS[item["type"]]]


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def _cwe(value: str, graph: CweGraph) -> tuple[str | None, str | None]:
    m = _CWE.fullmatch(value.strip().upper())
    if m is None:
        return None, "malformed"
    cwe = f"CWE-{int(m.group(1))}"
    return (cwe, None) if graph.in_view(cwe[4:]) else (None, "not_live_weakness")


def check(item: dict, value, code: set[int], graph: CweGraph) -> tuple[str | None, str | None]:
    """(canonical rejected text, None) if `value` is an admissible near miss for this item, else (None, reason)."""
    key, gold = rule_key(item), item["gold"]
    if key in ("exact_id", "find_error_0", "find_error_1"):
        cwe, reason = _cwe(value, graph)
        if reason:
            return None, reason
        if key != "find_error_1":
            if cwe == gold["cwe"]:
                return None, "gold"
            if cwe not in neighbours(gold["cwe"], graph):
                return None, "not_one_hop"
        text = cwe if key == "exact_id" else f"VULNERABLE: yes, {cwe}"
    elif key == "cvss":
        parsed = pinned.parse_cvss_v3(value)
        if parsed is None:
            return None, "malformed"
        gold_c = pinned.parse_cvss_v3(gold["vector"])[1]
        changed = sum(parsed[1][k] != gold_c[k] for k in pinned.CVSS_ORDER)
        if changed != 1:
            return None, "gold" if changed == 0 else "not_one_component"
        text = pinned.canonical_cvss(parsed[1])
    elif key == "line_loc":
        lines, g_set = set(value), set(gold["lines"])
        if lines == g_set:
            return None, "gold"
        if any(not 1 <= x <= gold["n_lines"] for x in lines):
            return None, "out_of_range"
        removed, added = g_set - lines, lines - g_set
        if len(removed) != 1 or len(added) > 1:
            return None, "not_one_change"
        if not added and len(g_set) < 2:
            return None, "single_line_drop"
        if added:
            (g,), (x,) = removed, added
            if abs(x - g) < 2:
                return None, "too_close"
            if x not in code:
                return None, "not_code_line"
        text = line_target(sorted(lines))
    else:
        raise ValueError(f"no DPO request rule for {key!r}")
    verdict = verifiers.verify_item(item["type"], text, gold, graph)
    if not verdict.strict_ok or verdict.dense >= 1:
        return None, "not_a_near_miss"
    return text, None


def parse_proposal(item: dict, row: dict) -> tuple[str, object]:
    """(status, value): status 'ok' carries the schema field's value."""
    status = stop_status(row)
    if status != "ok":
        return status, None
    field = pinned.DPO_FIELDS[item["type"]]
    try:
        value = json.loads(row["text"])[field]
    except (TypeError, ValueError, KeyError):
        return "bad_json", None
    if field == "lines":
        ok = isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    else:
        ok = isinstance(value, str)
    return ("ok", value) if ok else ("bad_json", None)


def validate(item: dict, row: dict, code: set[int], graph: CweGraph) -> tuple[str | None, object, str | None]:
    """(rejected text or None, the proposed value, reason or None)."""
    status, value = parse_proposal(item, row)
    if status != "ok":
        return None, value, status
    text, reason = check(item, value, code, graph)
    return text, value, reason


# ---------------------------------------------------------------------------
# Rule-built near misses
# ---------------------------------------------------------------------------


def _rank(item: dict, part: str) -> tuple[int, str]:
    return pinned.stable_rank(item["item_id"], pinned.TEACHER_SALTS["dpo"], part), part


def rule_value(item: dict, code: set[int], graph: CweGraph):
    """The fallback near miss, as a proposal value; deterministic in item_id."""
    key, gold = rule_key(item), item["gold"]
    if key in ("exact_id", "find_error_0"):
        return min(neighbours(gold["cwe"], graph), key=lambda c: _rank(item, c))
    if key == "find_error_1":
        return gold["cwe"]
    if key == "cvss":
        comps = pinned.parse_cvss_v3(gold["vector"])[1]
        k, v = min(((k, v) for k in pinned.CVSS_ORDER for v in pinned.CVSS_ALLOWED[k] if v != comps[k]),
                   key=lambda kv: _rank(item, f"{kv[0]}:{kv[1]}"))
        return pinned.canonical_cvss({**comps, k: v})
    if key == "line_loc":
        lines = gold["lines"]
        if len(lines) >= 2:
            g = min(lines, key=lambda x: _rank(item, str(x)))
            return [x for x in lines if x != g]
        (g,) = lines
        far = [x for x in code if abs(x - g) >= 2]
        nearest = min(abs(x - g) for x in far)
        return [min((x for x in far if abs(x - g) == nearest), key=lambda x: _rank(item, str(x)))]
    raise ValueError(f"no DPO rule for {key!r}")


def mcq_rejected(item: dict, decision: dict, graph: CweGraph) -> str:
    """The distractor closest to gold by undirected ChildOf distance; ties go to the earlier position in
    mcq_decisions.jsonl's distractors, which is the external model's step-4 order."""
    gold = item["gold"]["cwe"]
    if decision["gold"] != gold or decision["gold_letter"] != item["gold"]["letter"]:
        raise SystemExit(f"{item['item_id']}: mcq_decisions.jsonl does not match the bank")

    def far(c: str) -> float:
        d = distance(c, gold, graph, cap=len(graph.weaknesses))
        return _UNREACHABLE if d is None else d

    best = min(enumerate(decision["distractors"]), key=lambda ic: (far(ic[1]), ic[0]))[1]
    return "ANSWER: " + next(o["letter"] for o in item["gold"]["options"] if o["cwe"] == best)


# ---------------------------------------------------------------------------
# Decision per item, summary
# ---------------------------------------------------------------------------


def decide(item: dict, first: dict | None, retry: dict | None, code: set[int], graph: CweGraph,
           mcq_decision: dict | None = None) -> dict:
    """One dpo.jsonl row. `first` / `retry` are the runner's rows for attempts 1 and 2 (MCQ has neither)."""
    row = {"item_id": item["item_id"], "cve_id": item["cve_id"], "type": item["type"], "index": item["index"],
           "chosen": item["target"], "a1_reason": None, "a1_stop_category": None, "a2_reason": None,
           "a2_stop_category": None, "proposals": []}
    if item["type"] == "mcq":
        rejected = mcq_rejected(item, mcq_decision, graph)
        return {**row, "rejected": rejected, "source": "rule", "rejected_dense": _dense(item, rejected, graph)}
    for attempt, gen in ((1, first), (2, retry)):
        if gen is None:
            raise SystemExit(f"{item['item_id']}: the first DPO proposal is invalid ({row['a1_reason']}) and has no "
                             "regeneration; run `python -m generators teacher prepare-retry` and the external runner")
        text, value, reason = validate(item, gen, code, graph)
        row["proposals"].append(value)
        row[f"a{attempt}_reason"], row[f"a{attempt}_stop_category"] = reason, gen.get("stop_category")
        if text is not None:
            return {**row, "rejected": text, "source": f"model_a{attempt}", "rejected_dense": _dense(item, text, graph)}
    text, reason = check(item, rule_value(item, code, graph), code, graph)
    if text is None:
        raise AssertionError(f"{item['item_id']}: the rule-built near miss is inadmissible ({reason})")
    return {**row, "rejected": text, "source": "rule", "rejected_dense": _dense(item, text, graph)}


def _dense(item: dict, text: str, graph: CweGraph) -> str:
    return str(verifiers.verify_item(item["type"], text, item["gold"], graph).dense)


def summarise(rows: list[dict]) -> dict:
    """Per type: the plan's DPO audit line (near misses valid at the first attempt, without the rule), banded,
    the rule-fallback rate, reasons, refusal categories, and what the rejected answers score."""
    types = {}
    for t in pinned.BANK_TYPES:
        chosen = [r for r in rows if r["type"] == t]
        n = len(chosen)
        if not n:
            continue
        sources = Counter(r["source"] for r in chosen)
        rule_defined = t not in pinned.DPO_REQUEST_TYPES
        constructible = Fraction(sources["model_a1"], n)
        types[t] = {
            "n": n,
            "rule_defined": rule_defined,
            "sources": {s: sources[s] for s in SOURCES},
            "constructible_a1": None if rule_defined else str(constructible),
            "band": None if rule_defined else band(constructible),
            "rule_fallback_rate": None if rule_defined else str(Fraction(sources["rule"], n)),
            "a1_reasons": dict(sorted(Counter(r["a1_reason"] for r in chosen if r["a1_reason"]).items())),
            "a2_reasons": dict(sorted(Counter(r["a2_reason"] for r in chosen if r["a2_reason"]).items())),
            "refusal_categories": dict(sorted(Counter(
                str(r[f"{a}_stop_category"]) for r in chosen for a in ("a1", "a2")
                if r[f"{a}_reason"] == "refusal").items())),
            "rejected_dense": dict(sorted(Counter(r["rejected_dense"] for r in chosen).items(),
                                          key=lambda kv: Fraction(kv[0]))),
        }
    return {"types": types}
