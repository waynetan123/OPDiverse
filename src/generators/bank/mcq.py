"""MCQ distractors: the external model proposes, the rules admit, the prior-matched draw fills gaps.

Rules (pinned in docs/decisions/step4_decision_record.md):
- admission: a live view-1000 weakness, distinct, not gold, not an ancestor or descendant of gold;
  the first MCQ_DISTRACTORS admissible IDs are kept in the model's order, attempt 1 before attempt 2;
- regeneration: one more request when attempt 1 was refused, failed, or gave fewer than 3;
- prior-matched draw: fills whatever is still missing, weighted by non-test gold frequency;
- shortcut guard: if "pick the most familiar option" beats MCQ_SHORTCUT_MAX on non-test, every MCQ
  item in both pools is rebuilt from the draw alone.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction

from etl import pinned
from etl.cwe_graph import CweGraph

from .. import external
from .items import item_id, prompt_description

_PROPOSAL = re.compile(r"CWE-0*(\d+)")
SOURCES = ("model", "model_retry", "draw")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def request_row(fact: dict, pool: str, graph: CweGraph, attempt: int) -> dict:
    user = pinned.MCQ_REQUEST_PROMPT.format(
        description=prompt_description(fact), cwe=fact["cwe"], name=graph.weaknesses[fact["cwe"][4:]].name,
        n=pinned.MCQ_REQUEST_MAX, release=pinned.CWE_RELEASE)
    iid = item_id(fact["cve_id"], "mcq")
    return {
        "custom_id": external.custom_id(iid, attempt),
        "item_id": iid,
        "cve_id": fact["cve_id"],
        "pool": pool,
        "attempt": attempt,
        "params": external.message_params(user, pinned.MCQ_RESPONSE_SCHEMA),
    }


def pilot_sample(facts: list[dict], pool_of: dict[str, str]) -> list[dict]:
    ranked = sorted((f for f in facts if pool_of[f["cve_id"]] == "nontest"),
                    key=lambda f: (pinned.stable_rank(f["cve_id"], pinned.MCQ_SALTS["pilot"]), f["cve_id"]))
    return ranked[: pinned.MCQ_PILOT_SIZE]


# ---------------------------------------------------------------------------
# Generations -> admitted distractors
# ---------------------------------------------------------------------------


def parse_generation(row: dict | None) -> tuple[str, list[str]]:
    """(status, raw proposals). status is 'ok', 'missing', 'errored', 'refusal', 'max_tokens',
    'bad_json' or another stop reason; only 'ok' carries proposals."""
    if row is None:
        return "missing", []
    if row["result_type"] != "succeeded":
        return row["result_type"], []
    if row["stop_reason"] != "end_turn":
        return str(row["stop_reason"]), []
    try:
        proposals = json.loads(row["text"])["distractors"]
    except (TypeError, ValueError, KeyError):
        return "bad_json", []
    if not isinstance(proposals, list) or not all(isinstance(p, str) for p in proposals):
        return "bad_json", []
    return "ok", proposals


def excluded(gold: str, graph: CweGraph) -> set[str]:
    """Gold and every option that would also be a defensible answer: its ancestors and descendants."""
    g = gold[4:]
    return {f"CWE-{c}" for c in {g} | graph.ancestors(g) | graph.descendants(g)}


def admit(proposals: list[str], gold: str, graph: CweGraph, taken: list[str] = ()) -> tuple[list[str], list[list[str]]]:
    """(admitted in order, [[raw, reason], ...] for the rest). `taken` are already-admitted IDs."""
    bad = excluded(gold, graph)
    g = gold[4:]
    admitted: list[str] = []
    rejected: list[list[str]] = []
    for raw in proposals:
        m = _PROPOSAL.fullmatch(raw.strip().upper())
        if m is None:
            rejected.append([raw, "malformed"])
            continue
        cwe = f"CWE-{int(m.group(1))}"
        bare = cwe[4:]
        if cwe == gold:
            reason = "gold"
        elif not graph.in_view(bare):
            reason = "not_live_weakness"
        elif bare in graph.ancestors(g):
            reason = "ancestor"
        elif bare in graph.descendants(g):
            reason = "descendant"
        elif cwe in admitted or cwe in taken:
            reason = "duplicate"
        else:
            assert cwe not in bad
            admitted.append(cwe)
            continue
        rejected.append([raw, reason])
    return admitted, rejected


def needs_retry(status: str, admitted: list[str]) -> bool:
    return status != "ok" or len(admitted) < pinned.MCQ_DISTRACTORS


# ---------------------------------------------------------------------------
# Prior-matched draw
# ---------------------------------------------------------------------------


def gold_counts(facts: list[dict], pool_of: dict[str, str]) -> Counter:
    """How often each CWE is gold in the non-test pool. Test labels never enter."""
    return Counter(f["cwe"] for f in facts if pool_of[f["cve_id"]] == "nontest")


def prior_draw(cve_id: str, gold: str, counts: Counter, graph: CweGraph, k: int, taken: list[str] = ()) -> list[str]:
    """k CWEs drawn without replacement, each with probability proportional to its non-test gold
    count, from the non-test gold labels that are admissible for this gold. Exact integer
    arithmetic: slot s picks by stable_rank(cve_id, salt, s) modulo the remaining total weight."""
    bad = excluded(gold, graph) | set(taken)
    pool = sorted((c for c in counts if c not in bad), key=lambda c: int(c[4:]))
    out: list[str] = []
    for slot in range(len(taken), len(taken) + k):
        if not pool:
            raise SystemExit(f"{cve_id}: no admissible CWE left for the prior-matched draw")
        r = pinned.stable_rank(cve_id, pinned.MCQ_SALTS["draw"], str(slot)) % sum(counts[c] for c in pool)
        for c in pool:
            r -= counts[c]
            if r < 0:
                out.append(c)
                pool.remove(c)
                break
    return out


# ---------------------------------------------------------------------------
# Decision per CVE, layout, shortcut
# ---------------------------------------------------------------------------


def decide(fact: dict, pool: str, generations: list[dict | None], counts: Counter, graph: CweGraph) -> dict:
    """generations: the attempt-1 row and, if one was made, the attempt-2 row (None if absent)."""
    chosen: list[str] = []
    sources: list[str] = []
    attempts = []
    for n, row in enumerate(generations, 1):
        status, proposals = parse_generation(row)
        admitted, rejected = admit(proposals, fact["cwe"], graph, chosen)
        take = admitted[: pinned.MCQ_DISTRACTORS - len(chosen)]
        chosen += take
        sources += ["model" if n == 1 else "model_retry"] * len(take)
        attempts.append({"attempt": n, "status": status, "proposed": proposals, "admitted": admitted, "rejected": rejected})
        if len(chosen) == pinned.MCQ_DISTRACTORS:
            break
    missing = pinned.MCQ_DISTRACTORS - len(chosen)
    if missing:
        chosen += prior_draw(fact["cve_id"], fact["cwe"], counts, graph, missing, chosen)
        sources += ["draw"] * missing
    return {"cve_id": fact["cve_id"], "pool": pool, "gold": fact["cwe"], "attempts": attempts,
            "distractors": chosen, "sources": sources}


def draw_only(fact: dict, pool: str, counts: Counter, graph: CweGraph) -> dict:
    """The prior-matched draw alone: --dry-mcq, and every item when the shortcut guard fires."""
    return {"cve_id": fact["cve_id"], "pool": pool, "gold": fact["cwe"], "attempts": [],
            "distractors": prior_draw(fact["cve_id"], fact["cwe"], counts, graph, pinned.MCQ_DISTRACTORS),
            "sources": ["draw"] * pinned.MCQ_DISTRACTORS}


def layout(cve_id: str, gold: str, distractors: list[str], graph: CweGraph) -> tuple[list[dict], str]:
    """(options A-D, gold letter). The gold letter is a hash of the CVE ID; distractors fill the
    other letters in stable_rank order."""
    letters = pinned.MCQ_LETTERS
    gold_letter = letters[pinned.stable_rank(cve_id, pinned.MCQ_SALTS["letter"]) % len(letters)]
    rest = iter(sorted(distractors, key=lambda c: (pinned.stable_rank(cve_id, pinned.MCQ_SALTS["slot"], c), c)))
    options = []
    for letter in letters:
        cwe = gold if letter == gold_letter else next(rest)
        options.append({"letter": letter, "cwe": cwe, "name": graph.weaknesses[cwe[4:]].name})
    return options, gold_letter


def shortcut(decisions: list[dict], counts: Counter) -> Fraction:
    """Mean score, over non-test items, of picking the option most often gold in non-test
    (ties split evenly)."""
    rows = [d for d in decisions if d["pool"] == "nontest"]
    total = Fraction(0)
    for d in rows:
        options = [d["gold"], *d["distractors"]]
        best = max(counts[o] for o in options)
        top = [o for o in options if counts[o] == best]
        total += Fraction(d["gold"] in top, len(top))
    return total / len(rows) if rows else Fraction(0)
