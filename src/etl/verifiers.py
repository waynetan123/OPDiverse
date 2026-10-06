"""Verifiers: one model reply + gold -> scores. Part of the pinned definitions.

Every consumer that scores a reply calls these functions: the contamination probe (step 3),
the signal audit (5), DPO negative validation (6), the GRPO reward (11) and evaluation (12).

Each verifier parses, then scores.
- The lenient parser is what scoring uses; it is deliberately loose (the plan's format ramp).
- The strict parser requires exactly the target format; it is a robustness column.
- `metric` is the reported evaluation metric; `dense` is the training signal. Both are in
  [0, 1] and both are 0 when the lenient parser fails.

Parsers are tuned on dev outputs only, never test; bump PARSER_VERSION on any change.
MCQ, find-the-error and line localisation were added with the question bank (step 4), before
v1 had scored any bank output, so the version stays v1.

v2 (step 7, owner-approved after the parser review of the untrained backbone's non-test replies):
- MCQ: a reply whose last line is exactly one option, with or without its letter, reads as that letter;
- CVSS: the v3.1 specification's value names (AV:Network, S:Unchanged, ...) count as their letters;
- line localisation: without a LINES: field, numbers in code blocks and in echoed "N: code" listing
  lines are not read as predictions.
Details and counts: docs/decisions/step7_decision_record.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from . import pinned

PARSER_VERSION = "v2"


@dataclass(frozen=True)
class Verdict:
    parsed: str | None   # canonical answer; CVSS metrics the reply lacked are "?"
    parse_ok: bool       # lenient parser found an answer
    strict_ok: bool      # reply is exactly the target format
    metric: Fraction     # reported metric (0 on parse failure)
    dense: Fraction      # training score (0 on parse failure)


_FAILED = Verdict(None, False, False, Fraction(0), Fraction(0))

# ---------------------------------------------------------------------------
# Exact-ID: target "CWE-787"
# ---------------------------------------------------------------------------

_CWE_LENIENT = re.compile(r"(?<![A-Za-z0-9])CWE[-_ :]*0*(\d{1,5})(?!\d)", re.IGNORECASE)
_CWE_STRICT = re.compile(r"CWE-[1-9]\d*")


def parse_cwe(reply: str) -> str | None:
    """The last CWE ID mentioned, normalised: 'cwe 0787' -> 'CWE-787'."""
    found = _CWE_LENIENT.findall(reply)
    return f"CWE-{int(found[-1])}" if found else None


def strict_cwe(reply: str) -> bool:
    return _CWE_STRICT.fullmatch(reply.strip()) is not None


def verify_exact_id(reply: str, gold: str, graph: pinned.CweLookup) -> Verdict:
    parsed = parse_cwe(reply)
    if parsed is None:
        return _FAILED
    return Verdict(
        parsed=parsed,
        parse_ok=True,
        strict_ok=strict_cwe(reply),
        metric=Fraction(parsed == gold),
        dense=pinned.hierarchy_score(parsed, gold, graph),
    )


# ---------------------------------------------------------------------------
# CVSS: target "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
# ---------------------------------------------------------------------------

_CVSS_METRIC = re.compile(r"(?<![A-Za-z])(AV|AC|PR|UI|S|C|I|A)\s*:\s*(adjacent network|[A-Za-z]+)(?![A-Za-z])", re.IGNORECASE)
# v2: the CVSS v3.1 specification's own value names, read as their letters. Other words never count.
_CVSS_IMPACT = {"NONE": "N", "LOW": "L", "HIGH": "H"}
CVSS_WORDS = {
    "AV": {"NETWORK": "N", "ADJACENT NETWORK": "A", "ADJACENT": "A", "LOCAL": "L", "PHYSICAL": "P"},
    "AC": {"LOW": "L", "HIGH": "H"},
    "PR": dict(_CVSS_IMPACT),
    "UI": {"NONE": "N", "REQUIRED": "R"},
    "S": {"UNCHANGED": "U", "CHANGED": "C"},
    "C": dict(_CVSS_IMPACT), "I": dict(_CVSS_IMPACT), "A": dict(_CVSS_IMPACT),
}
_CVSS_STRICT = re.compile(
    r"(?:CVSS:3\.[01]/)?" + "/".join(f"{k}:[{pinned.CVSS_ALLOWED[k]}]" for k in pinned.CVSS_ORDER)
)


def parse_cvss(reply: str) -> dict[str, str] | None:
    """The last valid value of each base metric found anywhere in the reply, as a letter or (v2) as the
    specification's value name ('AV:Network'). Invalid values ('AV:X', 'S:Single') are ignored; metrics
    never given are absent. None if no metric is found."""
    found: dict[str, str] = {}
    for key, value in _CVSS_METRIC.findall(reply):
        key, value = key.upper(), " ".join(value.upper().split())
        if len(value) == 1:
            if value in pinned.CVSS_ALLOWED[key]:
                found[key] = value
        elif value in CVSS_WORDS[key]:
            found[key] = CVSS_WORDS[key][value]
    return {k: found[k] for k in pinned.CVSS_ORDER if k in found} or None


def strict_cvss(reply: str) -> bool:
    """The reply is exactly the target: all 8 metrics, canonical order, uppercase, optional CVSS:3.x/ prefix."""
    return _CVSS_STRICT.fullmatch(reply.strip()) is not None


def verify_cvss(reply: str, gold: str) -> Verdict:
    parsed = parse_cvss(reply)
    if parsed is None:
        return _FAILED
    gold_parsed = pinned.parse_cvss_v3(gold)
    if gold_parsed is None:
        raise ValueError(f"gold {gold!r} is not a CVSS v3.x base vector")
    gold_c = gold_parsed[1]
    score = Fraction(sum(parsed.get(k) == gold_c[k] for k in pinned.CVSS_ORDER), len(pinned.CVSS_ORDER))
    return Verdict(
        parsed="/".join(f"{k}:{parsed.get(k, '?')}" for k in pinned.CVSS_ORDER),
        parse_ok=True,
        strict_ok=strict_cvss(reply),
        metric=score,
        dense=score,
    )


# ---------------------------------------------------------------------------
# MCQ: target "ANSWER: B"
# ---------------------------------------------------------------------------

_MCQ_ANSWER = re.compile(r"(?i:ANSWER)\s*:\s*[(\[*]*\s*([A-D])(?![A-Za-z0-9])")
_MCQ_LONE = re.compile(r"[\s(\[*]*([A-D])[\s).\]*]*")
_MCQ_FALLBACK = re.compile(r"\(([A-D])\)|\b(?:option|answer is)\s+([A-D])(?![A-Za-z0-9])", re.IGNORECASE)
_MCQ_STRICT = re.compile(r"ANSWER: [A-D]")


def parse_mcq(reply: str, options: list[dict] | None = None) -> str | None:
    """The capital letter after the last ANSWER:, else a reply that is one letter, else the last
    (X) / 'option X' / 'answer is X'. Letters must be capitals, so 'answer: a buffer overflow'
    is not read as A. Else (v2, given the item's options) a last line that is exactly one option,
    'B. CWE-787: Out-of-bounds Write' or 'CWE-787: Out-of-bounds Write', reads as that option's letter."""
    if found := _MCQ_ANSWER.findall(reply):
        return found[-1].upper()
    if m := _MCQ_LONE.fullmatch(reply):
        return m.group(1)
    found = [a or b for a, b in _MCQ_FALLBACK.findall(reply)]
    found = [f for f in found if f in pinned.MCQ_LETTERS]
    if found:
        return found[-1]
    lines = [line.strip() for line in reply.split("\n") if line.strip()]
    if options and lines:
        for o in options:
            text = f"{o['cwe']}: {o['name']}"
            if lines[-1] in (pinned.MCQ_OPTION.format(letter=o["letter"], cwe=o["cwe"], name=o["name"]), text):
                return o["letter"]
    return None


def strict_mcq(reply: str) -> bool:
    return _MCQ_STRICT.fullmatch(reply.strip()) is not None


def verify_mcq(reply: str, gold: str, options: list[dict] | None = None) -> Verdict:
    parsed = parse_mcq(reply, options)
    if parsed is None:
        return _FAILED
    score = Fraction(parsed == gold)
    return Verdict(parsed=parsed, parse_ok=True, strict_ok=strict_mcq(reply), metric=score, dense=score)


# ---------------------------------------------------------------------------
# Find-the-error: target "VULNERABLE: yes, CWE-787" or "VULNERABLE: no"
# ---------------------------------------------------------------------------

_FE_FIELD = re.compile(r"VULNERABLE\s*:\s*\**\s*(yes|no)(?![A-Za-z])", re.IGNORECASE)
_FE_NEGATIVE = re.compile(
    r"\b(?:not|isn't|is\s+not|no\s+longer)\s+vulnerable\b|\bno\s+(?:known\s+|security\s+)?vulnerabilit(?:y|ies)\b",
    re.IGNORECASE,
)
_FE_POSITIVE = re.compile(r"\bvulnerab(?:le|ilit(?:y|ies))\b", re.IGNORECASE)
_FE_STRICT = re.compile(r"VULNERABLE: (?:yes, CWE-[1-9]\d*|no)")


def parse_find_error(reply: str) -> tuple[bool | None, str | None]:
    """(vulnerable?, CWE or None). The label is the yes/no after the last VULNERABLE:; without
    that field, the last vulnerability statement in the reply decides, negations checked first."""
    if found := _FE_FIELD.findall(reply):
        label = found[-1].lower() == "yes"
    else:
        negatives = [m.span() for m in _FE_NEGATIVE.finditer(reply)]
        events = [(s, False) for s, _ in negatives]
        events += [(m.start(), True) for m in _FE_POSITIVE.finditer(reply)
                   if not any(s <= m.start() < e for s, e in negatives)]
        if not events:
            return None, None
        label = max(events)[1]
    return label, parse_cwe(reply) if label else None


def strict_find_error(reply: str) -> bool:
    return _FE_STRICT.fullmatch(reply.strip()) is not None


def verify_find_error(reply: str, vulnerable: bool, gold_cwe: str, graph: pinned.CweLookup) -> Verdict:
    """Per-function score. Vulnerable prompt: 0.5*label + 0.5*label*hierarchy credit. Patched
    prompt: label. `metric` is per-function label accuracy; paired accuracy is paired_accuracy()."""
    label, cwe = parse_find_error(reply)
    if label is None:
        return _FAILED
    correct = Fraction(label == vulnerable)
    if vulnerable:
        credit = pinned.hierarchy_score(cwe, gold_cwe, graph) if cwe else Fraction(0)
        dense = correct / 2 + correct * credit / 2
    else:
        dense = correct
    parsed = "VULNERABLE: no" if not label else f"VULNERABLE: yes, {cwe}" if cwe else "VULNERABLE: yes"
    return Verdict(parsed=parsed, parse_ok=True, strict_ok=strict_find_error(reply), metric=correct, dense=dense)


def paired_accuracy(vulnerable: Verdict, patched: Verdict) -> Fraction:
    """The headline find-the-error score for one CVE: both functions labelled correctly."""
    return Fraction(vulnerable.metric == 1 and patched.metric == 1)


def paired_cwe_accuracy(vulnerable: Verdict, patched: Verdict, gold_cwe: str) -> Fraction:
    """The stricter column: both labels right and the vulnerable function's CWE exactly right."""
    return Fraction(paired_accuracy(vulnerable, patched) == 1 and vulnerable.parsed == f"VULNERABLE: yes, {gold_cwe}")


# ---------------------------------------------------------------------------
# Line localisation: target "LINES: 12, 13, 17" or "LINES: none"
# ---------------------------------------------------------------------------

_LINES_FIELD = re.compile(r"LINES\s*:", re.IGNORECASE)
_LINES_NONE = re.compile(r"\bnone\b", re.IGNORECASE)
_LINES_STRICT = re.compile(r"LINES: (?:none|[1-9]\d*(?:, [1-9]\d*)*)")
_CODE_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)  # an unclosed fence runs to the end of a cut-off reply
_LISTING_LINE = re.compile(r"\s*\d+:")                     # an echoed line of the numbered function


def without_code(reply: str) -> str:
    """The reply minus code blocks and echoed "N: code" listing lines (v2's LINES:-less fallback)."""
    return "\n".join(line for line in _CODE_FENCE.sub(" ", reply).split("\n") if not _LISTING_LINE.match(line))


def parse_lines(reply: str, n_lines: int) -> list[int] | None:
    """The plan's parser: text after the last LINES: (else the whole reply, without code blocks and
    echoed listing lines: v2) -> every integer -> drop those outside [1, n_lines] -> dedupe and sort.
    Integers found (even if all dropped) or a literal 'none' parse; neither is a failure."""
    fields = list(_LINES_FIELD.finditer(reply))
    tail = reply[fields[-1].end():] if fields else without_code(reply)
    numbers = [int(x) for x in re.findall(r"\d+", tail)]
    if not numbers:
        return [] if _LINES_NONE.search(tail) else None
    return sorted({x for x in numbers if 1 <= x <= n_lines})


def strict_lines(reply: str, n_lines: int) -> bool:
    text = reply.strip()
    if _LINES_STRICT.fullmatch(text) is None:
        return False
    if text == "LINES: none":
        return True
    numbers = [int(x) for x in text[len("LINES: "):].split(", ")]
    return numbers == sorted(set(numbers)) and numbers[-1] <= n_lines


def matched_lines(predicted: list[int], gold: list[int]) -> int:
    """Size of a maximum one-to-one matching in which a prediction may claim a gold line at most
    1 away. Each prediction p is the interval [p-1, p+1]; taking predictions in ascending order and
    giving each the lowest unclaimed gold line in its interval is optimal. Among maximum matchings
    the pin prefers the most exact hits, then the lowest lines; that choice changes which pairs are
    matched, never how many, so F1 does not depend on it."""
    gold = sorted(set(gold))
    j = matched = 0
    for p in sorted(set(predicted)):
        while j < len(gold) and gold[j] < p - 1:
            j += 1
        if j < len(gold) and gold[j] <= p + 1:
            matched += 1
            j += 1
    return matched


def line_f1(predicted: list[int], gold: list[int]) -> Fraction:
    if not gold:
        raise ValueError("gold line set is empty; step 1 drops empty patches")
    if not predicted:
        return Fraction(0)
    tp = matched_lines(predicted, gold)
    return Fraction(2 * tp, len(set(predicted)) + len(set(gold)))


def verify_line_loc(reply: str, gold: list[int], n_lines: int) -> Verdict:
    parsed = parse_lines(reply, n_lines)
    if parsed is None:
        return _FAILED
    score = line_f1(parsed, gold)
    return Verdict(
        parsed="LINES: " + (", ".join(map(str, parsed)) if parsed else "none"),
        parse_ok=True,
        strict_ok=strict_lines(reply, n_lines),
        metric=score,
        dense=score,
    )


# ---------------------------------------------------------------------------
# Registry and bank dispatch
# ---------------------------------------------------------------------------

VERIFIERS = {
    "mcq": verify_mcq,
    "exact_id": verify_exact_id,
    "cvss": verify_cvss,
    "find_error": verify_find_error,
    "line_loc": verify_line_loc,
}


def verify_item(item_type: str, reply: str, gold: dict, graph: pinned.CweLookup) -> Verdict:
    """Score a reply to a question-bank item from the item's `gold` field."""
    if item_type == "mcq":
        return verify_mcq(reply, gold["letter"], gold.get("options"))
    if item_type == "exact_id":
        return verify_exact_id(reply, gold["cwe"], graph)
    if item_type == "cvss":
        return verify_cvss(reply, gold["vector"])
    if item_type == "find_error":
        return verify_find_error(reply, gold["vulnerable"], gold["cwe"], graph)
    if item_type == "line_loc":
        return verify_line_loc(reply, gold["lines"], gold["n_lines"])
    raise ValueError(f"unknown item type {item_type!r}")
