"""Verifiers: one model reply + gold -> scores. Part of the pinned definitions.

Every consumer that scores a reply calls these functions: the contamination probe (step 3),
the signal audit (5), DPO negative validation (6), the GRPO reward (11) and evaluation (12).

Each verifier parses, then scores.
- The lenient parser is what scoring uses; it is deliberately loose (the plan's format ramp).
- The strict parser requires exactly the target format; it is a robustness column.
- `metric` is the reported evaluation metric; `dense` is the training signal. Both are in
  [0, 1] and both are 0 when the lenient parser fails.

Parsers are tuned on dev outputs only, never test; bump PARSER_VERSION on any change.
MCQ, find-the-error and line localisation are added with the question bank (step 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from . import pinned

PARSER_VERSION = "v1"


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

_CVSS_METRIC = re.compile(r"(?<![A-Za-z])(AV|AC|PR|UI|S|C|I|A)\s*:\s*([A-Za-z])(?![A-Za-z])", re.IGNORECASE)
_CVSS_STRICT = re.compile(
    r"(?:CVSS:3\.[01]/)?" + "/".join(f"{k}:[{pinned.CVSS_ALLOWED[k]}]" for k in pinned.CVSS_ORDER)
)


def parse_cvss(reply: str) -> dict[str, str] | None:
    """The last valid value of each base metric found anywhere in the reply. Invalid values
    (e.g. 'AV:X') are ignored; metrics never given are absent. None if no metric is found."""
    found: dict[str, str] = {}
    for key, value in _CVSS_METRIC.findall(reply):
        key, value = key.upper(), value.upper()
        if value in pinned.CVSS_ALLOWED[key]:
            found[key] = value
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


VERIFIERS = {"exact_id": verify_exact_id, "cvss": verify_cvss}
