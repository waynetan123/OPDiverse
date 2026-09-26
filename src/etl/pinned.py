"""Pinned definitions for the OPDiverse fact table, test window and hierarchy scoring.

Everything here can move a census count, a split or a score. It is frozen before step 0: any
change needs a matching entry in docs/decisions/step{1,2}_decision_record.md. Pure functions,
stdlib only, no I/O.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, Protocol

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

CWE_RELEASE = "4.20"
CWE_VIEW = "1000"
NVD_SOURCE = "nvd@nist.gov"
CWE_PLACEHOLDERS = frozenset({"NVD-CWE-Other", "NVD-CWE-noinfo"})

TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"
# HF commit of Qwen/Qwen2.5-7B-Instruct (lastModified 2025-01-12) and the sha256 of its tokenizer.json.
TOKENIZER_REVISION: str | None = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_SHA256: str | None = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
# Public release date of Qwen2.5. Its pretraining cutoff is not published; reported beside the split boundary.
BACKBONE_RELEASED = "2024-09-19"

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

TOKEN_CAP = 10_000
TOKEN_CAP_REPORT = (1_024, 2_048, 4_096)
PATCH_FRACTION = Fraction(1, 5)

# ---------------------------------------------------------------------------
# Test window (step 2)
# ---------------------------------------------------------------------------

TEST_FRACTION = Fraction(3, 20)
NEAR_DUP_JACCARD = Fraction(4, 5)
SHINGLE_N = 3

# ---------------------------------------------------------------------------
# Exact-ID hierarchy credit (step 2)
# ---------------------------------------------------------------------------

# If the best constant answer's mean symmetric score over non-test facts exceeds this, the
# direction-aware schedule is adopted. The computed number decides; see baselines.json.
EXACT_ID_THRESHOLD = Fraction(1, 5)
SCHEDULES = ("symmetric", "direction_aware")
# Decided at step 2: best constant CWE-119 scores 0.262 under the symmetric schedule (> 0.2).
EXACT_ID_SCHEDULE = "direction_aware"
SYMMETRIC_CREDIT = {0: Fraction(1), 1: Fraction(1, 2), 2: Fraction(1, 4)}
# Relation of prediction to gold -> credit. At distance 2, a prediction that is an ancestor of
# gold (by any path) takes the ancestor discount; every other 2-hop relation scores 1/4.
DIRECTION_AWARE_CREDIT = {
    "exact": Fraction(1),
    "child": Fraction(1, 2),
    "parent": Fraction(1, 4),
    "descendant_2": Fraction(1, 4),   # grandchild
    "sibling": Fraction(1, 4),        # shares a parent
    "coparent": Fraction(1, 4),       # shares a child
    "ancestor_2": Fraction(1, 8),     # grandparent, or an ancestor also reachable in 2 hops
    "far": Fraction(0),
}

# Deprecated CWE-1000 weaknesses -> replacement, read from each entry's Description in the
# 4.20 XML. A replacement is recorded only where MITRE names exactly one successor; None
# means the row is dropped. Keys must cover every deprecated weakness in the release.
DEPRECATED_REPLACEMENT: dict[str, str | None] = {
    "71": "62",      # "Please refer to CWE-62"
    "92": "75",      # "CWE-75 is a more appropriate mapping"
    "132": "170",    # duplicate of CWE-170
    "216": None,     # no successor named
    "217": None,     # split into CWE-766 and CWE-767
    "218": "493",    # duplicate of CWE-493
    "225": "199",    # "can be found at CWE-199" (a category, so the row still drops)
    "247": "350",    # duplicate of CWE-350
    "249": "785",    # "most of its content has been transferred to CWE-785"
    "292": "350",    # duplicate of CWE-350
    "365": None,     # no successor named
    "373": None,     # overlaps CWE-362 and CWE-662
    "423": "441",    # duplicate of CWE-441
    "443": "113",    # "can be found at CWE-113"
    "458": None,     # description duplicated CWE-454, name suggested CWE-665
    "516": "385",    # "can be found at CWE-385"
    "533": "532",    # "See CWE-532"
    "534": "532",    # "See CWE-532"
    "542": "532",    # "See CWE-532"
    "545": None,     # "partially overlaps CWE-470" - not a replacement
    "592": "287",    # redundant with CWE-287
    "596": "1023",   # "Its closest equivalent is CWE-1023"
    "769": "774",    # duplicate of CWE-774
    "1187": "908",   # duplicate of CWE-908
    "1324": "319",   # "integrated into CWE-319"
}

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_rank(*parts: str) -> int:
    """Deterministic 64-bit rank. Never use hash(): it is salted per process."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Function text
# ---------------------------------------------------------------------------


def split_lines(text: str) -> list[str]:
    """Split on LF only. str.splitlines() also breaks on \\f, \\v, \\x1c-\\x1e, \\x85, \\u2028."""
    return text.split("\n")


def clean_function(raw: str) -> str:
    """The stored, model-facing function: CRLF/CR -> LF, leading/trailing blank lines removed.

    Nothing else changes; line numbers everywhere refer to this text, 1-indexed.
    """
    lines = split_lines(raw.replace("\r\n", "\n").replace("\r", "\n"))
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def render_numbered(text: str) -> str:
    """The line-numbered form the model sees for line localisation."""
    return "\n".join(f"{i}: {line}" for i, line in enumerate(split_lines(text), 1))


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _raw_string_start(text: str, i: int) -> bool:
    """True if the quote at text[i] opens a C++ raw string (R"..., u8R"..., LR"...)."""
    if i == 0 or text[i - 1] != "R":
        return False
    k = i - 1
    if text[max(0, k - 2):k] == "u8":
        k -= 2
    elif k >= 1 and text[k - 1] in "uUL":
        k -= 1
    return k == 0 or not _is_ident_char(text[k - 1])


def _digit_separator(text: str, i: int) -> bool:
    """True if the apostrophe at text[i] is a C++14 digit separator (1'000'000)."""
    if i == 0 or i + 1 >= len(text) or not _is_ident_char(text[i - 1]) or not text[i + 1].isalnum():
        return False
    k = i - 1
    while k > 0 and (_is_ident_char(text[k - 1]) or text[k - 1] == "'"):
        k -= 1
    return text[k].isdigit()


def mask_comments(text: str) -> tuple[str, bool]:
    """Blank out // and /* */ comments, keeping every newline so line numbers survive.

    String and character literals are skipped so comment markers inside them are left alone.
    Returns (text, False) if a literal or block comment is unterminated; the caller then falls
    back to the unmasked text.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = i
            while j < n and text[j] != "\n":
                if text[j] == "\\" and j + 1 < n and text[j + 1] == "\n":
                    out[j] = " "
                    j += 2  # backslash-newline continues the comment; keep the newline
                    continue
                out[j] = " "
                j += 1
            i = j
        elif c == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                return text, False
            for j in range(i, end + 2):
                if text[j] != "\n":
                    out[j] = " "
            i = end + 2
        elif c == '"' and _raw_string_start(text, i):
            paren = text.find("(", i + 1)
            if paren < 0 or paren - i - 1 > 16:
                return text, False
            terminator = ")" + text[i + 1:paren] + '"'
            end = text.find(terminator, paren + 1)
            if end < 0:
                return text, False
            i = end + len(terminator)
        elif c == "'" and _digit_separator(text, i):
            i += 1
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    return text, False
                j += 1
            if j >= n:
                return text, False
            i = j + 1
        else:
            i += 1
    return "".join(out), True


def line_key(line: str) -> str:
    """Comparison key for one line: all whitespace removed."""
    return "".join(line.split())


def _code_lines(text: str) -> list[tuple[int, str]]:
    """(1-indexed line number, key) for every line whose key is non-empty."""
    return [(i, k) for i, line in enumerate(split_lines(text), 1) if (k := line_key(line))]


def _code_keys(text: str) -> list[str]:
    masked, ok = mask_comments(text)
    return [k for _, k in _code_lines(masked if ok else text)]


def norm_body_hash(text: str) -> str:
    """Whitespace- and comment-insensitive hash of a function, for near-duplicate checks."""
    return sha256_text("\n".join(_code_keys(text)))


def code_shingles(text: str) -> frozenset[tuple[str, ...]]:
    """SHINGLE_N consecutive code-line keys (comment-masked, whitespace removed). A function
    shorter than SHINGLE_N code lines is one shingle."""
    keys = _code_keys(text)
    return frozenset(tuple(keys[i:i + SHINGLE_N]) for i in range(max(1, len(keys) - SHINGLE_N + 1)))


def jaccard_at_least(a: frozenset, b: frozenset, threshold: Fraction = NEAR_DUP_JACCARD) -> bool:
    """|a & b| / |a | b| >= threshold, in exact integer arithmetic."""
    inter = len(a & b)
    return inter * threshold.denominator >= (len(a) + len(b) - inter) * threshold.numerator


# ---------------------------------------------------------------------------
# Patch line set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchResult:
    lines: tuple[int, ...]        # gold patch line set on the vulnerable function, sorted
    n_lines: int                  # all lines of the vulnerable function
    n_code_lines: int             # lines with a non-empty key (the 20% denominator)
    n_inserted: int               # patched-side lines not matched ("+" lines)
    n_deleted: int                # vulnerable-side lines not matched ("-" lines)
    mask_ok: bool                 # comment masking applied to both sides
    opcodes: tuple[tuple[str, int, int, int, int], ...]  # difflib opcodes over code lines
    vuln_code: tuple[int, ...]    # original line number of each vulnerable code line
    patched_code: tuple[int, ...]  # original line number of each patched code line


def patch_line_set(vuln: str, patched: str) -> PatchResult:
    """Gold lines for line localisation. Both inputs are clean_function() output.

    Lines are compared on their key after comment masking; blank and comment-only lines take
    no part in the diff. Replaced and deleted lines are gold; an insertion is attributed to
    the preceding code line, or to the first code line if it comes before all of them.
    """
    masked_v, ok_v = mask_comments(vuln)
    masked_p, ok_p = mask_comments(patched)
    ok = ok_v and ok_p
    a = _code_lines(masked_v if ok else vuln)
    b = _code_lines(masked_p if ok else patched)
    matcher = difflib.SequenceMatcher(None, [k for _, k in a], [k for _, k in b], autojunk=False)
    opcodes = tuple(matcher.get_opcodes())
    gold: set[int] = set()
    n_inserted = n_deleted = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        n_deleted += i2 - i1
        n_inserted += j2 - j1
        if tag in ("replace", "delete"):
            gold.update(a[i][0] for i in range(i1, i2))
        elif a:  # insert
            gold.add(a[i1 - 1][0] if i1 > 0 else a[0][0])
    return PatchResult(
        lines=tuple(sorted(gold)),
        n_lines=len(split_lines(vuln)),
        n_code_lines=len(a),
        n_inserted=n_inserted,
        n_deleted=n_deleted,
        mask_ok=ok,
        opcodes=opcodes,
        vuln_code=tuple(ln for ln, _ in a),
        patched_code=tuple(ln for ln, _ in b),
    )


def exceeds_fraction(count: int, denominator: int) -> bool:
    """count > PATCH_FRACTION * denominator, in exact integer arithmetic."""
    return count * PATCH_FRACTION.denominator > denominator * PATCH_FRACTION.numerator


# ---------------------------------------------------------------------------
# CVSS v3.x
# ---------------------------------------------------------------------------

CVSS_VERSIONS = ("3.0", "3.1")
CVSS_ORDER = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")
CVSS_ALLOWED = {"AV": "NALP", "AC": "LH", "PR": "NLH", "UI": "NR", "S": "UC", "C": "HLN", "I": "HLN", "A": "HLN"}
_CVSS_DECOMPOSED = {
    "AV": "attackVector", "AC": "attackComplexity", "PR": "privilegesRequired", "UI": "userInteraction",
    "S": "scope", "C": "confidentialityImpact", "I": "integrityImpact", "A": "availabilityImpact",
}
_CVSS_METRIC_KEYS = (("3.1", "cvssMetricV31"), ("3.0", "cvssMetricV30"))


def parse_cvss_v3(vector: str) -> tuple[str | None, dict[str, str]] | None:
    """Parse a v3.x base vector in any component order, case-insensitively.

    Returns (declared version or None if unprefixed, components in canonical order), or None
    if any base metric is missing, duplicated, unknown or out of range.
    """
    parts = vector.strip().split("/")
    version = None
    if parts[0].upper().startswith("CVSS:"):
        version = parts[0][5:]
        if version not in CVSS_VERSIONS:
            return None
        parts = parts[1:]
    components: dict[str, str] = {}
    for part in parts:
        key, sep, value = part.partition(":")
        key, value = key.strip().upper(), value.strip().upper()
        if not sep or key not in CVSS_ALLOWED or key in components or len(value) != 1 or value not in CVSS_ALLOWED[key]:
            return None
        components[key] = value
    if len(components) != len(CVSS_ORDER):
        return None
    return version, {k: components[k] for k in CVSS_ORDER}


def canonical_cvss(components: dict[str, str]) -> str:
    return "/".join(f"{k}:{components[k]}" for k in CVSS_ORDER)


def select_nvd_cvss(metrics: dict) -> tuple[dict | None, str]:
    """NVD's own v3.x vector, preferring v3.1. Filters on source, never on type.

    Returns ({version, vector, components, both_versions}, "ok") or (None, drop reason).
    """
    by_version = {}
    for version, key in _CVSS_METRIC_KEYS:
        entries = [m for m in metrics.get(key, []) if m.get("source") == NVD_SOURCE]
        if entries:
            by_version[version] = entries
    if not by_version:
        if metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30"):
            return None, "cna_only_v3"
        if metrics.get("cvssMetricV2"):
            return None, "v2_only"
        if metrics.get("cvssMetricV40"):
            return None, "v4_only"
        return None, "no_cvss"
    version = "3.1" if "3.1" in by_version else "3.0"
    entries = by_version[version]
    if len({m["cvssData"]["vectorString"] for m in entries}) > 1:
        return None, "conflicting_nvd_vectors"
    data = entries[0]["cvssData"]
    parsed = parse_cvss_v3(data["vectorString"])
    if parsed is None:
        return None, "invalid_vector"
    declared, components = parsed
    if declared != version or data.get("version", version) != version:
        return None, "version_mismatch"
    for key, field in _CVSS_DECOMPOSED.items():
        full = data.get(field)
        if full is not None and full[:1].upper() != components[key]:
            return None, "decomposed_mismatch"
    return {
        "version": version,
        "vector": canonical_cvss(components),
        "components": components,
        "both_versions": len(by_version) == 2,
    }, "ok"


# ---------------------------------------------------------------------------
# CWE
# ---------------------------------------------------------------------------


class CweLookup(Protocol):
    def is_deprecated_weakness(self, cwe: str) -> bool: ...
    def is_category(self, cwe: str) -> bool: ...
    def is_view(self, cwe: str) -> bool: ...
    def in_view(self, cwe: str) -> bool: ...
    def parents(self, cwe: str) -> frozenset[str]: ...
    def children(self, cwe: str) -> frozenset[str]: ...
    def ancestors(self, cwe: str) -> frozenset[str]: ...
    def descendants(self, cwe: str) -> frozenset[str]: ...


_CWE_ID = re.compile(r"CWE-(\d+)")


def nvd_cwe_values(weaknesses: list[dict]) -> tuple[bool, list[str]]:
    """(any NVD-sourced weakness block present, sorted distinct values across those blocks)."""
    blocks = [w for w in weaknesses if w.get("source") == NVD_SOURCE]
    values = {
        d["value"].strip()
        for w in blocks
        for d in w.get("description", [])
        if d.get("lang") == "en" and d.get("value", "").strip()
    }
    return bool(blocks), sorted(values)


def resolve_cwe(has_nvd_block: bool, values: list[str], graph: CweLookup) -> tuple[str | None, str, tuple[str, ...]]:
    """One CWE-1000 weakness per CVE, or a drop reason.

    Order: strip placeholders -> map deprecated -> dedupe -> multi-CWE -> must be a
    non-deprecated view-1000 weakness. Returns (CWE-id or None, reason, mapped-from IDs).
    """
    if not has_nvd_block:
        return None, "no_nvd_cwe", ()
    real = [v for v in values if v not in CWE_PLACEHOLDERS]
    if not real:
        return None, "placeholder_only", ()
    ids: set[str] = set()
    mapped: list[str] = []
    for value in real:
        m = _CWE_ID.fullmatch(value)
        if not m:
            return None, "malformed_cwe", ()
        cwe = m.group(1)
        if graph.is_deprecated_weakness(cwe):
            replacement = DEPRECATED_REPLACEMENT[cwe]
            if replacement is None:
                return None, "deprecated_no_replacement", (f"CWE-{cwe}",)
            mapped.append(f"CWE-{cwe}")
            cwe = replacement
        ids.add(cwe)
    mapped_from = tuple(sorted(mapped, key=lambda s: int(s[4:])))
    if len(ids) > 1:
        return None, "multi_cwe", mapped_from
    (cwe,) = ids
    if graph.is_category(cwe):
        return None, "category", mapped_from
    if graph.is_view(cwe):
        return None, "view", mapped_from
    if not graph.in_view(cwe):
        return None, "not_in_view_1000", mapped_from
    return f"CWE-{cwe}", "ok", mapped_from


def _neighbours(cwe: str, graph: CweLookup) -> frozenset[str]:
    return graph.parents(cwe) | graph.children(cwe)


def cwe_distance(a: str, b: str, graph: CweLookup) -> int | None:
    """Shortest undirected ChildOf distance in view 1000 if it is at most 2, else None. Bare IDs."""
    if a == b:
        return 0
    near = _neighbours(a, graph)
    if b in near:
        return 1
    if any(b in _neighbours(n, graph) for n in near):
        return 2
    return None


def cwe_relation(pred: str, gold: str, graph: CweLookup) -> str:
    """Key into DIRECTION_AWARE_CREDIT. Bare IDs; gold must be a live view-1000 weakness."""
    if not graph.in_view(pred):
        return "far"
    d = cwe_distance(pred, gold, graph)
    if d == 0:
        return "exact"
    if d == 1:
        return "child" if pred in graph.children(gold) else "parent"
    if d == 2:
        if pred in graph.ancestors(gold):
            return "ancestor_2"
        if pred in graph.descendants(gold):
            return "descendant_2"
        if any(pred in graph.children(p) for p in graph.parents(gold)):
            return "sibling"
        return "coparent"
    return "far"


def hierarchy_score(pred: str, gold: str, graph: CweLookup, schedule: str = EXACT_ID_SCHEDULE) -> Fraction:
    """Exact-ID hierarchy credit in [0, 1]. 'CWE-n' strings; anything else as pred scores 0."""
    m, g = _CWE_ID.fullmatch(pred.strip().upper()), _CWE_ID.fullmatch(gold)
    if g is None or not graph.in_view(g.group(1)):
        raise ValueError(f"gold {gold!r} is not a live view-{CWE_VIEW} weakness")
    if m is None or not graph.in_view(m.group(1)):
        return Fraction(0)
    p, g = m.group(1), g.group(1)
    if schedule == "symmetric":
        return SYMMETRIC_CREDIT.get(cwe_distance(p, g, graph), Fraction(0))
    if schedule == "direction_aware":
        return DIRECTION_AWARE_CREDIT[cwe_relation(p, g, graph)]
    raise ValueError(f"unknown schedule {schedule!r}")


def siblings(cwe: str, graph: CweLookup) -> list[str]:
    """Other children of any view-1000 parent (2 ChildOf edges away), excluding the CWE's own
    ancestors and descendants. Bare IDs in, 'CWE-n' out, sorted numerically."""
    found: set[str] = set()
    for parent in graph.parents(cwe):
        found |= graph.children(parent)
    found -= {cwe} | graph.ancestors(cwe) | graph.descendants(cwe)
    return [f"CWE-{c}" for c in sorted(found, key=int)]


# ---------------------------------------------------------------------------
# One function per CVE
# ---------------------------------------------------------------------------

_NOT_A_NAME = frozenset(
    "if else for while do switch case return sizeof typeof alignof _Alignof __alignof__ defined "
    "__attribute__ __attribute __declspec alignas _Alignas decltype __typeof__ asm __asm__ "
    "static_assert _Static_assert noexcept throw "
    "int char void long short unsigned signed float double bool const volatile static inline "
    "extern struct union enum register auto".split()
)
_NAME_BEFORE_PAREN = re.compile(r"((?:[A-Za-z_]\w*\s*::\s*)*~?\s*[A-Za-z_]\w*)\s*$")
_FIRST_ARG = re.compile(r"\s*([A-Za-z_]\w*)")
_MACRO = re.compile(r"[A-Z][A-Z0-9_]+")
_HEADER_LIMIT = 2_000


def extract_function_names(func: str) -> tuple[str, ...]:
    """Candidate names from the header (text before the first '{').

    Every identifier directly before a depth-0 '(' counts; qualified names also contribute
    their last component, and ALL-CAPS macros (PHP_FUNCTION(x), SYSCALL_DEFINE4(f, ...))
    their first argument. Extra candidates are harmless: they only break ties within one CVE.
    """
    masked, ok = mask_comments(func)
    text = (masked if ok else func)[:_HEADER_LIMIT]
    brace = text.find("{")
    header = text if brace < 0 else text[:brace]
    names: list[str] = []
    depth = 0
    for pos, ch in enumerate(header):
        if ch == "(":
            if depth == 0 and (m := _NAME_BEFORE_PAREN.search(header[:pos])):
                full = re.sub(r"\s+", "", m.group(1))
                last = full.rsplit("::", 1)[-1].lstrip("~")
                if last not in _NOT_A_NAME:
                    names += [full, last]
                    if _MACRO.fullmatch(last) and (arg := _FIRST_ARG.match(header, pos + 1)):
                        if arg.group(1) not in _NOT_A_NAME:
                            names.append(arg.group(1))
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return tuple(dict.fromkeys(names))


def name_in_description(names: tuple[str, ...], description: str) -> bool:
    """Whole-word, case-sensitive match of any candidate name."""
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", description) for n in names)


class Candidate(NamedTuple):
    pair_id: str
    vuln_norm_hash: str
    name_in_desc: bool


def choose_function(cve_id: str, candidates: list[Candidate]) -> tuple[str, str]:
    """Pick one surviving pair per CVE: a function named in the NVD description first, then
    the lowest stable_rank(cve_id, vuln_norm_hash), with pair_id as the final tie-break."""
    if len(candidates) == 1:
        return candidates[0].pair_id, "single"
    named = [c for c in candidates if c.name_in_desc]
    if len(named) == 1:
        return named[0].pair_id, "name_in_description"
    pool, rule = (named, "name_in_description+hash") if named else (candidates, "hash")
    best = min(pool, key=lambda c: (stable_rank(cve_id, c.vuln_norm_hash), c.pair_id))
    return best.pair_id, rule
