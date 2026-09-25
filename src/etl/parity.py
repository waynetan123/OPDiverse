"""CVSS v3.0 / v3.1 parity check: does the majority baseline agree equally well on both?

Report only; no threshold is pinned. Version is confounded with time (v3.1 starts mid-2019),
so the check is repeated on the years where both versions occur.
"""

from __future__ import annotations

import math
from collections import Counter

from . import pinned

Z95 = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom
    return (centre - half, centre + half)


def majority(vectors: list[dict[str, str]]) -> dict[str, str]:
    """Most common value per component; ties go to the earlier value in CVSS_ALLOWED."""
    out = {}
    for key in pinned.CVSS_ORDER:
        counts = Counter(v[key] for v in vectors)
        out[key] = min(pinned.CVSS_ALLOWED[key], key=lambda val: (-counts[val], pinned.CVSS_ALLOWED[key].index(val)))
    return out


def _mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return mean, var, mean - Z95 * se, mean + Z95 * se


def _agreement(vectors: list[dict[str, str]], ref: dict[str, str]) -> dict:
    n = len(vectors)
    if n == 0:
        return {"n": 0}
    per_row = [sum(v[k] == ref[k] for k in pinned.CVSS_ORDER) / len(pinned.CVSS_ORDER) for v in vectors]
    mean, var, lo, hi = _mean_ci(per_row)
    per_component = {}
    for k in pinned.CVSS_ORDER:
        hits = sum(v[k] == ref[k] for v in vectors)
        per_component[k] = {"agree": hits / n, "ci95": wilson(hits, n)}
    return {"n": n, "mean": mean, "var": var, "ci95": (lo, hi), "per_component": per_component}


def _compare(rows: list[dict], pooled: dict[str, str]) -> dict:
    groups = {v: [pinned.parse_cvss_v3(r["cvss_vector"])[1] for r in rows if r["cvss_version"] == v]
              for v in pinned.CVSS_VERSIONS}
    out = {"years": sorted({r["published"][:4] for r in rows})}
    for version, vectors in groups.items():
        own = majority(vectors) if vectors else None
        out[version] = {
            "vs_pooled_majority": _agreement(vectors, pooled),
            "own_majority": pinned.canonical_cvss(own) if own else None,
            "vs_own_majority": _agreement(vectors, own) if own else {"n": 0},
        }
    a, b = out["3.0"]["vs_pooled_majority"], out["3.1"]["vs_pooled_majority"]
    if a.get("n", 0) > 1 and b.get("n", 0) > 1:
        diff = b["mean"] - a["mean"]
        se = math.sqrt(a["var"] / a["n"] + b["var"] / b["n"])
        out["difference_3.1_minus_3.0"] = {"mean": diff, "ci95": (diff - Z95 * se, diff + Z95 * se)}
    return out


def parity(rows: list[dict]) -> dict:
    pooled = majority([pinned.parse_cvss_v3(r["cvss_vector"])[1] for r in rows])
    by_year: dict[str, set[str]] = {}
    for r in rows:
        by_year.setdefault(r["published"][:4], set()).add(r["cvss_version"])
    overlap = {y for y, versions in by_year.items() if len(versions) == 2}
    return {
        "pooled_majority": pinned.canonical_cvss(pooled),
        "all_years": _compare(rows, pooled),
        "overlap_years": _compare([r for r in rows if r["published"][:4] in overlap], pooled),
    }
