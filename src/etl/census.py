"""Filtering census: a ledger of what survives each stage, and its markdown rendering."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from .primevul import Pair


def nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def quantiles(values: Iterable[float], qs: Sequence[float] = (0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)) -> list[list]:
    """[[name, value], ...] in quantile order."""
    ordered = sorted(values)
    return [[f"p{round(q * 100)}", nearest_rank(ordered, q)] for q in qs] if ordered else []


class Ledger:
    """Tracks surviving pairs through the census. A CVE is alive while any of its pairs is."""

    def __init__(self, pairs: list[Pair]):
        self.cve_of = {p.pair_id: p.cve_id for p in pairs}
        self.commit_of = {p.pair_id: p.commit_id for p in pairs}
        self.alive: set[str] = set(self.cve_of)
        self.stages: list[dict] = [{"stage": "loaded", "unit": "pair", "after": self.counts(), "dropped": {}}]
        self.pair_fate: dict[str, tuple[str, str]] = {}
        self.cve_fate: dict[str, tuple[str, str]] = {}
        self.snapshots: dict[str, dict[str, int]] = {}

    def counts(self, alive: set[str] | None = None) -> dict[str, int]:
        alive = self.alive if alive is None else alive
        return {
            "pairs": len(alive),
            "cves": len({self.cve_of[p] for p in alive}),
            "commits": len({self.commit_of[p] for p in alive}),
        }

    def alive_cves(self) -> set[str]:
        return {self.cve_of[p] for p in self.alive}

    def alive_by_cve(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for p in sorted(self.alive):
            groups.setdefault(self.cve_of[p], []).append(p)
        return groups

    def _apply(self, stage: str, unit: str, dropped: dict[str, str], order: Sequence[str]) -> None:
        before = self.counts()
        cves_before = self.alive_cves()
        self.alive -= dropped.keys()
        cves_lost = cves_before - self.alive_cves()
        rank = {r: i for i, r in enumerate(order)}
        # A lost CVE is attributed to the earliest reason (in `order`) among its pairs dropped here.
        cve_reason: dict[str, str] = {}
        for p, reason in dropped.items():
            self.pair_fate[p] = (stage, reason)
            c = self.cve_of[p]
            if c in cves_lost and (c not in cve_reason or rank[reason] < rank[cve_reason[c]]):
                cve_reason[c] = reason
        for c, reason in cve_reason.items():
            self.cve_fate[c] = (stage, reason)
        after = self.counts()
        by_pairs = Counter(dropped.values())
        by_cves = Counter(cve_reason.values())
        assert before["pairs"] - after["pairs"] == sum(by_pairs.values()), stage
        assert before["cves"] - after["cves"] == sum(by_cves.values()), stage
        self.stages.append({
            "stage": stage,
            "unit": unit,
            "after": after,
            "dropped": {r: {"pairs": by_pairs[r], "cves": by_cves[r]} for r in order if by_pairs[r]},
        })

    def drop_pairs(self, stage: str, failing: dict[str, str], order: Sequence[str]) -> None:
        """failing: pair_id -> reason. Pairs already dropped are ignored."""
        self._apply(stage, "pair", {p: r for p, r in failing.items() if p in self.alive}, order)

    def drop_cves(self, stage: str, failing: dict[str, str], order: Sequence[str]) -> None:
        """failing: cve_id -> reason. Drops every surviving pair of those CVEs."""
        self._apply(stage, "cve", {p: failing[self.cve_of[p]] for p in self.alive if self.cve_of[p] in failing}, order)

    def snapshot(self, name: str, year_of: Callable[[str], int]) -> None:
        years = Counter(year_of(c) for c in self.alive_cves())
        self.snapshots[name] = {str(y): years[y] for y in sorted(years)}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(c: dict) -> str:
    out = ["# Step 1 filtering census", ""]
    final = c["final"]
    tr = final["time_range"]
    out += [
        "## Outcome", "",
        f"- **Fact table: {final['cves']:,} CVEs** — size band: {final['size_band']}",
        f"- **Time-range gate: {tr['gate']}** — full span {tr['span_years']:.2f} years "
        f"({tr['min'][:10]} → {tr['max'][:10]}); 5th–95th percentile span {tr['p5_p95_span_years']:.2f} years "
        f"({tr['p5'][:10]} → {tr['p95'][:10]})",
        "",
        "## Stages", "",
        "Counts are survivors after each stage. Drops list pairs (and the CVEs whose last pair went at that stage).",
        "",
    ]
    rows = []
    for s in c["stages"]:
        drops = "; ".join(f"{r}: {d['pairs']:,} pairs / {d['cves']:,} CVEs" for r, d in s["dropped"].items()) or "—"
        a = s["after"]
        rows.append((s["stage"], s["unit"], f"{a['cves']:,}", f"{a['pairs']:,}", f"{a['commits']:,}", drops))
    out += [_table(("Stage", "Unit", "CVEs", "Pairs", "Commits", "Dropped"), rows), ""]

    out += ["## Drops if each filter were applied alone", "",
            f"Measured from the `{c['drop_alone_base']}` survivors, so they don't depend on stage order.", ""]
    out += [_table(("Filter", "CVEs", "Pairs"), ((d["filter"], f"{d['cves']:,}", f"{d['pairs']:,}") for d in c["drop_alone"])), ""]

    snaps = c["year_histogram"]
    order = ("after_join", "pre_v3", "post_v3", "final")
    years = sorted({y for s in snaps.values() for y in s})
    out += ["## Publication year, before and after the v3.x stage", "",
            "`pre_v3` is after the CWE stage, `post_v3` after the CVSS v3.x stage.", ""]
    out += [_table(("Year", *order, "dropped at v3.x"),
                   ((y, *(snaps[k].get(y, 0) for k in order), snaps["pre_v3"].get(y, 0) - snaps["post_v3"].get(y, 0))
                    for y in years)), ""]

    kept = c["cvss_kept"]
    out += ["## CVSS", "",
            f"Kept: v3.1 {kept['3.1']:,} (of which {kept['both_versions']:,} also had an NVD v3.0), v3.0 {kept['3.0']:,}.", ""]
    by_year = c["final_cvss_version_by_year"]
    out += [_table(("Year", "v3.0", "v3.1"), ((y, v.get("3.0", 0), v.get("3.1", 0)) for y, v in by_year.items())), ""]

    tcr = c["token_cap_report"]
    out += ["## Token cap", "",
            f"Entering the cap stage: {tcr['entering']['cves']:,} CVEs / {tcr['entering']['pairs']:,} pairs. "
            "Survivors under alternative caps (for comparison only):", ""]
    out += [_table(("Cap", "CVEs", "Pairs"), ((f"{d['cap']:,}", f"{d['cves']:,}", f"{d['pairs']:,}") for d in tcr["caps"])), ""]
    out += ["Final table, token quantiles:", ""]
    for side, q in c["final_tokens"].items():
        out.append(f"- `{side}`: " + ", ".join(f"{k} {v:,}" for k, v in q))
    out.append("")

    sel = c["selection"]
    out += ["## One function per CVE", "",
            f"{sel['cves_with_multiple_candidates']:,} CVEs had more than one surviving candidate. Rules used: "
            + ", ".join(f"{k} {v:,}" for k, v in sel["rules"].items()) + ".", ""]

    out += ["## Final table", ""]
    out += ["**Top CWEs**", "", _table(("CWE", "CVEs"), c["final_cwe_top"]), ""]
    out += ["**Sibling count (CVE-weighted)**", "", _table(("Siblings", "CVEs"), c["final_sibling_counts"]), ""]
    out += ["**Patch size |S|**", "", _table(("Lines", "CVEs"), c["final_patch_size"]), ""]
    out += ["**Patch fraction |S| / code lines**", "", _table(("Fraction", "CVEs"), c["final_patch_fraction"]), ""]
    out += [f"Comment masking fell back to unmasked on {c['final_mask_fallbacks']:,} rows.", ""]
    nvd = c["nvd"]
    out += ["## NVD", "", f"Status of joined records: " + ", ".join(f"{k} {v:,}" for k, v in nvd["vuln_status"].items()) + ".", ""]
    return "\n".join(out)
