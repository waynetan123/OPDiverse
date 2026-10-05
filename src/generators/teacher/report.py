"""step6_report.{json,md} (the DPO audit line, usage and cost) and the pilot report."""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction

from etl.build import write_json
from etl.paths import Paths
from frozen_model.prepare import load_bank

from ..bank.build import load_inputs
from ..bank.check import code_lines
from ..bank.files import run_meta_for
from . import dpo
from .files import TeacherFiles
from .prepare import pilot_items
from .results import load_results

# claude-opus-5-5 list prices, $ per million tokens (thinking is billed as output). Used for reporting only.
PRICE_PER_MTOK = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_creation_input_tokens": 5.0,
                  "cache_read_input_tokens": 0.2}


def cost(usage: dict) -> float:
    return sum(usage.get(k, 0) * p for k, p in PRICE_PER_MTOK.items()) / 1e6


def usage_section(files: TeacherFiles) -> dict:
    """Per run (dpo, dpo_retry): requests, token totals and cost, from the runner's run_meta."""
    out = {}
    for retry in (False, True):
        meta_path = run_meta_for(files.requests(retry))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out[meta_path.stem.removesuffix("_run_meta")] = {
                "n_requests": meta["n_requests"], "usage": meta["usage_totals"],
                "cost_usd": round(cost(meta["usage_totals"]), 2), "stop_reasons": meta["stop_reasons"],
                "models": meta["models"], "git": meta["git"], "finished_utc": meta["finished_utc"]}
    return out


def write(paths: Paths, summary: dict) -> dict:
    files = TeacherFiles.of(paths)
    usage = usage_section(files)
    out = {**summary, "usage": usage, "total_cost_usd": round(sum(u["cost_usd"] for u in usage.values()), 2)}
    write_json(files.report_json, out)
    files.report_md.write_text(render_markdown(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------


def pilot_report(paths: Paths) -> dict:
    """First-attempt validity on the pilot, reasons, refusals, and the full run's cost extrapolated from the
    pilot's usage (the pilot holds every item of its CVEs, so the type mix matches the full pool)."""
    files = TeacherFiles.of(paths, pilot=True)
    bank, _ = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    facts, _, graph = load_inputs(paths)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    pilot_cves = len({r["cve_id"] for r in pilot_items(bank)})
    scale = len({r["cve_id"] for r in bank}) / pilot_cves

    requests, results = load_results(files.requests())
    per_type: dict[str, dict] = {}
    for r in requests:
        item, row = by_id[r["item_id"]], results[r["item_id"]]
        s = per_type.setdefault(item["type"], {"n": 0, "valid": 0, "reasons": Counter(), "refusal_categories": Counter()})
        reason = dpo.validate(item, row, code[item["cve_id"]], graph)[2]
        s["n"] += 1
        s["valid"] += reason is None
        if reason:
            s["reasons"][reason] += 1
        if reason == "refusal":
            s["refusal_categories"][str(row["stop_category"])] += 1
    meta = json.loads(run_meta_for(files.requests()).read_text(encoding="utf-8"))
    out = {
        "pilot_cves": pilot_cves, "scale_to_full_run": scale,
        "types": {t: {"n": s["n"], "valid_a1": s["valid"], "reasons": dict(sorted(s["reasons"].items())),
                      "refusal_categories": dict(sorted(s["refusal_categories"].items()))} for t, s in per_type.items()},
        "cost_usd": {"pilot": round(cost(meta["usage_totals"]), 2),
                     "full_run_estimate": round(cost(meta["usage_totals"]) * scale, 2), "usage": meta["usage_totals"]},
    }
    write_json(files.pilot_report_json, out)
    files.pilot_report_md.write_text(render_pilot(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(s) -> str:
    return "—" if s is None else f"{100 * float(Fraction(s)):.1f}%"


BAND_LABELS = {"as_specified": "≥ 50%", "report_quality": "10–50%", "floored": "**< 10%**", None: "rule-defined"}


def render_markdown(r: dict) -> str:
    dp = r["dpo"]["types"]
    rows = [
        f"| {t} | {s['n']:,} | {_pct(s['constructible_a1'])} | {s['sources']['model_a2']:,} | {_pct(s['rule_fallback_rate'])} "
        f"| {BAND_LABELS[s['band']]} | {s['rejected_dense']} |"
        for t, s in dp.items()]
    usage = [f"| {k} | {u['n_requests']:,} | {u['usage'].get('input_tokens', 0):,} | {u['usage'].get('output_tokens', 0):,} "
             f"| ${u['cost_usd']:,.2f} |" for k, u in r["usage"].items()]
    reasons = [f"- {t}: attempt 1 {s['a1_reasons'] or '—'}; regeneration {s['a2_reasons'] or '—'}; "
               f"refusal categories {s['refusal_categories'] or '—'}" for t, s in dp.items() if not s["rule_defined"]]
    return "\n".join([
        "# Step 6: DPO rejected answers", "",
        f"{r['n_items']:,} non-test items; external model `{r['external_model']['model']}`, effort "
        f"`{r['external_model']['effort']}`. Chosen is the gold target; rejected is a near miss one unit of error "
        "from gold, in the same format.", "",
        "Constructible = a valid near miss from the external model at the first attempt, without the rule. MCQ sends no "
        "request: its rejected letter is the distractor closest to gold in the hierarchy.", "",
        "| Type | Items | Constructible (attempt 1) | Recovered by regeneration | Rule fallback | Band | Rejected scores (dense: count) |",
        "|---|---|---|---|---|---|---|",
        *rows, "",
        "## Failure reasons", "",
        *reasons, "",
        "## Usage", "",
        "| Run | Requests | Input tokens | Output tokens | Cost |", "|---|---|---|---|---|",
        *usage, "",
        f"Total: **${r['total_cost_usd']:,.2f}** at list price.", "",
    ])


def render_pilot(r: dict) -> str:
    rows = [f"| {t} | {s['n']} | {s['valid_a1']} | {s['reasons'] or '—'} | {s['refusal_categories'] or '—'} |"
            for t, s in r["types"].items()]
    c = r["cost_usd"]
    return "\n".join([
        "# Step 6: DPO pilot", "",
        f"{r['pilot_cves']} non-test CVEs, every non-MCQ item, first attempt only. The pilot decides nothing; it checks "
        "the prompts and refusals, and prices the full run.", "",
        "| Type | Items | Valid | Reasons | Refusal categories |", "|---|---|---|---|---|",
        *rows, "",
        f"Cost: ${c['pilot']:,.2f} for the pilot; full run, first attempt ≈ **${c['full_run_estimate']:,.2f}** "
        "(regenerations add their share of the invalid rate).", "",
    ])
