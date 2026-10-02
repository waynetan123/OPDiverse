"""Build distill_self.jsonl (one target per non-test item) and evaluate the substitution trigger."""

from __future__ import annotations

from fractions import Fraction

from etl.build import jsonl_bytes, write_json
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from generators.bank.files import generations_for
from generators.progress import track

from . import rationale
from .files import SessionFiles
from .prepare import load_bank, load_results


def run(paths: Paths) -> dict:
    files = SessionFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    graph = load_cwe_graph(paths.cwe_xml)
    requests, results = load_results(files.rationale_requests)
    retry: dict[str, dict] = {}
    sources = {"rationale_requests_sha256": file_sha256(files.rationale_requests),
               "rationale_generations_sha256": file_sha256(generations_for(files.rationale_requests))}
    if files.rationale_retry_requests.exists():
        retry_requests, retry_results = load_results(files.rationale_retry_requests)
        retry = {r["item_id"]: retry_results[r["request_id"]]["outputs"][0] for r in retry_requests}
        sources["rationale_retry_requests_sha256"] = file_sha256(files.rationale_retry_requests)
        sources["rationale_retry_generations_sha256"] = file_sha256(generations_for(files.rationale_retry_requests))
        requests_all = requests + retry_requests
    else:
        requests_all = requests
    if any(r["bank_sha256"] != bank_sha for r in requests_all):
        raise SystemExit("rationale requests were built from a different bank_nontest.jsonl")
    if sorted(r["item_id"] for r in requests) != sorted(by_id):
        raise SystemExit("rationale requests do not cover exactly the non-test bank")

    rows = [rationale.decide(by_id[r["item_id"]], results[r["request_id"]]["outputs"][0], retry.get(r["item_id"]), graph)
            for r in track(requests, "Rationale validity")]
    extra = sorted(set(retry) - {r["item_id"] for r in rows if r["a1_reason"]})
    if extra:
        raise SystemExit(f"{len(extra)} regenerations for items whose first attempt was valid, e.g. {extra[:3]}")
    summary = rationale.summarise(rows)
    sources["bank_nontest_sha256"] = bank_sha
    files.distill_self.write_bytes(jsonl_bytes(rows))
    sources["distill_self_sha256"] = file_sha256(files.distill_self)
    report = {**summary, "n_items": len(rows), "sources": sources}
    write_json(files.substitution, summary)
    write_json(files.rationale_report_json, report)
    files.rationale_report_md.write_text(render_markdown(report), encoding="utf-8")
    return report


BAND_LABELS = {"as_specified": "≥ 50%", "report_quality": "10–50%", "floored": "**< 10%**"}


def render_markdown(r: dict) -> str:
    pct = lambda s: f"{100 * float(Fraction(s)):.1f}%"  # noqa: E731
    rows = []
    for t, s in r["types"].items():
        src = s["sources"]
        rows.append(f"| {t} | {s['n']:,} | {pct(s['pass_rate_a1'])} | {src['self_a2']:,} | {pct(s['fallback_rate'])} "
                    f"| {pct(s['hint_leak_rate_a1'])} | {BAND_LABELS[s['band']]} | {'**yes**' if s['substitution_fires'] else 'no'} |")
    reasons = [f"- {t}: attempt 1 {s['a1_reasons'] or '—'}; regeneration {s['a2_reasons'] or '—'}" for t, s in r["types"].items()]
    sub = r["substituted_types"]
    verdict = ("none" if not sub else ", ".join(sub)) + (
        f". **{len(sub)} ≥ {r['rule']['drop_at_types']} types: distill-self leaves the primary test** (run it on SFT, DPO, GRPO)."
        if r["drop_distill_self_from_primary_test"] else ".")
    return "\n".join([
        "# Step 5: distill-self rationales", "",
        f"{r['n_items']:,} non-test items. A rationale is surface-valid if it was not cut off, its last line is the gold "
        "answer, it has reasoning above that line, and the reasoning does not restate the hint. Invalid first attempts are "
        "regenerated once; a second failure falls back to the gold answer alone.", "",
        f"Substitution rule (per type): first-attempt pass rate below {pct(r['rule']['pass_rate_below'])}, or fallback rate "
        f"above {pct(r['rule']['or_fallback_rate_above'])}. Substituted types draw their rationales from the external model at step 6.", "",
        "| Type | Items | Pass (attempt 1) | Recovered by regeneration | Fallback to gold only | Restates hint (attempt 1) | Band | Substitution |",
        "|---|---|---|---|---|---|---|---|",
        *rows, "",
        f"**Substituted types: {verdict}**", "",
        "## Failure reasons", "",
        *reasons, "",
    ])
