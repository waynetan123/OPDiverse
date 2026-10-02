"""step5_report.{json,md}: the audit and the rationale outcome side by side, for the decision record."""

from __future__ import annotations

import json
from fractions import Fraction

from etl.build import write_json
from etl.paths import Paths

from .files import SessionFiles


def run(paths: Paths) -> dict:
    files = SessionFiles.of(paths)
    for path, cmd in ((files.audit_report_json, "audit"), (files.rationale_report_json, "rationales")):
        if not path.exists():
            raise SystemExit(f"{path.name} is missing: run `python -m frozen_model {cmd}` first")
    audit = json.loads(files.audit_report_json.read_text(encoding="utf-8"))
    rat = json.loads(files.rationale_report_json.read_text(encoding="utf-8"))
    out = {
        "grpo": {t: {"cap": s["chosen_cap"], "band": s["band"],
                     "live_dense": s["by_cap"][str(s["chosen_cap"])]["live_dense"]} for t, s in audit["types"].items()},
        "distill_self": {t: {"pass_rate_a1": s["pass_rate_a1"], "fallback_rate": s["fallback_rate"], "band": s["band"],
                             "substitution_fires": s["substitution_fires"]} for t, s in rat["types"].items()},
        "substituted_types": rat["substituted_types"],
        "drop_distill_self_from_primary_test": rat["drop_distill_self_from_primary_test"],
        "sources": {**audit["sources"], **rat["sources"]},
    }
    write_json(files.report_json, out)
    files.report_md.write_text(render_markdown(out), encoding="utf-8")
    return out


def render_markdown(r: dict) -> str:
    pct = lambda s: f"{100 * float(Fraction(s)):.1f}%"  # noqa: E731
    rows = []
    for t in r["grpo"].keys() | r["distill_self"].keys():
        g, d = r["grpo"].get(t), r["distill_self"].get(t)
        rows.append((t, f"{g['cap']} / {pct(g['live_dense'])} / {g['band']}" if g else "—",
                     f"{pct(d['pass_rate_a1'])} / {pct(d['fallback_rate'])} / {'substituted' if d['substitution_fires'] else 'self'}"
                     if d else "—"))
    order = ["mcq", "exact_id", "cvss", "find_error", "line_loc"]
    rows.sort(key=lambda x: order.index(x[0]) if x[0] in order else len(order))
    sub = r["substituted_types"]
    return "\n".join([
        "# Step 5: frozen-model session", "",
        "| Type | GRPO: cap / live groups / band | distill-self: pass (attempt 1) / fallback / source |",
        "|---|---|---|",
        *(f"| {a} | {b} | {c} |" for a, b, c in rows), "",
        f"Substituted types: {', '.join(sub) or 'none'}."
        + (" **distill-self leaves the primary test.**" if r["drop_distill_self_from_primary_test"] else ""), "",
        "Details: `audit_report.md`, `rationale_report.md`. DPO and distill-external audit lines are computed at step 6.", "",
    ])
