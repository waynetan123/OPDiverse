"""Score the probe's generations and apply the pre-registered decision rule.

Recall is measured against a permutation control: each answer scored against every *other*
CVE's gold. That is what the model's answering habits earn without any CVE-specific memory.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from etl import pinned, verifiers
from etl.cwe_graph import CweGraph, load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths

UNIT = 8  # every probe score is a multiple of 1/8 (exact 0/1, CVSS k/8, hierarchy credit in eighths)
PRIMARY = ("cwe_exact", "cvss")
SECONDARY = ("cwe_hierarchy",)
OUTCOMES = ("not detected", "material", "large")


# ---------------------------------------------------------------------------
# Per-item scores
# ---------------------------------------------------------------------------


def score_items(requests: list[dict], generations: list[dict], facts: dict[str, dict], graph: CweGraph) -> list[dict]:
    by_id = {g["request_id"]: g for g in generations}
    missing = sorted({r["request_id"] for r in requests} - by_id.keys())
    extra = sorted(by_id.keys() - {r["request_id"] for r in requests})
    if missing or extra:
        raise SystemExit(f"generations don't match requests: {len(missing)} missing, {len(extra)} unexpected")
    for r in requests:
        if by_id[r["request_id"]]["prompt_sha256"] != hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest():
            raise SystemExit(f"{r['request_id']}: generation was made from a different prompt")
    items = []
    for cve_id in sorted({r["cve_id"] for r in requests}):
        fact = facts[cve_id]
        cwe_gen, cvss_gen = by_id[f"{cve_id}:cwe"], by_id[f"{cve_id}:cvss"]
        cwe = verifiers.verify_exact_id(cwe_gen["text"], fact["cwe"], graph)
        cvss = verifiers.verify_cvss(cvss_gen["text"], fact["cvss_vector"])
        items.append({
            "cve_id": cve_id,
            "gold_cwe": fact["cwe"],
            "gold_cvss": fact["cvss_vector"],
            "cwe_reply": cwe_gen["text"],
            "cwe_parsed": cwe.parsed,
            "cwe_parse_ok": cwe.parse_ok,
            "cwe_strict_ok": cwe.strict_ok,
            "cwe_exact": str(cwe.metric),
            "cwe_hierarchy": str(cwe.dense),
            "cwe_finish_reason": cwe_gen["finish_reason"],
            "cvss_reply": cvss_gen["text"],
            "cvss_parsed": cvss.parsed,
            "cvss_parse_ok": cvss.parse_ok,
            "cvss_strict_ok": cvss.strict_ok,
            "cvss": str(cvss.metric),
            "cvss_finish_reason": cvss_gen["finish_reason"],
        })
    return items


# ---------------------------------------------------------------------------
# Observed vs control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measure:
    """answers[i] (None = parse failure) and golds[i], and s(answer, gold) in eighths."""
    answers: list[str | None]
    golds: list[str]
    score8: Callable[[str, str], int]


def _cvss8(answer: str, gold: str) -> int:
    return sum(a == g for a, g in zip(answer.split("/"), gold.split("/")))


def measures(items: list[dict], graph: CweGraph, strict: bool = False) -> dict[str, Measure]:
    def answer(item: dict, prefix: str) -> str | None:
        ok = item[f"{prefix}_strict_ok"] if strict else item[f"{prefix}_parse_ok"]
        return item[f"{prefix}_parsed"] if ok else None

    cwe = [answer(i, "cwe") for i in items]
    cvss = [answer(i, "cvss") for i in items]
    gold_cwe = [i["gold_cwe"] for i in items]
    gold_cvss = [i["gold_cvss"] for i in items]
    hier = lambda a, g: int(pinned.hierarchy_score(a, g, graph) * UNIT)  # noqa: E731
    return {
        "cwe_exact": Measure(cwe, gold_cwe, lambda a, g: UNIT * (a == g)),
        "cvss": Measure(cvss, gold_cvss, _cvss8),
        "cwe_hierarchy": Measure(cwe, gold_cwe, hier),
    }


class _Scorer:
    """s(answer, gold) in eighths as a matrix over the distinct answers and golds, so the
    permutation and bootstrap loops are plain integer lookups."""

    def __init__(self, m: Measure):
        golds = sorted(set(m.golds))
        col = {g: k for k, g in enumerate(golds)}
        zero = [0] * len(golds)
        self.row_of = {a: [m.score8(a, g) for g in golds] for a in sorted({a for a in m.answers if a is not None})}
        self.item_row = [zero if a is None else self.row_of[a] for a in m.answers]
        self.item_col = [col[g] for g in m.golds]
        self.answers, self.golds = m.answers, m.golds
        self.col = col

    def observed8(self, idx: list[int], gold_idx: list[int] | None = None) -> int:
        rows, cols = self.item_row, self.item_col
        gi = idx if gold_idx is None else gold_idx
        return sum(rows[i][cols[j]] for i, j in zip(idx, gi))

    def pairs8(self, idx: list[int]) -> int:
        """Sum of s(answer_i, gold_j) over all ordered (i, j) in idx, including i == j."""
        answers = Counter(self.answers[i] for i in idx if self.answers[i] is not None)
        golds = Counter(self.item_col[i] for i in idx)
        return sum(na * ng * self.row_of[a][g] for a, na in answers.items() for g, ng in golds.items())


def observed_and_control(m: Measure) -> tuple[Fraction, Fraction]:
    sc = _Scorer(m)
    idx = list(range(len(m.golds)))
    n = len(idx)
    diag = sc.observed8(idx)
    return Fraction(diag, UNIT * n), Fraction(sc.pairs8(idx) - diag, UNIT * n * (n - 1))


def permutation_p(m: Measure, permutations: int | None = None, seed: int | None = None) -> float:
    """One-sided: how often a random reassignment of gold to answers scores at least as well.
    The control is identical under every permutation, so comparing observed sums suffices."""
    permutations = pinned.PROBE_PERMUTATIONS if permutations is None else permutations
    sc = _Scorer(m)
    idx = list(range(len(m.golds)))
    target = sc.observed8(idx)
    rng = random.Random(pinned.PROBE_SEED if seed is None else seed)
    perm = idx[:]
    hits = 0
    for _ in range(permutations):
        rng.shuffle(perm)
        hits += sc.observed8(idx, perm) >= target
    return (1 + hits) / (1 + permutations)


def bootstrap_ci(m: Measure, resamples: int | None = None, seed: int | None = None) -> tuple[float, float]:
    """95% percentile interval of the margin over item resamples."""
    resamples = pinned.PROBE_BOOTSTRAP if resamples is None else resamples
    sc = _Scorer(m)
    n = len(m.golds)
    rng = random.Random(pinned.PROBE_SEED if seed is None else seed)
    margins = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diag = sc.observed8(idx)
        margins.append(diag / (UNIT * n) - (sc.pairs8(idx) - diag) / (UNIT * n * (n - 1)))
    margins.sort()
    return margins[int(0.025 * resamples)], margins[int(0.975 * resamples) - 1]


def classify(margin: Fraction, p: float) -> str:
    if p < pinned.PROBE_ALPHA and margin >= pinned.PROBE_LARGE:
        return "large"
    if p < pinned.PROBE_ALPHA and margin >= pinned.PROBE_MATERIAL:
        return "material"
    return "not detected"


def analyse(m: Measure, with_ci: bool = True) -> dict:
    observed, control = observed_and_control(m)
    p = permutation_p(m)
    out = {
        "observed": float(observed),
        "control": float(control),
        "margin": float(observed - control),
        "margin_exact": str(observed - control),
        "p": p,
        "parse_rate": sum(a is not None for a in m.answers) / len(m.answers),
        "outcome": classify(observed - control, p),
    }
    if with_ci:
        out["margin_ci95"] = bootstrap_ci(m)
    return out


# ---------------------------------------------------------------------------
# Plan's constant baselines, recalled CVEs, report
# ---------------------------------------------------------------------------


def constant_baselines(items: list[dict], base: dict, graph: CweGraph) -> dict:
    n = len(items)
    cwe = base["most_frequent_cwe"]["cwe"]
    hier_cwe = base["exact_id_hierarchy"]["adopted_schedule_top"][0]["cwe"]
    vec = base["cvss_majority"]["vector"]
    golds = Counter(i["gold_cwe"] for i in items)
    oracle_cwe, oracle_n = min(golds.items(), key=lambda kv: (-kv[1], int(kv[0][4:])))
    return {
        "cwe_exact": {"answer": cwe, "score": sum(i["gold_cwe"] == cwe for i in items) / n},
        "cwe_hierarchy": {"answer": hier_cwe,
                          "score": float(sum(pinned.hierarchy_score(hier_cwe, i["gold_cwe"], graph) for i in items) / n)},
        "cvss": {"answer": vec, "score": sum(_cvss8(vec, i["gold_cvss"]) for i in items) / (UNIT * n)},
        "descriptive_best_constant_on_this_sample": {"answer": oracle_cwe, "cwe_exact": oracle_n / n},
    }


def recalled(items: list[dict]) -> list[dict]:
    """CVEs answered exactly right with something other than the model's habitual answer."""
    def modal(key: str, ok: str) -> str | None:
        counts = Counter(i[key] for i in items if i[ok])
        return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counts else None

    modal_cwe, modal_cvss = modal("cwe_parsed", "cwe_parse_ok"), modal("cvss_parsed", "cvss_parse_ok")
    out = []
    for i in items:
        reasons = []
        if i["cwe_exact"] == "1" and i["cwe_parsed"] != modal_cwe:
            reasons.append("cwe")
        if i["cvss"] == "1" and i["cvss_parsed"] != modal_cvss:
            reasons.append("cvss")
        if reasons:
            out.append({"cve_id": i["cve_id"], "reasons": reasons})
    return out


def run(paths: Paths) -> dict:
    with open(paths.probe_requests, encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    with open(paths.probe_generations, encoding="utf-8") as f:
        generations = [json.loads(line) for line in f]
    with open(paths.facts, encoding="utf-8") as f:
        facts = {r["cve_id"]: r for r in map(json.loads, f)}
    base = json.loads(paths.baselines_json.read_text(encoding="utf-8"))
    graph = load_cwe_graph(paths.cwe_xml)

    items = score_items(requests, generations, facts, graph)
    lenient, strict = measures(items, graph), measures(items, graph, strict=True)
    results = {name: analyse(lenient[name]) for name in PRIMARY + SECONDARY}
    results_strict = {name: analyse(strict[name], with_ci=False) for name in PRIMARY + SECONDARY}
    outcome = max((results[name]["outcome"] for name in PRIMARY), key=OUTCOMES.index)
    run_meta = json.loads(paths.probe_run_meta.read_text(encoding="utf-8")) if paths.probe_run_meta.exists() else None
    if run_meta and run_meta["requests_sha256"] != file_sha256(paths.probe_requests):
        raise SystemExit("run_meta.json was produced from a different requests.jsonl")
    report = {
        "outcome": outcome,
        "n": len(items),
        "parser_version": verifiers.PARSER_VERSION,
        "thresholds": {"material": str(pinned.PROBE_MATERIAL), "large": str(pinned.PROBE_LARGE), "alpha": pinned.PROBE_ALPHA},
        "lenient": results,
        "strict": results_strict,
        "constant_baselines": constant_baselines(items, base, graph),
        "recalled": len(recalled(items)),
        "sources": {
            "requests_sha256": file_sha256(paths.probe_requests),
            "generations_sha256": file_sha256(paths.probe_generations),
            "facts_sha256": file_sha256(paths.facts),
            "baselines_sha256": file_sha256(paths.baselines_json),
        },
        "run_meta": run_meta,
    }
    for path, rows in ((paths.probe_scores, items), (paths.probe_recalled, recalled(items))):
        versioned(path).write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    versioned(paths.probe_report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    versioned(paths.probe_report_md).write_text(render_markdown(report), encoding="utf-8")
    return report


def versioned(path: Path) -> Path:
    """The probe was scored and decided under parsers v1 (step 3); a later parser version writes its re-score
    beside those files (report_v2.md, ...), so both stay on record."""
    return path if verifiers.PARSER_VERSION == "v1" else path.with_name(f"{path.stem}_{verifiers.PARSER_VERSION}{path.suffix}")


ACTIONS = {
    "not detected": "Memorisation was probed and not detected; proceed unchanged.",
    "material": "Report the margin and add the pre-registered sensitivity run of the primary test "
                "restricted to test CVEs not in recalled.jsonl.",
    "large": "Report the margin; reconsider the backbone, or extend the fact table forward with newer NVD entries.",
}
LABELS = {"cwe_exact": "CWE exact", "cvss": "CVSS agreement", "cwe_hierarchy": "*CWE hierarchy (secondary)*"}


def render_markdown(r: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    base = r["constant_baselines"]
    rows = []
    for name in PRIMARY + SECONDARY:
        a, s = r["lenient"][name], r["strict"][name]
        lo, hi = a["margin_ci95"]
        rows.append(
            f"| {LABELS[name]} | {a['observed']:.3f} | {a['control']:.3f} | {pct(a['margin'])} pts ({pct(lo)}, {pct(hi)}) "
            f"| {a['p']:.4f} | {a['outcome']} | {pct(a['observed'] - base[name]['score'])} pts vs {base[name]['answer']} "
            f"({base[name]['score']:.3f}) | {a['parse_rate']:.2f} / {s['parse_rate']:.2f} | {pct(s['margin'])} pts |"
        )
    meta = r["run_meta"] or {}
    desc = base["descriptive_best_constant_on_this_sample"]
    return "\n".join([
        "# Step 3: contamination probe", "",
        f"**Outcome: {r['outcome']}.** {ACTIONS[r['outcome']]}", "",
        f"{r['n']} test CVEs, the raw backbone given only each CVE ID. Rule: an outcome is *material* at a recall margin "
        f"≥ {pct(float(Fraction(r['thresholds']['material'])))} points and *large* at ≥ "
        f"{pct(float(Fraction(r['thresholds']['large'])))}, each with p < {r['thresholds']['alpha']}, on either primary measure. "
        "Margin = observed − control, where the control scores each answer against every other CVE's gold.", "",
        "| Measure | Observed | Control | Margin (95% CI) | p | Outcome | vs training-era constant | Parse rate (lenient / strict) | Strict margin |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows, "",
        f"Descriptive only (uses test labels): the best constant CWE on this sample is {desc['answer']} "
        f"at {desc['cwe_exact']:.3f} exact.", "",
        f"CVEs answered exactly right with a non-habitual answer (`recalled.jsonl`): {r['recalled']}.", "",
        f"Parser version {r['parser_version']}. vLLM {meta.get('versions', {}).get('vllm', '?')}, "
        f"GPUs {', '.join(meta.get('gpus', [])) or '?'}, determinism recheck mismatches: "
        f"{len(meta.get('determinism_recheck', {}).get('mismatched', [])) if meta else '?'}.",
        "",
    ])
