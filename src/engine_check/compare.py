"""Score both engines' replies and apply the pre-registered decision rule (step 7).

Per question type there is one score per CVE: the item's `metric` for MCQ, exact-ID, CVSS and line
localisation, and paired accuracy (both labels right) for find-the-error. HF is compared with vLLM pass A;
pass A against pass B is reported alongside as vLLM's own batch noise.

Decision (owner, step 7): **material** if on any type |HF - vLLM A| >= ENGINE_GAP and the paired bootstrap
interval over CVEs at ENGINE_CI excludes 0; otherwise **agree**. Lenient parsers decide; strict parse rates
are reported. A material outcome stops the pipeline before step 8 until its cause is found.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from fractions import Fraction

from etl import pinned, verifiers
from etl.build import read_jsonl, write_json
from etl.census import quantiles
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.tokens import TokenCounter
from frozen_model.prepare import load_bank
from generators.bank.files import generations_for, run_meta_for

from .files import EngineFiles
from .run_hf import body, first_divergence, normalised

ENGINES = ("vllm_a", "vllm_b", "hf")
PAIRS = (("vllm_a", "hf"), ("vllm_a", "vllm_b"))  # (x, y): gap = y - x; the first pair decides


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _check(rows: dict[str, dict], requests: list[dict], requests_sha: str, name: str) -> None:
    if set(rows) != {r["request_id"] for r in requests}:
        raise SystemExit(f"{name} does not cover exactly the engine-check requests")
    for r in requests:
        g = rows[r["request_id"]]
        if g["requests_sha256"] != requests_sha:
            raise SystemExit(f"{name} came from a different requests file")
        if g["prompt_sha256"] != pinned.sha256_text(r["prompt"]):
            raise SystemExit(f"{r['request_id']}: {name} was generated from a different prompt")
        if len(g["outputs"]) != 1:
            raise SystemExit(f"{r['request_id']}: {name} has {len(g['outputs'])} outputs, expected 1")


def load_engines(files: EngineFiles) -> tuple[list[dict], dict[str, dict[str, dict]], dict]:
    """(pass-A requests, engine -> request_id -> row, sources). Exits on any mismatch with the requests."""
    a_path, b_path = files.vllm_requests("a"), files.vllm_requests("b")
    requests = read_jsonl(a_path)
    b_requests = read_jsonl(b_path)
    if sorted(json.dumps(r, sort_keys=True) for r in b_requests) != sorted(json.dumps(r, sort_keys=True) for r in requests):
        raise SystemExit("pass B's requests are not pass A's rows")
    a_sha, b_sha = file_sha256(a_path), file_sha256(b_path)
    out: dict[str, dict[str, dict]] = {}
    sources = {"vllm_a_requests_sha256": a_sha, "vllm_b_requests_sha256": b_sha}
    for engine, path, sha in (("vllm_a", a_path, a_sha), ("vllm_b", b_path, b_sha)):
        gens = generations_for(path)
        if not gens.exists():
            raise SystemExit(f"{gens} is missing: run `python -m evaluate.run_vllm --requests {path}` on the GPU machine")
        out[engine] = {r["request_id"]: r for r in read_jsonl(gens)}
        _check(out[engine], requests, sha, gens.name)
        sources[f"{engine}_generations_sha256"] = file_sha256(gens)
    hf: dict[str, dict] = {}
    for path in files.hf_shard_files():
        for row in read_jsonl(path):
            if row["request_id"] in hf:
                raise SystemExit(f"{row['request_id']} appears in two HF shards")
            hf[row["request_id"]] = row
        sources[f"{path.stem}_sha256"] = file_sha256(path)
    _check(hf, requests, a_sha, "the HF shards")
    out["hf"] = hf
    return requests, out, sources


def run_metas(files: EngineFiles) -> dict:
    metas = {}
    for p in ("a", "b"):
        path = run_meta_for(files.vllm_requests(p))
        metas[f"vllm_{p}"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    for path in files.hf_shard_files():
        meta = files.dir / path.name.replace("_generations.jsonl", "_run_meta.json")
        metas[path.stem.removesuffix("_generations")] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
    return metas


# ---------------------------------------------------------------------------
# Scores and the decision
# ---------------------------------------------------------------------------


def unit_scores(verdicts: dict[str, verifiers.Verdict], items: list[dict]) -> dict[str, dict[str, Fraction]]:
    """type -> cve_id -> score: `metric`, or paired accuracy for find-the-error."""
    out: dict[str, dict[str, Fraction]] = {t: {} for t in pinned.BANK_TYPES}
    for item in items:
        t, cve = item["type"], item["cve_id"]
        if t == "find_error":
            if item["index"] == 0:
                out[t][cve] = verifiers.paired_accuracy(verdicts[item["item_id"]], verdicts[f"{cve}:find_error:1"])
        else:
            out[t][cve] = verdicts[item["item_id"]].metric
    return out


def bootstrap_interval(diffs: list[float], level: Fraction = pinned.ENGINE_CI, resamples: int = pinned.ENGINE_BOOTSTRAP,
                       seed: int = pinned.ENGINE_SEED) -> tuple[float, float]:
    """Percentile interval of the mean paired difference over resampled CVEs (probe.score.bootstrap_ci's style)."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    return means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


def material(gap: Fraction, interval: tuple[float, float]) -> bool:
    """The owner's rule: a gap of at least ENGINE_GAP that the interval separates from zero."""
    lo, hi = interval
    return abs(gap) >= pinned.ENGINE_GAP and (lo > 0 or hi < 0)


def compare_type(x: dict[str, Fraction], y: dict[str, Fraction]) -> dict:
    cves = sorted(x)
    if sorted(y) != cves:
        raise ValueError("the two engines were scored on different CVEs")
    n = len(cves)
    mean_x, mean_y = sum(x.values(), Fraction(0)) / n, sum(y.values(), Fraction(0)) / n
    gap = mean_y - mean_x
    interval = bootstrap_interval([float(y[c] - x[c]) for c in cves])
    return {"n": n, "mean_x": float(mean_x), "mean_y": float(mean_y), "gap": float(gap), "gap_exact": str(gap),
            "interval": list(interval), "y_better": sum(y[c] > x[c] for c in cves),
            "x_better": sum(x[c] > y[c] for c in cves), "material": material(gap, interval)}


def engine_stats(rows: dict[str, dict], verdicts: dict[str, verifiers.Verdict], items: list[dict],
                 stop_ids: set[int]) -> dict[str, dict]:
    out = {}
    for t in pinned.BANK_TYPES:
        chosen = [i for i in items if i["type"] == t]
        outs = [rows[f"eval:{i['item_id']}"]["outputs"][0] for i in chosen]
        n = len(chosen)
        out[t] = {
            "items": n,
            "parse_rate": sum(verdicts[i["item_id"]].parse_ok for i in chosen) / n,
            "strict_rate": sum(verdicts[i["item_id"]].strict_ok for i in chosen) / n,
            "cut_off_rate": sum(o["finish_reason"] == "length" for o in outs) / n,
            "mean_output_tokens": sum(len(body(o["token_ids"], stop_ids)) for o in outs) / n,
        }
    return out


def identical_text(rows_x: dict[str, dict], rows_y: dict[str, dict], items: list[dict]) -> dict[str, float]:
    out = {}
    for t in pinned.BANK_TYPES:
        chosen = [f"eval:{i['item_id']}" for i in items if i["type"] == t]
        out[t] = sum(rows_x[r]["outputs"][0]["text"] == rows_y[r]["outputs"][0]["text"] for r in chosen) / len(chosen)
    return out


def divergence_section(engines: dict[str, dict[str, dict]], stop_ids: set[int]) -> dict:
    """Where HF first leaves vLLM pass A, and whether vLLM's token there was a near-tie under HF's logits."""
    divs = [r["divergence_from_vllm_a"] for r in engines["hf"].values() if r.get("divergence_from_vllm_a")]
    recomputed = sum(first_divergence(normalised(r["outputs"][0], stop_ids),
                                      normalised(engines["vllm_a"][rid]["outputs"][0], stop_ids)) is not None
                     for rid, r in engines["hf"].items())
    ranks = Counter("0" if d["vllm_token_rank"] == 0 else "1" if d["vllm_token_rank"] == 1 else "2+" for d in divs)
    return {
        "diverged": len(divs),
        "diverged_recomputed": recomputed,
        "of": len(engines["hf"]),
        "index_quantiles": quantiles(d["index"] for d in divs),
        "margin_quantiles": quantiles(d["margin"] for d in divs),
        "vllm_token_rank_under_hf": dict(sorted(ranks.items())),  # 1 = HF's second choice: a near-tie flip
        "not_hf_second_choice": ranks["2+"] + ranks["0"],
    }


def decode_mismatches(rows: dict[str, dict], tokens: TokenCounter, stop_ids: set[int]) -> int:
    """Rows whose text is not the decoding of their own tokens (the audit's re-decode check)."""
    return sum(tokens.decode(body(r["outputs"][0]["token_ids"], stop_ids)) != r["outputs"][0]["text"] for r in rows.values())


def run(paths: Paths, tokens: TokenCounter) -> dict:
    files = EngineFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    graph = load_cwe_graph(paths.cwe_xml)
    requests, engines, sources = load_engines(files)
    if any(r["bank_sha256"] != bank_sha for r in requests):
        raise SystemExit("engine-check requests were built from a different bank_nontest.jsonl")
    items = [by_id[r["item_id"]] for r in requests]
    stop_ids = {i for i in (tokens.token_id(t) for t in pinned.EVAL_STOP_TOKENS) if i is not None}

    verdicts = {e: {i["item_id"]: verifiers.verify_item(i["type"], rows[f"eval:{i['item_id']}"]["outputs"][0]["text"], i["gold"], graph)
                    for i in items} for e, rows in engines.items()}
    units = {e: unit_scores(v, items) for e, v in verdicts.items()}
    comparisons = {f"{x}_vs_{y}": {t: compare_type(units[x][t], units[y][t]) for t in pinned.BANK_TYPES} for x, y in PAIRS}
    deciding = comparisons["vllm_a_vs_hf"]
    material_types = [t for t, c in deciding.items() if c["material"]]
    report = {
        "outcome": "material" if material_types else "agree",
        "material_types": material_types,
        "rule": {"gap": str(pinned.ENGINE_GAP), "interval": str(pinned.ENGINE_CI), "resamples": pinned.ENGINE_BOOTSTRAP,
                 "seed": pinned.ENGINE_SEED, "decided_by": "vllm_a_vs_hf, lenient parsers"},
        "parser_version": verifiers.PARSER_VERSION,
        "n_requests": len(requests),
        "n_cves": len({r["cve_id"] for r in requests}),
        "comparisons": comparisons,
        "engines": {e: engine_stats(engines[e], verdicts[e], items, stop_ids) for e in ENGINES},
        "identical_text": {f"{x}_vs_{y}": identical_text(engines[x], engines[y], items) for x, y in PAIRS},
        "divergence_hf_from_vllm_a": divergence_section(engines, stop_ids),
        "decode_mismatches": {e: decode_mismatches(engines[e], tokens, stop_ids) for e in ENGINES},
        "run_metas": run_metas(files),
        "sources": {**sources, "bank_nontest_sha256": bank_sha, "tokenizer_sha256": tokens.sha256},
    }
    write_json(files.report_json, report)
    files.report_md.write_text(render_markdown(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

LABELS = {"vllm_a": "vLLM A", "vllm_b": "vLLM B", "hf": "HF"}


def _meta_line(name: str, meta: dict | None) -> str:
    if not meta:
        return f"- {name}: no run_meta"
    det = meta.get("determinism_recheck", {})
    extra = (f"attention {meta.get('attn_implementation')}, argmax checked {meta.get('argmax_checked')}" if meta.get("engine") == "hf"
             else f"engine settings {meta.get('engine_settings')}, prompt token ids checked {meta.get('prompt_token_ids_checked_this_session')}")
    return (f"- {name}: commit `{meta.get('git', {}).get('commit', '?')[:12]}`, versions {meta.get('versions')}; {extra}; "
            f"determinism recheck {len(det.get('mismatched', []))} of {det.get('n', '?')} mismatched")


def render_markdown(r: dict) -> str:
    pts = lambda x: f"{100 * x:+.1f}"  # noqa: E731
    rows = []
    for name, comp in r["comparisons"].items():
        x, y = name.split("_vs_")
        for t, c in comp.items():
            lo, hi = c["interval"]
            rows.append(f"| {LABELS[x]} vs {LABELS[y]} | {t} | {c['n']} | {c['mean_x']:.3f} | {c['mean_y']:.3f} | {pts(c['gap'])} "
                        f"({pts(lo)}, {pts(hi)}) | {c['y_better']} / {c['x_better']} | {r['identical_text'][name][t]:.1%} "
                        f"| {'**material**' if c['material'] and name == 'vllm_a_vs_hf' else 'material (reference only)' if c['material'] else '—'} |")
    eng = []
    for e, stats in r["engines"].items():
        for t, s in stats.items():
            eng.append(f"| {LABELS[e]} | {t} | {s['parse_rate']:.1%} | {s['strict_rate']:.1%} | {s['cut_off_rate']:.1%} | {s['mean_output_tokens']:.0f} |")
    d = r["divergence_hf_from_vllm_a"]
    rule = r["rule"]
    return "\n".join([
        "# Step 7: engine-agreement check", "",
        f"**Outcome: {r['outcome']}.**" + (f" Material on: {', '.join(r['material_types'])}. Stop before step 8 and find the cause."
                                            if r["material_types"] else " HF and vLLM agree under the pre-registered rule; proceed."), "",
        f"{r['n_requests']:,} prompts ({r['n_cves']} non-test CVEs from the late window, all six items each), the raw backbone, greedy, "
        f"512 tokens. Rule: material if on any type |HF − vLLM A| ≥ {100 * float(Fraction(rule['gap'])):.0f} point and the "
        f"{float(Fraction(rule['interval'])):.0%} paired bootstrap interval over CVEs ({rule['resamples']:,} resamples) excludes 0. "
        f"vLLM A vs vLLM B (the same prompts in a shuffled order) is vLLM's own batch noise, reported for reference. "
        f"Parser version {r['parser_version']}.", "",
        "| Comparison | Type | CVEs | Mean (first) | Mean (second) | Gap, points (interval) | Second better / first better | Identical text | Decision |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows, "",
        "## Per engine", "",
        "| Engine | Type | Parsed (lenient) | Strict format | Cut off | Mean output tokens |", "|---|---|---|---|---|---|",
        *eng, "",
        "## Where HF first leaves vLLM A", "",
        f"- {d['diverged']:,} of {d['of']:,} replies diverge (recomputed from tokens: {d['diverged_recomputed']:,}).",
        f"- First-divergence token index: {d['index_quantiles']}.",
        f"- HF's logit margin between its token and vLLM's there: {d['margin_quantiles']}.",
        f"- Rank of vLLM's token under HF's logits (1 = HF's second choice, a near-tie flip): {d['vllm_token_rank_under_hf']}; "
        f"not HF's second choice: {d['not_hf_second_choice']}.",
        f"- Text that is not the decoding of its own tokens: {r['decode_mismatches']}.", "",
        "## Runs", "",
        *(_meta_line(k, v) for k, v in r["run_metas"].items()), "",
    ])
