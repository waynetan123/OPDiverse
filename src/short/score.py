"""Scoring the day's test evaluations into the frozen analysis's input, and a per-run summary.

    m2_scores.jsonl          one row per (run, test item): {"arm", "seed", "item_id", "metric", "dropped_type"}
    score_summary.{json,md}  per run: each type's score, parse, strict and cut-off rates and mean output tokens (the
                             dropped type is M2; the other four are diagnostics), and the run's training record

Each item is scored by its verifier (etl.verifiers.verify_item) and written with its own metric as an exact fraction
("3/8"): the analysis pairs the two find-the-error items itself and decides zero-spread columns from exact sums.
Every run's test generations must come from test_requests.jsonl, through the chain run_meta.json (arm, configuration,
seed, LR, all TRAIN_STEPS steps) -> its last adapter -> merge_meta.json -> the evaluated weights. m2_scores.jsonl is
written only when every run is scored, since the primary test needs the whole matrix; until then the summary lists
what is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

from converters import masks
from engine_check.compare import unit_scores
from etl import pinned
from etl.build import jsonl_bytes, write_json
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from frozen_model.prepare import load_bank
from sweep.select import type_stats, verdicts
from train.config import train_steps

from .files import ShortFiles, lr, run_name


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_run(files: ShortFiles, arm: str, config: str, seed: int, steps: int) -> dict:
    """The run's run_meta.json, after checking its evaluation came from its own last checkpoint."""
    name = run_name(arm, config, seed)
    run_meta = files.run_dir(arm, config, seed) / "run_meta.json"
    merge_meta, eval_meta = files.merge_meta(arm, config, seed), files.eval_meta(arm, config, seed)
    for p in (run_meta, merge_meta, eval_meta):
        if not p.exists():
            raise SystemExit(f"{name}: {p.name} is missing")
    run, merge, ev = _json(run_meta), _json(merge_meta), _json(eval_meta)
    c = run["config"]
    if (c["arm"], c["config"], c["seed"], c["lr"]) != (arm, config, seed, lr(arm)) or run["pilot"]:
        raise SystemExit(f"{name}: run_meta.json describes {c['arm']} {c['config']} seed {c['seed']} LR {c['lr']}"
                         + (" (a pilot)" if run["pilot"] else ""))
    if run["steps_done"] != steps or c["steps"] != steps:
        raise SystemExit(f"{name}: trained {run['steps_done']} of {c['steps']} steps; TRAIN_STEPS is {steps}")
    adapter = Path(merge["adapter"])
    if adapter.name != f"checkpoint-{steps}" or merge["adapter_sha256"] != run["adapters"].get(str(steps)):
        raise SystemExit(f"{name}: the merged adapter {adapter} is not the run's last checkpoint")
    if ev.get("checkpoint_sha256") != merge["weights_sha256"]:
        raise SystemExit(f"{name}: the evaluated weights are not the ones merged from the run's last checkpoint")
    if ev["requests_sha256"] != file_sha256(files.test_requests):
        raise SystemExit(f"{name}: evaluated on a different request file")
    return run


def score_run(files: ShortFiles, run: tuple[str, str, int], by_id: dict, graph, steps: int) -> tuple[list[dict], dict]:
    arm, config, seed = run
    meta = check_run(files, arm, config, seed, steps)
    gen = files.generations(*run)
    items, outs, vs = verdicts(files.test_requests, gen, by_id, graph)
    dropped = masks.dropped_type(config)
    rows = [{"arm": arm, "seed": seed, "item_id": i["item_id"], "metric": str(vs[i["item_id"]].metric),
             "dropped_type": dropped} for i in items]
    summary = {
        "arm": arm, "config": config, "seed": seed, "dropped_type": dropped,
        "types": type_stats(items, outs, vs, unit_scores(vs, items)),
        "generations_sha256": file_sha256(gen),
        **{k: meta.get(k) for k in ("monitor_events", "gpu_hours", "flops", "tokens", "config_sha256")},
    }
    return rows, summary


def score(paths: Paths, arms: tuple[str, ...] = pinned.SHORT_ARMS) -> dict:
    files = ShortFiles.of(paths)
    if not files.test_requests.exists():
        raise SystemExit("test_requests.jsonl is missing: run `python -m short prepare` first")
    steps = train_steps()
    bank, _ = load_bank(paths, "test")
    by_id, graph = {r["item_id"]: r for r in bank}, load_cwe_graph(paths.cwe_xml)
    rows, runs, missing = [], {}, []
    for run in [(a, c, s) for a in arms for c in masks.M2 for s in pinned.SHORT_SEEDS]:
        if not files.generations(*run).exists():
            missing.append(run_name(*run))
            continue
        r, summary = score_run(files, run, by_id, graph, steps)
        rows += r
        runs[run_name(*run)] = summary
    out = {"arms": list(arms), "seeds": list(pinned.SHORT_SEEDS), "steps": steps, "runs": runs, "missing": missing,
           "test_requests_sha256": file_sha256(files.test_requests), "scores_written": not missing}
    if not missing:
        files.scores.write_bytes(jsonl_bytes(rows))
        out["scores_sha256"] = file_sha256(files.scores)
    write_json(files.summary_json, out)
    files.summary_md.write_text(markdown(out), encoding="utf-8")
    return out


def markdown(out: dict) -> str:
    types = pinned.BANK_TYPES
    lines = ["# One-day run: test scores", "",
             f"{len(out['runs'])} of {len(out['runs']) + len(out['missing'])} runs scored, {out['steps']} steps each. "
             "Each cell is the mean per-CVE test score (find-the-error: paired); **bold** is the run's dropped "
             "type (M2), the rest are diagnostics. Parse is the dropped type's parse rate.", "",
             "| Run | " + " | ".join(types) + " | Parse | Cut off | Monitor events | GPU-h |",
             "|---|" + "---|" * (len(types) + 4)]
    for name, r in out["runs"].items():
        cells = [(f"**{r['types'][t]['score']:.3f}**" if t == r["dropped_type"] else f"{r['types'][t]['score']:.3f}")
                 for t in types]
        d = r["types"][r["dropped_type"]]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {d['parse_rate']:.2f} | {d['cut_off_rate']:.2f} | "
                     f"{len(r['monitor_events'] or [])} | {r['gpu_hours']} |")
    events = [(name, e) for name, r in out["runs"].items() for e in r["monitor_events"] or []]
    if events:
        lines += ["", "## Monitor events", "", "Candidates for `--flagged arm:type` (the owner decides).", ""]
        lines += [f"- {name}: {json.dumps(e, sort_keys=True)}" for name, e in events]
    if out["missing"]:
        lines += ["", "## Missing", "", "m2_scores.jsonl is not written until every run is scored.", ""]
        lines += [f"- {name}" for name in out["missing"]]
    return "\n".join(lines) + "\n"
