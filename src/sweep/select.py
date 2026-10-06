"""Scoring evaluations, the frozen type scale, and selection (owner, step 10).

Per evaluation and type there is one score per CVE (engine_check.compare.unit_scores): the item's metric, or the
paired score for find_error. The selection statistic of an evaluation is the column-standardised mean across types:

    stat = mean over types t of ( mean over CVEs c of u[t][c] ) / scale[t]

scale[t] is the per-CVE spread: in each scale evaluation, the ddof-1 SD over CVEs of u[t][.]; pooled as the root
mean of those variances over every scale evaluation (pinned.SCALE_ARMS x LR_GRID x the checkpoints, on the checkpoint
subsample). It is written to scale.json and pinned as pinned.SELECTION_SCALE in a commit before any choice is made
(data/ is not in git; the pin is the verifiable freeze), and every later choice uses it.

    checkpoint of a run  the highest stat on the checkpoint subsample; ties go to the earlier step
    LR of an arm         the highest stat on full dev, at each run's chosen checkpoint; ties go to the lower LR

Every dev CVE carries all five types, so stat is also the mean over CVEs of a per-CVE standardised score; paired
differences between two LRs, and their cluster-bootstrap interval over CVEs, follow from that (reported, not decided
on). Choices, once written to selection.json, are frozen.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from engine_check.compare import bootstrap_interval, unit_scores
from etl import pinned, verifiers
from etl.build import read_jsonl, write_json
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from frozen_model.prepare import load_bank
from train.config import checkpoint_steps, run_name, train_steps

from .files import SweepFiles

Units = dict[str, dict[str, Fraction]]  # type -> cve -> score


# ---------------------------------------------------------------------------
# One evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evaluation:
    units: Units
    stats: dict[str, dict]      # type -> parse, strict, cut-off rates, mean output tokens
    generations_sha256: str


def load_generations(requests_path: Path, generations_path: Path) -> tuple[list[dict], dict[str, dict]]:
    """(requests, request_id -> row). Exits unless the rows cover exactly these requests, from this file and prompts."""
    requests = read_jsonl(requests_path)
    sha = file_sha256(requests_path)
    rows = {r["request_id"]: r for r in read_jsonl(generations_path)}
    if set(rows) != {r["request_id"] for r in requests}:
        raise SystemExit(f"{generations_path} does not cover exactly the requests in {requests_path.name}")
    for r in requests:
        g = rows[r["request_id"]]
        if g["requests_sha256"] != sha or g["prompt_sha256"] != pinned.sha256_text(r["prompt"]) or len(g["outputs"]) != 1:
            raise SystemExit(f"{generations_path}: {r['request_id']} was not generated from {requests_path.name}")
    return requests, rows


def evaluate(requests_path: Path, generations_path: Path, by_id: dict[str, dict], graph,
             types: tuple[str, ...] = pinned.BANK_TYPES) -> Evaluation:
    requests, rows = load_generations(requests_path, generations_path)
    items = [by_id[r["item_id"]] for r in requests]
    outs = {i["item_id"]: rows[f"eval:{i['item_id']}"]["outputs"][0] for i in items}
    vs = {i["item_id"]: verifiers.verify_item(i["type"], outs[i["item_id"]]["text"], i["gold"], graph) for i in items}
    units = {t: u for t, u in unit_scores(vs, items).items() if t in types}
    stats = {}
    for t in types:
        chosen = [i["item_id"] for i in items if i["type"] == t]
        stats[t] = {
            "score": float(sum(units[t].values(), Fraction(0)) / len(units[t])),
            "parse_rate": sum(vs[i].parse_ok for i in chosen) / len(chosen),
            "strict_rate": sum(vs[i].strict_ok for i in chosen) / len(chosen),
            "cut_off_rate": sum(outs[i]["finish_reason"] == "length" for i in chosen) / len(chosen),
            "mean_output_tokens": sum(len(outs[i]["token_ids"]) for i in chosen) / len(chosen),
        }
    return Evaluation(units, stats, file_sha256(generations_path))


# ---------------------------------------------------------------------------
# Scale and statistic
# ---------------------------------------------------------------------------


def variance(values: list[float]) -> float:
    mu = math.fsum(values) / len(values)
    return math.fsum((v - mu) ** 2 for v in values) / (len(values) - 1)


def type_scale(evals: list[Units], types: tuple[str, ...] = pinned.BANK_TYPES) -> dict[str, float]:
    """The per-CVE spread per type, pooled over evaluations as the root mean of the per-evaluation variances."""
    out = {}
    for t in types:
        var = [variance([float(v) for v in e[t].values()]) for e in evals]
        out[t] = math.sqrt(math.fsum(var) / len(var))
        if out[t] == 0:
            raise SystemExit(f"type {t} has no spread in any scale evaluation; it cannot be standardised")
    return out


def per_cve(units: Units, scale: dict[str, float], types: tuple[str, ...]) -> dict[str, float]:
    """Each CVE's standardised score, averaged over types; their mean is the statistic."""
    cves = sorted(units[types[0]])
    if any(sorted(units[t]) != cves for t in types):
        raise SystemExit("the types were scored on different CVEs")
    return {c: math.fsum(float(units[t][c]) / scale[t] for t in types) / len(types) for c in cves}


def statistic(units: Units, scale: dict[str, float], types: tuple[str, ...] = pinned.BANK_TYPES) -> float:
    values = per_cve(units, scale, types)
    return math.fsum(values.values()) / len(values)


def argmax(candidates: list[tuple[object, float]]):
    """The candidate with the highest value; on an exact tie, the first (the earlier step, or the lower LR)."""
    best = candidates[0]
    for c in candidates[1:]:
        if c[1] > best[1]:
            best = c
    return best[0]


def paired(a: Units, b: Units, scale: dict[str, float], types: tuple[str, ...] = pinned.BANK_TYPES) -> dict:
    """b minus a: the standardised difference and its cluster-bootstrap interval over CVEs."""
    pa, pb = per_cve(a, scale, types), per_cve(b, scale, types)
    diffs = [pb[c] - pa[c] for c in sorted(pa)]
    lo, hi = bootstrap_interval(diffs, pinned.SWEEP_CI, pinned.SWEEP_BOOTSTRAP, pinned.SWEEP_BOOTSTRAP_SEED)
    return {"difference": math.fsum(diffs) / len(diffs), "interval": [lo, hi], "level": str(pinned.SWEEP_CI),
            "resamples": pinned.SWEEP_BOOTSTRAP, "seed": pinned.SWEEP_BOOTSTRAP_SEED}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    files: SweepFiles
    by_id: dict[str, dict]
    graph: object
    steps: list[int]

    @classmethod
    def of(cls, paths: Paths) -> Context:
        bank, _ = load_bank(paths)
        return cls(SweepFiles.of(paths), {r["item_id"]: r for r in bank}, load_cwe_graph(paths.cwe_xml),
                   checkpoint_steps(train_steps()))

    def available(self, arm: str, lr: float, step: int, which: str) -> bool:
        return self.files.generations(arm, lr, step, which).exists()

    def evaluation(self, arm: str, lr: float, step: int, which: str) -> Evaluation:
        return evaluate(self.files.requests(which), self.files.generations(arm, lr, step, which), self.by_id, self.graph)


def scale_evaluations(ctx: Context) -> list[tuple[str, float, int]]:
    return [(a, lr, s) for a in pinned.SCALE_ARMS for lr in pinned.LR_GRID for s in ctx.steps]


def freeze_scale(paths: Paths) -> dict:
    """Compute the type scale from every scale evaluation and write scale.json; refuse to change a frozen one."""
    ctx = Context.of(paths)
    wanted = scale_evaluations(ctx)
    missing = [run_name(a, lr) + f"/step{s}" for a, lr, s in wanted if not ctx.available(a, lr, s, "checkpoint")]
    if missing:
        raise SystemExit(f"{len(missing)} scale evaluations are missing, e.g. {missing[:3]}")
    evals = {f"{run_name(a, lr)}/step{s}": ctx.evaluation(a, lr, s, "checkpoint") for a, lr, s in wanted}
    out = {
        "scale": type_scale([e.units for e in evals.values()]),
        "rule": "per type: root mean over evaluations of the ddof-1 variance over CVEs of the per-CVE score",
        "evaluations": {k: e.generations_sha256 for k, e in evals.items()},
        "checkpoint_requests_sha256": file_sha256(ctx.files.requests("checkpoint")),
        "steps": ctx.steps,
    }
    if ctx.files.scale.exists():
        old = json.loads(ctx.files.scale.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(out)):
            raise SystemExit("scale.json is frozen and this computation differs; a change needs a decision-record entry")
    write_json(ctx.files.scale, out)
    return out


def frozen_scale(files: SweepFiles) -> dict[str, float]:
    if not files.scale.exists():
        raise SystemExit("scale.json is missing: run `python -m sweep scale` once every scale evaluation exists")
    scale = json.loads(files.scale.read_text(encoding="utf-8"))["scale"]
    if pinned.SELECTION_SCALE is None:
        raise SystemExit("pin scale.json's values as pinned.SELECTION_SCALE and commit them before any selection: "
                         f"{scale}")
    if pinned.SELECTION_SCALE != scale:
        raise SystemExit("scale.json differs from pinned.SELECTION_SCALE")
    return scale


def select(paths: Paths) -> dict:
    """Choose every checkpoint and LR whose evaluations exist; list what is still needed. Earlier choices are frozen."""
    ctx = Context.of(paths)
    scale = frozen_scale(ctx.files)
    old = json.loads(ctx.files.selection.read_text(encoding="utf-8")) if ctx.files.selection.exists() else {"runs": {}, "arms": {}}
    runs, arms, todo = {}, {}, []
    for arm in pinned.SWEEP_ARMS:
        chosen: dict[float, int] = {}
        for lr in pinned.LR_GRID:
            name = run_name(arm, lr)
            have = [s for s in ctx.steps if ctx.available(arm, lr, s, "checkpoint")]
            if len(have) < len(ctx.steps):
                todo += [f"{name}/step{s}: checkpoint evaluation" for s in ctx.steps if s not in have]
                continue
            stats = {s: statistic(ctx.evaluation(arm, lr, s, "checkpoint").units, scale) for s in ctx.steps}
            step = argmax([(s, stats[s]) for s in ctx.steps])
            chosen[lr] = step
            runs[name] = {"arm": arm, "lr": lr, "checkpoint_stats": {str(s): v for s, v in stats.items()},
                          "chosen_step": step}
        if len(chosen) < len(pinned.LR_GRID):
            continue
        missing = [lr for lr in pinned.LR_GRID if not ctx.available(arm, lr, chosen[lr], "dev")]
        if missing:
            todo += [f"{run_name(arm, lr)}/step{chosen[lr]}: full-dev evaluation" for lr in missing]
            continue
        dev = {lr: ctx.evaluation(arm, lr, chosen[lr], "dev") for lr in pinned.LR_GRID}
        stats = {lr: statistic(dev[lr].units, scale) for lr in pinned.LR_GRID}
        winner = argmax([(lr, stats[lr]) for lr in pinned.LR_GRID])
        for lr in pinned.LR_GRID:
            runs[run_name(arm, lr)]["dev_stat"] = stats[lr]
        arms[arm] = {
            "lr": winner,
            "step": chosen[winner],
            "dev_stats": {f"{lr:.0e}": stats[lr] for lr in pinned.LR_GRID},
            "edge": winner in (min(pinned.LR_GRID), max(pinned.LR_GRID)),
            "paired_vs_winner": {f"{lr:.0e}": paired(dev[winner].units, dev[lr].units, scale)
                                 for lr in pinned.LR_GRID if lr != winner},
        }
    for name, r in old["runs"].items():
        if name in runs and r["chosen_step"] != runs[name]["chosen_step"]:
            raise SystemExit(f"{name}: the chosen checkpoint would change from step {r['chosen_step']}; choices are frozen")
    for arm, a in old["arms"].items():
        if arm in arms and (a["lr"], a["step"]) != (arms[arm]["lr"], arms[arm]["step"]):
            raise SystemExit(f"{arm}: the chosen LR would change; choices are frozen")
    runs = {**old["runs"], **runs}   # a recorded choice stays recorded even if its evaluation files are removed
    arms = {**old["arms"], **arms}
    out = {"scale": scale, "steps": ctx.steps, "runs": runs, "arms": arms, "todo": todo}
    write_json(ctx.files.selection, out)
    return out
