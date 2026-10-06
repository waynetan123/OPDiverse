"""The step-10 report, from the files: every evaluation's per-type scores and parse, strict and cut-off rates; each
run's training record (tokens, FLOPs, GPU-hours, monitor events); the frozen scale and the selection; the flags carried
beside the distill-self and DPO cells; and the raw backbone on full dev as a reference row."""

from __future__ import annotations

import json
from pathlib import Path

from converters.files import ConverterFiles
from etl import pinned
from etl.build import write_json
from etl.paths import Paths
from train.config import run_name

from .select import Context, evaluate, frozen_scale, statistic


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def training_record(meta: dict | None) -> dict | None:
    if meta is None:
        return None
    keys = ("steps_done", "tokens", "flops", "gpu_hours", "config_sha256", "monitor_events", "versions", "git",
            "adapters", "pilot")
    return {k: meta.get(k) for k in keys}


def flags(paths: Paths) -> dict:
    """distill-self gold-only and DPO rule-built shares per type, for the sweep's seed and configuration."""
    rep = _json(ConverterFiles.of(paths).report_json)
    if rep is None:
        return {}
    f = rep["files"][str(pinned.SWEEP_SEED)][pinned.SWEEP_CONFIG]
    return {"distill_self_gold_only": f["distill_self"]["gold_only_share"], "dpo_rule_built": f["dpo"]["rule_share"]}


def run(paths: Paths) -> dict:
    ctx = Context.of(paths)
    scale = frozen_scale(ctx.files) if ctx.files.scale.exists() and pinned.SELECTION_SCALE is not None else None
    runs = {}
    for arm in pinned.SWEEP_ARMS:
        for lr in pinned.LR_GRID:
            name = run_name(arm, lr)
            evals = {}
            for s in ctx.steps:
                for which in ("checkpoint", "dev"):
                    if ctx.available(arm, lr, s, which):
                        e = ctx.evaluation(arm, lr, s, which)
                        evals[f"step{s}/{which}"] = {"types": e.stats,
                                                     "stat": None if scale is None else statistic(e.units, scale)}
            meta = _json(ctx.files.run_dir(arm, lr) / "run_meta.json")
            if evals or meta:
                runs[name] = {"arm": arm, "lr": lr, "training": training_record(meta), "evaluations": evals}
    backbone = None
    if ctx.files.backbone_generations.exists():
        e = evaluate(ctx.files.requests("dev"), ctx.files.backbone_generations, ctx.by_id, ctx.graph)
        backbone = {"types": e.stats, "stat": None if scale is None else statistic(e.units, scale)}
    out = {
        "seed": pinned.SWEEP_SEED, "config": pinned.SWEEP_CONFIG, "lr_grid": list(pinned.LR_GRID), "steps": ctx.steps,
        "scale": scale, "selection": _json(ctx.files.selection), "runs": runs, "backbone_dev": backbone,
        "flags": flags(paths),
    }
    write_json(ctx.files.report_json, out)
    ctx.files.report_md.write_text(render_markdown(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _f(x, spec=".3f") -> str:
    return "—" if x is None else format(x, spec)


def _n(x) -> str:
    return "—" if x is None else f"{x:,}"


def render_markdown(r: dict) -> str:
    types = pinned.BANK_TYPES
    out = ["# Step 10: LR sweep", "",
           f"M1, seed {r['seed']}. Grid {', '.join(f'{lr:.0e}' for lr in r['lr_grid'])}; checkpoints at steps "
           f"{', '.join(map(str, r['steps']))}. The statistic is the mean over types of each type's score divided by its "
           "frozen per-CVE spread.", ""]
    if r["scale"]:
        out += ["## Type scale (frozen)", "", "| " + " | ".join(types) + " |", "|" + "---|" * len(types),
                "| " + " | ".join(f"{r['scale'][t]:.4f}" for t in types) + " |", ""]
    sel = r["selection"] or {"arms": {}, "runs": {}, "todo": []}
    if sel["arms"]:
        out += ["## Winners", "", "| Arm | LR | Step | Full-dev statistic per LR | Edge | Winner minus each other LR (interval) |",
                "|---|---|---|---|---|---|"]
        for arm, a in sel["arms"].items():
            stats = ", ".join(f"{k}: {v:.3f}" for k, v in a["dev_stats"].items())
            pairs = "; ".join(f"{k}: {-p['difference']:+.3f} ({-p['interval'][1]:+.3f}, {-p['interval'][0]:+.3f})"
                              for k, p in a["paired_vs_winner"].items())
            out.append(f"| {arm} | {a['lr']:.0e} | {a['step']} | {stats} | {'**yes**' if a['edge'] else 'no'} | {pairs} |")
        out.append("")
    out += ["## Evaluations", "", "| Run | Evaluation | Statistic | " + " | ".join(types) + " | Parsed (min) | Cut off (max) |",
            "|" + "---|" * (len(types) + 5)]
    for name, run in r["runs"].items():
        chosen = sel["runs"].get(name, {}).get("chosen_step")
        for key, e in run["evaluations"].items():
            mark = " ✓" if chosen is not None and key.startswith(f"step{chosen}/") else ""
            ts = e["types"]
            out.append(f"| {name} | {key}{mark} | {_f(e['stat'])} | " + " | ".join(_f(ts[t]['score']) for t in types)
                       + f" | {min(ts[t]['parse_rate'] for t in types):.1%} | {max(ts[t]['cut_off_rate'] for t in types):.1%} |")
    if r["backbone_dev"]:
        ts = r["backbone_dev"]["types"]
        out.append(f"| raw backbone (reference) | dev | {_f(r['backbone_dev']['stat'])} | "
                   + " | ".join(_f(ts[t]['score']) for t in types)
                   + f" | {min(ts[t]['parse_rate'] for t in types):.1%} | {max(ts[t]['cut_off_rate'] for t in types):.1%} |")
    out += ["", "## Training", "", "| Run | Steps | Forward tokens | Generated tokens | FLOPs | GPU-hours | Monitor events |",
            "|---|---|---|---|---|---|---|"]
    for name, run in r["runs"].items():
        t = run["training"]
        if t:
            tok = t.get("tokens") or {}
            out.append(f"| {name}{' (pilot)' if t.get('pilot') else ''} | {t.get('steps_done')} | {_n(tok.get('forward'))} | "
                       f"{_n(tok.get('generated'))} | {_f(t.get('flops'), '.2e')} | {_f(t.get('gpu_hours'), '.1f')} | "
                       f"{len(t.get('monitor_events') or [])} |")
    if r["flags"]:
        out += ["", "## Flags beside the cells", "", "| Type | distill-self gold only | DPO rule-built |", "|---|---|---|"]
        for t in types:
            out.append(f"| {t} | {r['flags']['distill_self_gold_only'][t]} | {r['flags']['dpo_rule_built'][t]} |")
    if sel["todo"]:
        out += ["", "## Still to do", "", *(f"- {x}" for x in sel["todo"])]
    return "\n".join(out) + "\n"
