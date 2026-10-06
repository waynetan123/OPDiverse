"""The step-9 report, read from the written training files: rows and duplicates, tokens seen per epoch (the plan's
per-arm token report; GRPO's completions are generated during training, so only its prompts are counted), the
longest sequence each trainer must fit, the per-type flags carried beside the distill-self and DPO cells, the
GRPO caps, and the M1-volume masks."""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from statistics import median_low

from etl import pinned
from etl.build import write_json
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.tokens import TokenCounter
from generators.progress import track

from . import masks
from .arms import base
from .build import Inputs, load_inputs, specs
from .files import ConverterFiles, read_rows

END = pinned.COMPLETION_END
CHUNK = 512


def _count(tokens: TokenCounter, texts: list[str], label: str) -> list[int]:
    out: list[int] = []
    for start in track(range(0, len(texts), CHUNK), label):
        out += tokens.count(texts[start:start + CHUNK])
    return out


def token_tables(inputs: Inputs, tokens: TokenCounter) -> dict[str, dict[str, int]]:
    """Token counts of every distinct text a training file can carry, keyed by item_id (base: cve_id). Each
    completion is counted with its end token."""
    bank = inputs.bank
    ids = [r["item_id"] for r in bank]
    texts = {
        "prompt": None,
        "sft": [r["target"] + END for r in bank],
        "distill_self": [inputs.distill[i]["target"] + END for i in ids],
        "rejected": [inputs.dpo[i]["rejected"] + END for i in ids],
    }
    out = {"prompt": {r["item_id"]: r["prompt_tokens"] for r in bank}}
    for name, values in texts.items():
        if values is not None:
            out[name] = dict(zip(ids, _count(tokens, values, f"Tokens ({name})")))
    cves = sorted({r["cve_id"] for r in bank})
    docs = [base(inputs.facts[c], inputs.graph)["text"] for c in cves]
    out["base"] = dict(zip(cves, _count(tokens, docs, "Tokens (base)")))
    return out


def _dist(values: list[int]) -> dict:
    return {"sum": sum(values), "median": median_low(values), "max": max(values)} if values else None


def file_stats(arm: str, rows: list[dict], tok: dict[str, dict[str, int]]) -> dict:
    if arm == "base":
        doc = [tok["base"][r["cve_id"]] for r in rows]
        return {"rows": len(rows), "doc_tokens": _dist(doc), "max_sequence_tokens": max(doc)}
    ids = [r["item_id"] for r in rows]
    prompt = [tok["prompt"][i] for i in ids]
    out = {
        "rows": len(rows),
        "distinct_items": len(set(ids)),
        "duplicated_items": sum(r["copy"] > 0 for r in rows),
        "rows_by_type": {t: n for t, n in sorted(Counter(r["type"] for r in rows).items(),
                                                   key=lambda kv: pinned.BANK_TYPES.index(kv[0]))},
        "prompt_tokens": _dist(prompt),
    }
    if arm in ("sft", "distill_self"):
        comp = [tok[arm][i] for i in ids]
        out["completion_tokens"] = _dist(comp)
        out["max_sequence_tokens"] = max(p + c for p, c in zip(prompt, comp))
    elif arm == "dpo":
        chosen, rejected = [tok["sft"][i] for i in ids], [tok["rejected"][i] for i in ids]
        out["chosen_tokens"], out["rejected_tokens"] = _dist(chosen), _dist(rejected)
        out["max_sequence_tokens"] = max(p + max(c, r) for p, c, r in zip(prompt, chosen, rejected))
    else:  # grpo: the rollout is generated, so the longest sequence is the prompt plus its type's cap
        out["max_sequence_tokens"] = max(tok["prompt"][r["item_id"]] + r["max_completion_tokens"] for r in rows)
    if arm in ("distill_self", "dpo"):
        key, flag = ("source", "gold_only") if arm == "distill_self" else ("rejected_source", "rule")
        out[f"{flag}_share"] = {t: str(Fraction(sum(r[key] == flag for r in rows if r["type"] == t), n))
                                for t, n in out["rows_by_type"].items()}
    return out


def run(paths: Paths, tokens: TokenCounter) -> dict:
    files = ConverterFiles.of(paths)
    meta = json.loads(files.meta.read_text(encoding="utf-8"))
    inputs = load_inputs(paths)
    tok = token_tables(inputs, tokens)

    stats: dict[str, dict[str, dict]] = {}
    m1v: dict[str, dict[str, set[str]]] = {c: {} for c in masks.M1V}
    for seed, config, arm in track(specs(), "Report"):
        rows = read_rows(files.training(seed, config, arm))
        stats.setdefault(str(seed), {}).setdefault(config, {})[arm] = file_stats(arm, rows, tok)
        if config in m1v and arm == "sft":
            train = {r["item_id"] for r in inputs.train_items(seed)}
            m1v[config][str(seed)] = train - {r["item_id"] for r in rows}

    masks_out = {}
    for config, by_seed in m1v.items():
        seeds = sorted(by_seed)
        masks_out[config] = {
            "masked_by_type": {s: dict(Counter(i.split(":")[1] for i in by_seed[s])) for s in seeds},
            "overlap_jaccard": {f"{a}-{b}": round(len(by_seed[a] & by_seed[b]) / len(by_seed[a] | by_seed[b]), 4)
                                for k, a in enumerate(seeds) for b in seeds[k + 1:]},
        }
    longest = {arm: max(f[arm]["max_sequence_tokens"] for seed in stats.values() for f in seed.values() if arm in f)
               for arm in pinned.CONVERTER_ARMS}
    out = {
        "sources": {**meta["sources"], "converters_meta_sha256": file_sha256(files.meta),
                    "tokenizer_sha256": tokens.sha256},
        "seeds": list(pinned.PARTITION_SEEDS),
        "configs": {c: {"dropped_type": masks.dropped_type(c), "m1v_comparator": masks.comparator(c),
                        "dev_types": masks.dev_types(c)} for c in masks.CONFIGS},
        "grpo_types": inputs.grpo_types,
        "files": stats,
        "m1v_masks": masks_out,
        "max_sequence_tokens": longest,
    }
    write_json(files.report_json, out)
    files.report_md.write_text(render_markdown(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _span(values) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"


def _pct(frac: str) -> str:
    return f"{float(Fraction(frac)):.1%}"


def render_markdown(r: dict) -> str:
    seeds = [str(s) for s in r["seeds"]]
    files = r["files"]
    out = [
        "# Step 9: converters", "",
        f"Training files for seeds {', '.join(seeds)}: {len(r['configs'])} configurations, "
        f"{sum(len(c) for s in files.values() for c in s.values())} files. Every file of a configuration holds the same "
        "rows; ranges are across seeds.", "",
        "## Rows", "",
    ]
    rows = []
    for config in r["configs"]:
        for arm in files[seeds[0]][config]:
            per = [files[s][config][arm] for s in seeds]
            rows.append((config, arm, _span([f["rows"] for f in per]),
                         _span([f.get("distinct_items", f["rows"]) for f in per]),
                         _span([f.get("duplicated_items", 0) for f in per])))
    out += [_table(("Config", "Arm", "Rows", "Distinct items", "Duplicate rows"), rows), ""]

    out += ["## Configurations", ""]
    out += [_table(("Config", "Dropped type", "M1-volume comparator", "Dev types at checkpoint selection"),
                   ((c, v["dropped_type"] or "—", v["m1v_comparator"] or "—", ", ".join(v["dev_types"]))
                    for c, v in r["configs"].items())), ""]

    out += ["## Tokens seen per epoch (M1)", "",
            "Sums over one pass of the file, completions counted with their end token. GRPO's completions are "
            "generated in training, so only its prompts are counted.", ""]
    rows = []
    for arm in files[seeds[0]]["m1"]:
        per = [files[s]["m1"][arm] for s in seeds]
        if arm == "base":
            rows.append((arm, "—", _span([f["doc_tokens"]["sum"] for f in per]), _span([f["max_sequence_tokens"] for f in per])))
            continue
        if arm == "grpo":
            completion = "generated"
        elif arm == "dpo":
            completion = _span([f["chosen_tokens"]["sum"] + f["rejected_tokens"]["sum"] for f in per])
        else:
            completion = _span([f["completion_tokens"]["sum"] for f in per])
        rows.append((arm, _span([f["prompt_tokens"]["sum"] for f in per]), completion,
                     _span([f["max_sequence_tokens"] for f in per])))
    out += [_table(("Arm", "Prompt tokens", "Completion tokens (DPO: chosen + rejected)", "Longest sequence"), rows), ""]
    out += ["**Longest sequence in any file**, which the trainer's maximum length must cover so nothing is truncated: "
            + ", ".join(f"{a} {n:,}" for a, n in r["max_sequence_tokens"].items()) + ".", ""]

    out += ["## Flags beside the cells (M1)", "",
            "distill-self rows whose target is the gold answer alone (no reasoning), and DPO rows whose rejected answer "
            "is rule-built (MCQ is rule-defined by design).", ""]
    rows = []
    for t in pinned.BANK_TYPES:
        gold = [Fraction(files[s]["m1"]["distill_self"]["gold_only_share"][t]) for s in seeds]
        rule = [Fraction(files[s]["m1"]["dpo"]["rule_share"][t]) for s in seeds]
        g = r["grpo_types"][t]
        rows.append((t, f"{float(min(gold)):.1%}–{float(max(gold)):.1%}", f"{float(min(rule)):.1%}–{float(max(rule)):.1%}",
                     g["cap"], g["band"].replace("_", " ")))
    out += [_table(("Type", "distill-self gold only", "DPO rule-built", "GRPO cap", "GRPO band"), rows), ""]

    out += ["## M1-volume masks", ""]
    for config, m in r["m1v_masks"].items():
        out += [f"**{config}**, masked items per type:", "",
                _table(("Seed", *pinned.BANK_TYPES), ((s, *(m["masked_by_type"][s].get(t, 0) for t in pinned.BANK_TYPES))
                                                      for s in seeds)), "",
                "Mask overlap between seeds (Jaccard): " + ", ".join(f"{k} {v:.3f}" for k, v in m["overlap_jaccard"].items())
                + ".", ""]
    return "\n".join(out)
