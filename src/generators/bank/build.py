"""Step 4: build the question bank once over each pool, then freeze it.

The bank is written to data/bank and frozen: a rebuild that would change any output byte is refused.
--dry-mcq builds into data/bank/dry with every MCQ from the prior-matched draw, which needs no
external request and is never frozen; it is also exactly what the bank becomes if the guard fires.
"""

from __future__ import annotations

import json
from pathlib import Path

from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from etl.cwe_graph import CweGraph, load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.split import POOLS
from etl.tokens import TokenCounter

from .. import external
from . import items, mcq
from .files import BankFiles
from ..progress import Bar, track

TOKEN_CHUNK = 500  # prompts per tokenizer call; only sets the progress granularity


def templates_sha256() -> str:
    body = json.dumps({"version": pinned.BANK_TEMPLATE_VERSION, "prompts": pinned.BANK_PROMPTS,
                       "option": pinned.MCQ_OPTION, "cwe_redacted": pinned.CWE_REDACTED,
                       "cve_redacted": pinned.CVE_REDACTED, "cwe_literal": pinned.CWE_LITERAL.pattern}, sort_keys=True)
    return pinned.sha256_text(body)


def load_inputs(paths: Paths) -> tuple[list[dict], dict[str, dict], CweGraph]:
    facts = read_jsonl(paths.facts)
    split = {r["cve_id"]: r for r in read_jsonl(paths.split)}
    if set(split) != {f["cve_id"] for f in facts}:
        raise SystemExit("split.jsonl and facts.jsonl cover different CVEs")
    return facts, split, load_cwe_graph(paths.cwe_xml)


def _by_custom_id(path: Path) -> dict[str, dict]:
    return {r["custom_id"]: r for r in read_jsonl(path)} if path.exists() else {}


def model_decisions(facts: list[dict], split: dict[str, dict], graph: CweGraph, files: BankFiles) -> list[dict]:
    if not files.mcq_generations.exists():
        raise SystemExit(f"{files.mcq_generations} is missing: run the MCQ batch, or build with --dry-mcq")
    first, retry = _by_custom_id(files.mcq_generations), _by_custom_id(files.mcq_retry_generations)
    counts = mcq.gold_counts(facts, {c: s["pool"] for c, s in split.items()})
    decisions, pending = [], []
    for f in track(facts, "MCQ options (model)"):
        iid = items.item_id(f["cve_id"], "mcq")
        row1 = first.get(external.custom_id(iid, 1))
        if row1 is None:
            raise SystemExit(f"{iid}: no attempt-1 generation")
        status, proposals = mcq.parse_generation(row1)
        rows = [row1]
        if mcq.needs_retry(status, mcq.admit(proposals, f["cwe"], graph)[0]):
            row2 = retry.get(external.custom_id(iid, 2))
            if row2 is None:
                pending.append(iid)
            rows.append(row2)
        decisions.append(mcq.decide(f, split[f["cve_id"]]["pool"], rows, counts, graph))
    if pending:
        raise SystemExit(f"{len(pending)} MCQ items need their one regeneration first "
                         f"(`python -m generators bank prepare-mcq --retry`), e.g. {pending[:3]}")
    return decisions


def build(paths: Paths, tokens: TokenCounter, dry: bool = False) -> dict:
    files = BankFiles.of(paths, dry=dry)
    facts, split, graph = load_inputs(paths)
    pool_of = {c: s["pool"] for c, s in split.items()}
    counts = mcq.gold_counts(facts, pool_of)
    draw = [mcq.draw_only(f, pool_of[f["cve_id"]], counts, graph) for f in track(facts, "MCQ options (draw)")]
    shortcut_draw = mcq.shortcut(draw, counts)
    if dry:
        decisions, guard = draw, {"mode": "dry", "shortcut_draw": str(shortcut_draw)}
    else:
        proposed = model_decisions(facts, split, graph, files)
        shortcut_model = mcq.shortcut(proposed, counts)
        fired = shortcut_model > pinned.MCQ_SHORTCUT_MAX
        decisions = draw if fired else proposed
        guard = {"mode": "model", "threshold": str(pinned.MCQ_SHORTCUT_MAX), "shortcut_model": str(shortcut_model),
                 "shortcut_draw": str(shortcut_draw), "fired": fired}
        if fired:  # keep what the model proposed on record, marked as discarded
            for d, p in zip(decisions, proposed):
                d["discarded_model_decision"] = {k: p[k] for k in ("attempts", "distractors", "sources")}

    rows = {pool: [] for pool in POOLS}
    for f, d in track(list(zip(facts, decisions)), "Question items"):
        s = split[f["cve_id"]]
        options, letter = mcq.layout(f["cve_id"], f["cwe"], d["distractors"], graph)
        d["gold_letter"] = letter
        rows[s["pool"]] += items.build_items(f, s["pool"], s["cluster_id"], options, letter)
    for pool, pool_rows in rows.items():
        with Bar(f"Token counts ({pool})", len(pool_rows)) as bar:
            for start in range(0, len(pool_rows), TOKEN_CHUNK):
                part = pool_rows[start:start + TOKEN_CHUNK]
                for row, n in zip(part, tokens.count([r["prompt"] for r in part])):
                    row["prompt_tokens"] = n
                bar.advance(len(part))

    sources = {"facts_sha256": file_sha256(paths.facts), "split_sha256": file_sha256(paths.split),
               "templates_sha256": templates_sha256(), "tokenizer_sha256": tokens.sha256}
    if not dry:
        sources["mcq_generations_sha256"] = file_sha256(files.mcq_generations)
        if files.mcq_retry_generations.exists():
            sources["mcq_retry_generations_sha256"] = file_sha256(files.mcq_retry_generations)
    meta = {
        "sources": sources,
        "template_version": pinned.BANK_TEMPLATE_VERSION,
        "mcq_guard": guard,
        "counts": {pool: {"cves": len(r) // pinned.ITEMS_PER_CVE, "items": len(r)} for pool, r in rows.items()},
    }
    outputs = {files.bank(pool): jsonl_bytes(r) for pool, r in rows.items()}
    outputs[files.mcq_decisions] = jsonl_bytes(decisions)
    outputs[files.meta] = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not dry:
        changed = [p.name for p, body in outputs.items() if p.exists() and p.read_bytes() != body]
        if changed:
            raise SystemExit(f"the bank is frozen and this build differs in {changed}; "
                             "a change to the bank needs a decision-record entry and a deliberate reset")
    files.dir.mkdir(parents=True, exist_ok=True)
    for path, body in outputs.items():
        path.write_bytes(body)
    return meta
