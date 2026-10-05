"""Invariants over a built bank. Returns every violation; the CLI exits 1 if there are any."""

from __future__ import annotations

import json
import re
from collections import Counter

from etl import pinned, verifiers
from etl.build import read_jsonl
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.split import POOLS
from probe.prompts import render_qwen_chat

from . import items, mcq
from .build import load_inputs, templates_sha256
from .files import BankFiles
from ..progress import track

_OPTION_LINE = re.compile(r"[A-D]\. CWE-\d+: ")


def code_lines(fact: dict) -> set[int]:
    """1-indexed numbers of the vulnerable function's code lines (non-empty after comment masking,
    when masking succeeded): the only lines gold, or a DPO near miss, may name."""
    text = pinned.mask_comments(fact["vuln_func"])[0] if fact["comment_mask_ok"] else fact["vuln_func"]
    return {i for i, line in enumerate(pinned.split_lines(text), 1) if pinned.line_key(line)}


def line_numbering(fact: dict) -> list[str]:
    """The whole-table guard that replaces a manual read of the 50-item sheet: gold lines are
    in range and are code lines, and they are reproducible from the two stored functions."""
    cve, lines = fact["cve_id"], fact["patch_lines"]
    out = []
    code = code_lines(fact)
    if not lines or any(not 1 <= n <= fact["n_lines"] for n in lines):
        out.append(f"{cve}: gold lines {lines} outside [1, {fact['n_lines']}]")
    if not set(lines) <= code:
        out.append(f"{cve}: gold lines {sorted(set(lines) - code)} are not code lines")
    if list(pinned.patch_line_set(fact["vuln_func"], fact["patched_func"]).lines) != lines:
        out.append(f"{cve}: patch_line_set no longer reproduces the stored gold lines")
    return out


def _cve_id_in(text: str, cve_id: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(cve_id)}(?!\d)", text, re.IGNORECASE) is not None


def check(paths: Paths, dry: bool = False) -> list[str]:
    files = BankFiles.of(paths, dry=dry)
    facts, split, graph = load_inputs(paths)
    by_cve = {f["cve_id"]: f for f in facts}
    meta = json.loads(files.meta.read_text(encoding="utf-8"))
    decisions = {d["cve_id"]: d for d in read_jsonl(files.mcq_decisions)}
    bad: list[str] = []

    src = meta["sources"]
    current = {"facts_sha256": file_sha256(paths.facts), "split_sha256": file_sha256(paths.split),
               "templates_sha256": templates_sha256()}
    if not dry:
        current["mcq_generations_sha256"] = file_sha256(files.mcq_generations)
        if files.mcq_retry_generations.exists():
            current["mcq_retry_generations_sha256"] = file_sha256(files.mcq_retry_generations)
    for k, v in current.items():
        if src.get(k) != v:
            bad.append(f"bank_meta.json {k} does not match the current file")

    for f in track(facts, "Check line numbering"):
        bad += line_numbering(f)

    seen: set[str] = set()
    per_cve: Counter = Counter()
    for pool in POOLS:
        for row in track(read_jsonl(files.bank(pool)), f"Check items ({pool})"):
            iid, cve, t = row["item_id"], row["cve_id"], row["type"]
            fact, s = by_cve.get(cve), split.get(cve)
            if fact is None:
                bad.append(f"{iid}: CVE not in facts.jsonl")
                continue
            if iid in seen:
                bad.append(f"{iid}: duplicate item_id")
            seen.add(iid)
            per_cve[cve] += 1
            if iid != items.item_id(cve, t, row["index"]):
                bad.append(f"{iid}: item_id does not match its fields")
            if (row["pool"], row["cluster_id"]) != (pool, s["cluster_id"]) or s["pool"] != pool:
                bad.append(f"{iid}: pool or cluster differs from split.jsonl")
            if row["prompt"] != render_qwen_chat(row["user"]):
                bad.append(f"{iid}: prompt is not the chat rendering of user")
            if row["template_version"] != pinned.BANK_TEMPLATE_VERSION:
                bad.append(f"{iid}: template version {row['template_version']}")

            v = verifiers.verify_item(t, row["target"], row["gold"], graph)
            if not (v.strict_ok and v.metric == 1 and v.dense == 1):
                bad.append(f"{iid}: target {row['target']!r} does not score 1 under its own strict verifier")
            if _cve_id_in(row["user"], cve):
                bad.append(f"{iid}: prompt contains its own CVE ID")
            outside = "\n".join(line for line in row["user"].split("\n") if not _OPTION_LINE.match(line))
            if re.search(rf"{re.escape(fact['cwe'])}(?!\d)", outside):
                bad.append(f"{iid}: prompt contains the gold CWE ID outside the MCQ options")

            if t == "mcq":
                bad += _check_mcq(row, fact, decisions.get(cve), graph)
            elif t == "find_error":
                side = "vuln" if row["index"] == 0 else "patched"
                if items.prompt_function(fact, side) not in row["user"]:
                    bad.append(f"{iid}: prompt does not contain the stored {side} function")
            elif t == "line_loc":
                if pinned.render_numbered(items.prompt_function(fact, "vuln")) not in row["user"]:
                    bad.append(f"{iid}: numbered function differs from render_numbered")
                if row["gold"] != {"lines": fact["patch_lines"], "n_lines": fact["n_lines"]}:
                    bad.append(f"{iid}: gold differs from the fact")

    longest = max((r["prompt_tokens"] for pool in POOLS for r in read_jsonl(files.bank(pool))), default=0)
    if longest + pinned.EVAL_MAX_TOKENS > pinned.EVAL_MAX_MODEL_LEN:
        bad.append(f"EVAL_MAX_MODEL_LEN {pinned.EVAL_MAX_MODEL_LEN} < longest prompt {longest} + {pinned.EVAL_MAX_TOKENS}")
    missing = sorted(set(by_cve) - set(per_cve))
    if missing:
        bad.append(f"{len(missing)} CVEs have no items, e.g. {missing[:3]}")
    wrong = sorted(c for c, n in per_cve.items() if n != pinned.ITEMS_PER_CVE)
    if wrong:
        bad.append(f"{len(wrong)} CVEs do not have {pinned.ITEMS_PER_CVE} items, e.g. {wrong[:3]}")
    return bad


def _check_mcq(row: dict, fact: dict, decision: dict | None, graph) -> list[str]:
    iid, gold = row["item_id"], row["gold"]
    out = []
    options = gold["options"]
    if [o["letter"] for o in options] != list(pinned.MCQ_LETTERS):
        out.append(f"{iid}: options are not lettered A-D")
    by_letter = {o["letter"]: o["cwe"] for o in options}
    if by_letter.get(gold["letter"]) != fact["cwe"] or gold["cwe"] != fact["cwe"]:
        out.append(f"{iid}: the gold letter does not point at the fact's CWE")
    distractors = [o["cwe"] for o in options if o["letter"] != gold["letter"]]
    admitted, rejected = mcq.admit(distractors, fact["cwe"], graph)
    if admitted != distractors:
        out.append(f"{iid}: inadmissible distractors {rejected}")
    if any(o["name"] != graph.weaknesses[o["cwe"][4:]].name for o in options):
        out.append(f"{iid}: an option name is not the MITRE name")
    for letter in pinned.MCQ_LETTERS:
        if letter != gold["letter"] and verifiers.verify_mcq(f"ANSWER: {letter}", gold["letter"]).metric != 0:
            out.append(f"{iid}: distractor letter {letter} scores")
    if decision is None or sorted(decision["distractors"]) != sorted(distractors) or decision["gold_letter"] != gold["letter"]:
        out.append(f"{iid}: options differ from mcq_decisions.jsonl")
    return out
