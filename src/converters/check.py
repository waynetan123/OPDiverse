"""Invariants over the written training files. Returns violations (empty = pass).

The checks re-derive what each file must contain from the frozen inputs and the plan, not from the builder's
selection code: full coverage, no dev or test item, the M2 and M1-volume masks and upsampling, identical rows
across arms, byte-identical prompts, and each arm's completions.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction

from etl import pinned, verifiers
from etl.build import read_jsonl
from etl.paths import Paths

from . import masks
from .build import Inputs, load_inputs
from .files import ConverterFiles, read_content, read_rows, sha256_bytes

END = pinned.COMPLETION_END
QUESTION_PHRASE = "End your reply"


def _strip_end(text: str, end: str = END) -> str | None:
    return text[: -len(end)] if text.endswith(end) and not text[: -len(end)].endswith(end) else None


def verified(inputs: Inputs) -> dict[str, set[str]]:
    """Item_ids whose cached step-5 and step-6 content is valid under the frozen verifiers, computed once:
    the distill-self target ends in the gold answer, and the DPO rejected answer is a strict near miss
    scoring the dense value step 6 recorded."""
    ok: dict[str, set[str]] = {"distill_self": set(), "dpo": set()}
    for item in inputs.bank:
        i, g = item["item_id"], item["gold"]
        target = inputs.distill[i]["target"]
        if target.rstrip().split("\n")[-1] == item["target"] and \
                verifiers.verify_item(item["type"], target, g, inputs.graph).metric == 1:
            ok["distill_self"].add(i)
        pair = inputs.dpo[i]
        v = verifiers.verify_item(item["type"], pair["rejected"], g, inputs.graph)
        if pair["chosen"] == item["target"] and v.strict_ok and v.dense < 1 and str(v.dense) == pair["rejected_dense"]:
            ok["dpo"].add(i)
    return ok


def check_question_file(inputs: Inputs, ok: dict[str, set[str]], arm: str, rows: list[dict], where: str) -> list[str]:
    """Per-row content of one question-arm file."""
    bad = []
    by_id = inputs.by_id
    for r in rows:
        i = r["item_id"]
        item = by_id.get(i)
        if item is None:
            bad.append(f"{where}: {i} is not a non-test bank item")
            continue
        if (r["cve_id"], r["type"], r["index"]) != (item["cve_id"], item["type"], item["index"]):
            bad.append(f"{where}: {i} key fields differ from the bank")
        if r["prompt"] != item["prompt"]:
            bad.append(f"{where}: {i} prompt is not the bank's")
        if arm == "sft" and r["completion"] != item["target"] + END:
            bad.append(f"{where}: {i} completion is not the bank target + end")
        elif arm == "distill_self":
            src = inputs.distill[i]
            if _strip_end(r["completion"]) != src["target"] or r["source"] != src["source"] or i not in ok["distill_self"]:
                bad.append(f"{where}: {i} completion is not the step-5 target ending in gold + end")
        elif arm == "dpo":
            pair = inputs.dpo[i]
            if _strip_end(r["chosen"]) != item["target"]:
                bad.append(f"{where}: {i} chosen is not the bank target + end")
            if _strip_end(r["rejected"]) != pair["rejected"] or r["rejected_dense"] != pair["rejected_dense"] \
                    or r["rejected_source"] != pair["source"] or i not in ok["dpo"]:
                bad.append(f"{where}: {i} rejected is not step 6's strict near miss + end")
        elif arm == "grpo":
            t = inputs.grpo_types[item["type"]]
            if r["gold"] != item["gold"] or r["max_completion_tokens"] != t["cap"] or r["band"] != t["band"] \
                    or r["dynamic_sampling"] != (t["band"] == "dynamic_sampling"):
                bad.append(f"{where}: {i} gold, cap or band differs from the bank and step 5")
    return bad


def check_base_file(inputs: Inputs, seed: int, rows: list[dict], where: str) -> list[str]:
    bad = []
    cves = [r["cve_id"] for r in rows]
    if Counter(cves) != Counter(inputs.train_cves(seed)):
        bad.append(f"{where}: documents are not exactly one per train CVE")
    for r in rows:
        text = r["text"]
        body = _strip_end(text, pinned.BASE_DOC_END)
        if body is None or pinned.COMPLETION_END in text:
            bad.append(f"{where}: {r['cve_id']} does not end in exactly one {pinned.BASE_DOC_END}")
        if r["cve_id"].lower() in text.lower():
            bad.append(f"{where}: {r['cve_id']} document contains its own CVE ID")
        if QUESTION_PHRASE in text or "<|im_start|>" in text:
            bad.append(f"{where}: {r['cve_id']} document contains a question or chat markup")
        fact = inputs.facts.get(r["cve_id"])
        if fact and (f"Weakness: {fact['cwe']}: " not in text or f"CVSS v3 base vector: {fact['cvss_vector']}\n" not in text):
            bad.append(f"{where}: {r['cve_id']} document lacks its CWE or CVSS label")
    return bad


def check_selection(inputs: Inputs, seed: int, config: str, keys: list[tuple[str, int]], where: str) -> list[str]:
    """M1: every train item once. M2: no dropped-type item, every other train item, copies within one of each
    other. M1-volume: exactly N x f items missing. Always N rows and no dev or test item."""
    bad = []
    train = {r["item_id"]: r["type"] for r in inputs.train_items(seed)}
    n = len(train)
    if len(keys) != n:
        bad.append(f"{where}: {len(keys)} rows, M1 has {n}")
    if len(set(keys)) != len(keys):
        bad.append(f"{where}: repeated (item_id, copy)")
    outside = {i for i, _ in keys} - set(train)
    if outside:
        bad.append(f"{where}: {len(outside)} items outside the seed's train set (dev or test), e.g. {sorted(outside)[:2]}")
    copies: dict[str, list[int]] = {}
    for i, c in keys:
        copies.setdefault(i, []).append(c)
    if any(sorted(cs) != list(range(len(cs))) for cs in copies.values()):
        bad.append(f"{where}: copies of an item are not numbered 0..k-1")
    count = Counter({i: len(cs) for i, cs in copies.items()})
    present = set(count)
    t, f = masks.dropped_type(config), masks.m1v_fraction(config)
    if config == masks.M1:
        expected = set(train)
    elif t is not None:
        expected = {i for i, it in train.items() if it != t}
    else:
        expected = None
        if len(set(train) - present) != n * f:
            bad.append(f"{where}: {len(set(train) - present)} items masked, expected {n} x {f}")
    if expected is not None and present != expected:
        bad.append(f"{where}: kept items are not the train items{'' if t is None else ' without ' + t}")
    if present:
        lo, hi = n // len(present), -(-n // len(present))
        if any(not lo <= k <= hi for k in count.values()):
            bad.append(f"{where}: copies outside {lo}..{hi}")
    return bad


def check(paths: Paths) -> list[str]:
    files = ConverterFiles.of(paths)
    if not files.meta.exists():
        return ["converters_meta.json is missing: run `python -m converters build`"]
    inputs = load_inputs(paths)
    meta = json.loads(files.meta.read_text(encoding="utf-8"))
    bad = []
    if meta["sources"] != inputs.sources:
        bad.append("converters_meta.json sources differ from the current inputs")
    test = {r["cve_id"] for r in read_jsonl(paths.split) if r["pool"] == "test"}
    ok = verified(inputs)
    bad += [f"distill_self.jsonl: {i} no longer ends in the gold answer" for i in sorted(set(inputs.distill) - ok["distill_self"])]
    bad += [f"dpo.jsonl: {i} is no longer a strict near miss" for i in sorted(set(inputs.dpo) - ok["dpo"])]

    m1v_masks: dict[str, list[set[str]]] = {c: [] for c in masks.M1V}
    for seed in pinned.PARTITION_SEEDS:
        if inputs.train_cves(seed) & test:
            bad.append(f"seed {seed}: a test CVE is in train")
        for config in masks.CONFIGS:
            keys_by_arm = {}
            for arm in masks.arms(config):
                rel, path = files.rel(seed, config, arm), files.training(seed, config, arm)
                if not path.exists():
                    bad.append(f"{rel}: missing")
                    continue
                content = read_content(path)
                rec = meta["outputs"].get(rel)
                if rec is None or rec["content_sha256"] != sha256_bytes(content):
                    bad.append(f"{rel}: content differs from converters_meta.json")
                rows = read_rows(path)
                if arm == "base":
                    bad += check_base_file(inputs, seed, rows, rel)
                    continue
                bad += check_question_file(inputs, ok, arm, rows, rel)
                keys_by_arm[arm] = [(r["item_id"], r["copy"]) for r in rows]
            if not keys_by_arm:
                continue
            first = next(iter(keys_by_arm.values()))
            for arm, keys in keys_by_arm.items():
                if keys != first:
                    bad.append(f"seed{seed}/{config}: {arm} rows differ from {next(iter(keys_by_arm))}'s")
            bad += check_selection(inputs, seed, config, first, f"seed{seed}/{config}")
            if config in m1v_masks:
                train = {r["item_id"] for r in inputs.train_items(seed)}
                m1v_masks[config].append(train - {i for i, _ in first})
    for config, sets in m1v_masks.items():
        if len(sets) > 1 and len({frozenset(s) for s in sets}) != len(sets):
            bad.append(f"{config}: two seeds have the same mask")
    for config in masks.M2:
        comp = masks.comparator(config)
        share = Fraction(masks.PER_CVE[masks.dropped_type(config)], pinned.ITEMS_PER_CVE)
        if masks.m1v_fraction(comp) != share:
            bad.append(f"{config}: comparator {comp} does not remove the same share ({share})")
    return bad
