"""Evaluation request files for the sweep, from the frozen non-test bank and seed 0's partition.

    dev_requests.jsonl          every dev item of SWEEP_SEED (full dev): LR selection
    checkpoint_requests.jsonl   the seed's checkpoint subsample, a subset of dev: checkpoint selection, and the scale

Rows are step 7's evaluation requests (engine_check.prepare.request_row): the bank's byte-identical prompt, the
pinned backbone and EVAL_SAMPLING, so evaluate.run_vllm accepts them unchanged. Test and train are never read.
"""

from __future__ import annotations

import json

from converters.files import ConverterFiles
from engine_check.prepare import request_row
from etl import pinned
from etl.build import read_jsonl
from etl.paths import Paths
from evaluate.run_vllm import check_request_pins
from frozen_model.prepare import load_bank, write_requests

from .files import SweepFiles


def seed_cves(partition: list[dict], seed: int) -> tuple[set[str], set[str], set[str]]:
    """(train, dev, checkpoint subsample) CVEs of one seed."""
    s = str(seed)
    train = {r["cve_id"] for r in partition if r["role"][s] == "train"}
    dev = {r["cve_id"] for r in partition if r["role"][s] == "dev"}
    checkpoint = {r["cve_id"] for r in partition if seed in r["checkpoint_seeds"]}
    return train, dev, checkpoint


def prepare(paths: Paths, seed: int = pinned.SWEEP_SEED) -> dict:
    files = SweepFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    partition = read_jsonl(paths.partition)
    train, dev, checkpoint = seed_cves(partition, seed)
    test = {r["cve_id"] for r in read_jsonl(paths.split) if r["pool"] == "test"}
    manifest = json.loads(ConverterFiles.of(paths).manifest.read_text(encoding="utf-8"))
    if sorted(checkpoint) != manifest["seeds"][str(seed)]["checkpoint_cves"]:
        raise SystemExit("the checkpoint subsample differs between partition.jsonl and the converters' manifest.json")
    if not checkpoint <= dev or dev & train or dev & test:
        raise SystemExit(f"seed {seed}: the checkpoint subsample must sit inside dev, and dev outside train and test")
    rows = {which: [request_row(r, bank_sha) for r in bank if r["cve_id"] in cves]
            for which, cves in (("dev", dev), ("checkpoint", checkpoint))}
    for which, cves in (("dev", dev), ("checkpoint", checkpoint)):
        if len(rows[which]) != len(cves) * pinned.ITEMS_PER_CVE:
            raise SystemExit(f"{which}: {len(rows[which])} bank items for {len(cves)} CVEs")
        check_request_pins(rows[which])
        write_requests(files.requests(which), rows[which])
    return {"seed": seed, "dev_cves": len(dev), "dev_items": len(rows["dev"]),
            "checkpoint_cves": len(checkpoint), "checkpoint_items": len(rows["checkpoint"])}
