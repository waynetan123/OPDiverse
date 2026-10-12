"""The test evaluation request file for the one-day run, from the frozen test bank.

    test_requests.jsonl   every test item (all five types, every test CVE): each M2 run's last checkpoint

Rows are step 7's evaluation requests (engine_check.prepare.request_row): the bank's byte-identical prompt, the
pinned backbone and EVAL_SAMPLING, so evaluate.run_vllm accepts them unchanged.
"""

from __future__ import annotations

import json

from engine_check.prepare import request_row
from etl import pinned
from etl.build import read_jsonl
from etl.paths import Paths
from evaluate.run_vllm import check_request_pins
from frozen_model.prepare import load_bank, write_requests
from generators.bank.files import BankFiles

from .files import ShortFiles


def prepare(paths: Paths) -> dict:
    files = ShortFiles.of(paths)
    bank, bank_sha = load_bank(paths, "test")
    nontest, _ = load_bank(paths)
    test = {r["cve_id"] for r in read_jsonl(paths.split) if r["pool"] == "test"}
    counts = json.loads(BankFiles.of(paths).meta.read_text(encoding="utf-8"))["counts"]["test"]
    cves = {r["cve_id"] for r in bank}
    if cves != test:
        raise SystemExit(f"the test bank covers {len(cves)} CVEs; split.jsonl's test pool has {len(test)}"
                         + (f", e.g. missing {sorted(test - cves)[:3]}" if test - cves else ""))
    if any(r["pool"] != "test" for r in bank) or test & {r["cve_id"] for r in nontest}:
        raise SystemExit("the test bank and the non-test bank overlap")
    if len(bank) != len(test) * pinned.ITEMS_PER_CVE or (counts["cves"], counts["items"]) != (len(test), len(bank)):
        raise SystemExit(f"{len(bank)} test items for {len(test)} CVEs; bank_meta.json counts {counts}")
    rows = [request_row(r, bank_sha) for r in bank]
    check_request_pins(rows)
    write_requests(files.test_requests, rows)
    return {"cves": len(test), "items": len(rows), "max_prompt_tokens": max(r["prompt_tokens"] for r in rows)}
