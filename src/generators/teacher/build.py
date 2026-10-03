"""Build dpo.jsonl (one row per non-test item, keyed by item_id) from the external runs and the MCQ rule."""

from __future__ import annotations

from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from etl.manifest import file_sha256
from etl.paths import Paths
from frozen_model.prepare import load_bank

from ..bank.build import load_inputs
from ..bank.check import code_lines
from ..bank.files import BankFiles, generations_for
from ..progress import track
from . import dpo, prepare
from .files import TeacherFiles
from .results import load_results


def _results(files: TeacherFiles, bank_sha: str, sources: dict) -> tuple[list[dict], dict, dict]:
    """(attempt-1 requests, item_id -> attempt-1 row, item_id -> attempt-2 row), recording input sha256s."""
    runs = {}
    for retry in (False, True):
        path = files.requests(retry)
        if retry and not path.exists():
            runs[retry] = ([], {})
            continue
        runs[retry] = load_results(path)
        name = path.stem.removesuffix("_requests")
        sources[f"{name}_requests_sha256"] = file_sha256(path)
        sources[f"{name}_generations_sha256"] = file_sha256(generations_for(path))
    for requests, _ in runs.values():
        if any(r["bank_sha256"] != bank_sha for r in requests):
            raise SystemExit("DPO requests were built from a different bank_nontest.jsonl")
    return runs[False][0], runs[False][1], runs[True][1]


def build(paths: Paths) -> dict:
    files = TeacherFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    facts, _, graph = load_inputs(paths)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    decisions_path = BankFiles.of(paths).mcq_decisions
    decisions = {d["cve_id"]: d for d in read_jsonl(decisions_path)}
    sources = {"bank_nontest_sha256": bank_sha, "mcq_decisions_sha256": file_sha256(decisions_path)}

    requests, first, retry = _results(files, bank_sha, sources)
    if sorted(r["item_id"] for r in requests) != sorted(r["item_id"] for r in bank if prepare.wanted(r)):
        raise SystemExit("DPO requests do not cover exactly the non-test items they should")
    rows = []
    for item in track(bank, "DPO rejected"):
        if item["type"] == "mcq":
            rows.append(dpo.decide(item, None, None, code[item["cve_id"]], graph, decisions[item["cve_id"]]))
        else:
            rows.append(dpo.decide(item, first[item["item_id"]], retry.get(item["item_id"]), code[item["cve_id"]], graph))
    extra = sorted(set(retry) - {r["item_id"] for r in rows if r["a1_reason"]})
    if extra:
        raise SystemExit(f"{len(extra)} DPO regenerations for items whose first proposal was valid, e.g. {extra[:3]}")

    files.dpo.parent.mkdir(parents=True, exist_ok=True)
    files.dpo.write_bytes(jsonl_bytes(rows))
    sources["dpo_sha256"] = file_sha256(files.dpo)
    return {"n_items": len(bank), "dpo": dpo.summarise(rows), "sources": sources, "external_model": pinned.EXTERNAL_MODEL}
