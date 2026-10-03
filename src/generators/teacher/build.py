"""Build distill_external.jsonl and dpo.jsonl (one row per non-test item, keyed by item_id) from the runs."""

from __future__ import annotations

from pathlib import Path

from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.tokens import TokenCounter
from frozen_model.prepare import load_bank

from ..bank.build import load_inputs
from ..bank.check import code_lines
from ..bank.files import BankFiles, generations_for
from ..progress import track
from . import dpo, prepare, traces
from .files import TeacherFiles
from .results import load_results


def _job_results(files: TeacherFiles, job: str, bank_sha: str, sources: dict) -> tuple[list[dict], dict, dict]:
    """(attempt-1 requests, item_id -> attempt-1 row, item_id -> attempt-2 row), recording input sha256s."""
    runs = {}
    for retry in (False, True):
        path = files.requests(job, retry)
        if retry and not path.exists():
            runs[retry] = ([], {})
            continue
        runs[retry] = load_results(path)
        name = path.stem.removesuffix("_requests")
        sources[f"{name}_requests_sha256"] = file_sha256(path)
        sources[f"{name}_generations_sha256"] = file_sha256(generations_for(path))
    for requests, _ in runs.values():
        if any(r["bank_sha256"] != bank_sha for r in requests):
            raise SystemExit(f"{job} requests were built from a different bank_nontest.jsonl")
    return runs[False][0], runs[False][1], runs[True][1]


def _check_coverage(job: str, requests: list[dict], bank: list[dict]) -> None:
    expected = sorted(r["item_id"] for r in bank if prepare.wanted(job, r))
    if sorted(r["item_id"] for r in requests) != expected:
        raise SystemExit(f"{job} requests do not cover exactly the non-test items they should")


def _extra_retries(job: str, rows: list[dict], retry: dict) -> None:
    extra = sorted(set(retry) - {r["item_id"] for r in rows if r["a1_reason"]})
    if extra:
        raise SystemExit(f"{len(extra)} {job} regenerations for items whose first attempt was valid, e.g. {extra[:3]}")


def build(paths: Paths, tokens: TokenCounter) -> dict:
    files = TeacherFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    facts, _, graph = load_inputs(paths)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    decisions = {d["cve_id"]: d for d in read_jsonl(BankFiles.of(paths).mcq_decisions)}
    sources = {"bank_nontest_sha256": bank_sha, "mcq_decisions_sha256": file_sha256(BankFiles.of(paths).mcq_decisions),
               "tokenizer_sha256": tokens.sha256}
    count = lambda text: tokens.count([text])[0]  # noqa: E731

    requests, first, retry = _job_results(files, "trace", bank_sha, sources)
    _check_coverage("trace", requests, bank)
    by_id = {r["item_id"]: r for r in bank}
    trace_rows = [traces.decide(by_id[r["item_id"]], first[r["item_id"]], retry.get(r["item_id"]), count, graph)
                  for r in track(requests, "distill-external")]
    _extra_retries("trace", trace_rows, retry)

    requests, first, retry = _job_results(files, "dpo", bank_sha, sources)
    _check_coverage("dpo", requests, bank)
    dpo_rows = []
    for item in track(bank, "DPO rejected"):
        if item["type"] == "mcq":
            dpo_rows.append(dpo.decide(item, None, None, code[item["cve_id"]], graph, decisions[item["cve_id"]]))
        else:
            dpo_rows.append(dpo.decide(item, first[item["item_id"]], retry.get(item["item_id"]), code[item["cve_id"]], graph))
    _extra_retries("dpo", [r for r in dpo_rows if r["type"] != "mcq"], retry)

    order = {r["item_id"]: i for i, r in enumerate(bank)}
    trace_rows.sort(key=lambda r: order[r["item_id"]])
    _write(files.distill_external, trace_rows)
    _write(files.dpo, dpo_rows)
    sources["distill_external_sha256"] = file_sha256(files.distill_external)
    sources["dpo_sha256"] = file_sha256(files.dpo)
    return {"n_items": len(bank), "distill_external": traces.summarise(trace_rows), "dpo": dpo.summarise(dpo_rows),
            "sources": sources, "external_model": pinned.EXTERNAL_MODEL}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes(rows))
