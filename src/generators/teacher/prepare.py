"""Request files for the external-model jobs, built from the frozen non-test bank.

    trace_requests.jsonl        one unhinted written-reasoning request per non-test item (distill-external)
    dpo_requests.jsonl          one near-miss request per non-test item except MCQ (rule-defined)
    *_retry_requests.jsonl      the one regeneration of invalid first attempts, identical requests

The pilot (data/teacher/pilot) takes every item of the TEACHER_PILOT_CVES lowest-ranked non-test CVEs.
It is never regenerated and never enters the final files. A request file that already has generations
never changes.
"""

from __future__ import annotations

from etl import pinned
from etl.cwe_graph import CweGraph, load_cwe_graph
from etl.paths import Paths
from etl.tokens import TokenCounter
from frozen_model.prepare import load_bank, write_requests

from .. import external
from ..bank.build import load_inputs
from ..bank.check import code_lines
from ..progress import track
from . import dpo, traces
from .files import TeacherFiles
from .results import load_results


def trace_user(item: dict) -> str:
    """The item's own question with written-out reasoning asked for. The training prompt stays `prompt`."""
    return pinned.TRACE_PROMPT.format(user=item["user"], words=pinned.TRACE_WORDS)


def request_row(item: dict, job: str, attempt: int, graph: CweGraph, bank_sha: str) -> dict:
    if job == "trace":
        params = external.message_params(trace_user(item))
    elif job == "dpo":
        params = external.message_params(dpo.request_user(item, graph), dpo.schema_for(item))
    else:
        raise ValueError(f"unknown job {job!r}")
    return {
        "custom_id": external.custom_id(f"{job}:{item['item_id']}", attempt),
        "job": job,
        "attempt": attempt,
        "item_id": item["item_id"],
        "cve_id": item["cve_id"],
        "type": item["type"],
        "index": item["index"],
        "bank_sha256": bank_sha,
        "params": params,
    }


def wanted(job: str, item: dict) -> bool:
    return job == "trace" or item["type"] in pinned.DPO_REQUEST_TYPES


def pilot_items(bank: list[dict]) -> list[dict]:
    """Every item of the TEACHER_PILOT_CVES non-test CVEs with the lowest stable rank, bank order kept."""
    ranked = sorted({r["cve_id"] for r in bank}, key=lambda c: (pinned.stable_rank(c, pinned.TEACHER_SALTS["pilot"]), c))
    chosen = set(ranked[: pinned.TEACHER_PILOT_CVES])
    return [r for r in bank if r["cve_id"] in chosen]


def prepare(paths: Paths, pilot: bool = False) -> dict[str, int]:
    files = TeacherFiles.of(paths, pilot=pilot)
    bank, bank_sha = load_bank(paths)
    graph = load_cwe_graph(paths.cwe_xml)
    chosen = pilot_items(bank) if pilot else bank
    out = {}
    for job in pinned.TEACHER_JOBS:
        rows = [request_row(r, job, 1, graph, bank_sha) for r in track(chosen, f"{job} requests") if wanted(job, r)]
        external.check_request_pins(rows)
        write_requests(files.requests(job), rows)
        out[job] = len(rows)
    return out


def prepare_retry(paths: Paths, tokens: TokenCounter) -> dict[str, int]:
    """The one regeneration: every item whose first trace or first DPO proposal is invalid. A job with
    nothing to regenerate writes no file."""
    files = TeacherFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    facts, _, graph = load_inputs(paths)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    count = lambda text: tokens.count([text])[0]  # noqa: E731
    out = {}
    for job in pinned.TEACHER_JOBS:
        requests, results = load_results(files.requests(job))
        failed = []
        for r in track(requests, f"{job} validity"):
            item = by_id[r["item_id"]]
            if job == "trace":
                invalid = traces.validate(item, results[r["item_id"]], count, graph)[1] is not None
            else:
                invalid = dpo.validate(item, results[r["item_id"]], code[item["cve_id"]], graph)[0] is None
            if invalid:
                failed.append(item)
        rows = [request_row(item, job, 2, graph, bank_sha) for item in failed]
        if rows:
            external.check_request_pins(rows)
            write_requests(files.requests(job, retry=True), rows)
        out[job] = len(rows)
    return out
