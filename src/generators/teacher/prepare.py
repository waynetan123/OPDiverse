"""Request files for the external-model job, built from the frozen non-test bank.

    dpo_requests.jsonl          one near-miss request per non-test item except MCQ (rule-defined)
    dpo_retry_requests.jsonl    the one regeneration of invalid first proposals, identical requests

The pilot (data/teacher/pilot) takes every item of the TEACHER_PILOT_CVES lowest-ranked non-test CVEs.
It is never regenerated and never enters the final files. A request file that already has generations
never changes.
"""

from __future__ import annotations

from etl import pinned
from etl.cwe_graph import CweGraph, load_cwe_graph
from etl.paths import Paths
from frozen_model.prepare import load_bank, write_requests

from .. import external
from ..bank.build import load_inputs
from ..bank.check import code_lines
from ..progress import track
from . import dpo
from .files import TeacherFiles
from .results import load_results

JOB = "dpo"


def request_row(item: dict, attempt: int, graph: CweGraph, bank_sha: str) -> dict:
    return {
        "custom_id": external.custom_id(f"{JOB}:{item['item_id']}", attempt),
        "job": JOB,
        "attempt": attempt,
        "item_id": item["item_id"],
        "cve_id": item["cve_id"],
        "type": item["type"],
        "index": item["index"],
        "bank_sha256": bank_sha,
        "params": external.message_params(dpo.request_user(item, graph), dpo.schema_for(item)),
    }


def wanted(item: dict) -> bool:
    return item["type"] in pinned.DPO_REQUEST_TYPES


def pilot_items(bank: list[dict]) -> list[dict]:
    """Every item of the TEACHER_PILOT_CVES non-test CVEs with the lowest stable rank, bank order kept."""
    ranked = sorted({r["cve_id"] for r in bank}, key=lambda c: (pinned.stable_rank(c, pinned.TEACHER_SALTS["pilot"]), c))
    chosen = set(ranked[: pinned.TEACHER_PILOT_CVES])
    return [r for r in bank if r["cve_id"] in chosen]


def prepare(paths: Paths, pilot: bool = False) -> int:
    files = TeacherFiles.of(paths, pilot=pilot)
    bank, bank_sha = load_bank(paths)
    graph = load_cwe_graph(paths.cwe_xml)
    chosen = pilot_items(bank) if pilot else bank
    rows = [request_row(r, 1, graph, bank_sha) for r in track(chosen, "DPO requests") if wanted(r)]
    external.check_request_pins(rows)
    write_requests(files.requests(), rows)
    return len(rows)


def prepare_retry(paths: Paths) -> int:
    """The one regeneration: every item whose first proposal is not an admissible near miss. Writes no file
    if there is nothing to regenerate."""
    files = TeacherFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    facts, _, graph = load_inputs(paths)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    requests, results = load_results(files.requests())
    failed = [by_id[r["item_id"]] for r in track(requests, "DPO validity")
              if dpo.validate(by_id[r["item_id"]], results[r["item_id"]], code[r["cve_id"]], graph)[0] is None]
    rows = [request_row(item, 2, graph, bank_sha) for item in failed]
    if rows:
        external.check_request_pins(rows)
        write_requests(files.requests(retry=True), rows)
    return len(rows)
