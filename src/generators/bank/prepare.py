"""MCQ request files for the external model: the pilot, the full run, and the one regeneration."""

from __future__ import annotations

from etl.build import jsonl_bytes, read_jsonl
from etl.paths import Paths

from .. import external
from . import mcq
from .build import load_inputs
from .files import BankFiles
from ..progress import track


def prepare_mcq(paths: Paths, pilot: bool = False, retry: bool = False) -> tuple[BankFiles, list[dict]]:
    facts, split, graph = load_inputs(paths)
    pool_of = {c: s["pool"] for c, s in split.items()}
    files = BankFiles.of(paths, pilot=pilot)
    if retry:
        if pilot:
            raise SystemExit("the pilot is never regenerated")
        first = {r["custom_id"]: r for r in read_jsonl(files.mcq_generations)}
        rows = []
        for f in track(facts, "MCQ retry requests"):
            status, proposals = mcq.parse_generation(first.get(external.custom_id(f"{f['cve_id']}:mcq:0", 1)))
            if mcq.needs_retry(status, mcq.admit(proposals, f["cwe"], graph)[0]):
                rows.append(mcq.request_row(f, pool_of[f["cve_id"]], graph, attempt=2))
        path = files.mcq_retry_requests
    else:
        chosen = mcq.pilot_sample(facts, pool_of) if pilot else facts
        rows = [mcq.request_row(f, pool_of[f["cve_id"]], graph, attempt=1) for f in track(chosen, "MCQ requests")]
        path = files.mcq_requests
    external.check_request_pins(rows)
    body = jsonl_bytes(rows)
    generations = files.mcq_retry_generations if retry else files.mcq_generations
    if generations.exists() and path.exists() and path.read_bytes() != body:
        raise SystemExit(f"{path} already has generations and would change; the requests they came from must stay as sent")
    files.dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return files, rows
