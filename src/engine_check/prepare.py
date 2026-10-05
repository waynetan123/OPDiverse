"""Request files for the engine-agreement check, built from the frozen non-test bank.

    vllm_a_requests.jsonl   the sample's 1,200 items in bank order (pass A; also what the HF reference reads)
    vllm_b_requests.jsonl   the same rows in a seeded shuffle (pass B: vLLM's own batch noise)

The sample: the ENGINE_CVES lowest-ranked CVEs of the late window, the latest ENGINE_LATE_WINDOW of the
non-test pool by (published, cve_id), which is the plan's pool for dev. Test is never read.
"""

from __future__ import annotations

import math

from etl import pinned
from etl.build import read_jsonl
from etl.paths import Paths
from evaluate.run_vllm import JOB, check_request_pins, request_id
from frozen_model.prepare import load_bank, write_requests

from .files import EngineFiles


def late_window(facts: list[dict], pool_of: dict[str, str]) -> list[str]:
    """The latest ENGINE_LATE_WINDOW of the non-test pool by (published, cve_id), in that order."""
    nontest = sorted((f["published"], f["cve_id"]) for f in facts if pool_of[f["cve_id"]] == "nontest")
    k = math.ceil(len(nontest) * pinned.ENGINE_LATE_WINDOW)
    return [cve for _, cve in nontest[len(nontest) - k:]]


def sample(window: list[str]) -> list[str]:
    """The ENGINE_CVES window CVEs with the lowest stable rank, sorted by CVE ID."""
    if len(window) < pinned.ENGINE_CVES:
        raise ValueError(f"the late window has {len(window)} CVEs; the check needs {pinned.ENGINE_CVES}")
    ranked = sorted(window, key=lambda c: (pinned.stable_rank(c, pinned.ENGINE_SALT), c))
    return sorted(ranked[: pinned.ENGINE_CVES])


def request_row(item: dict, bank_sha: str) -> dict:
    return {
        "request_id": request_id(item["item_id"]),
        "job": JOB,
        "attempt": 1,
        "item_id": item["item_id"],
        "cve_id": item["cve_id"],
        "type": item["type"],
        "index": item["index"],
        "messages": [{"role": "user", "content": item["user"]}],
        "prompt": item["prompt"],
        "prompt_tokens": item["prompt_tokens"],
        "model": pinned.TOKENIZER_REPO,
        "revision": pinned.TOKENIZER_REVISION,
        "sampling": dict(pinned.EVAL_SAMPLING),
        "bank_sha256": bank_sha,
    }


def shuffled(rows: list[dict]) -> list[dict]:
    """Pass B's order: a seeded shuffle, so batch companions differ from pass A's."""
    return sorted(rows, key=lambda r: (pinned.stable_rank(r["request_id"], pinned.ENGINE_SALT, "b"), r["request_id"]))


def prepare(paths: Paths) -> dict:
    files = EngineFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    facts = read_jsonl(paths.facts)
    pool_of = {r["cve_id"]: r["pool"] for r in read_jsonl(paths.split)}
    window = late_window(facts, pool_of)
    chosen = set(sample(window))
    rows = [request_row(r, bank_sha) for r in bank if r["cve_id"] in chosen]
    if len(rows) != pinned.ENGINE_CVES * pinned.ITEMS_PER_CVE or any(pool_of[r["cve_id"]] != "nontest" for r in rows):
        raise SystemExit("the engine-check sample is not ENGINE_CVES whole non-test CVEs")
    check_request_pins(rows)
    write_requests(files.vllm_requests("a"), rows)
    write_requests(files.vllm_requests("b"), shuffled(rows))
    published = {f["cve_id"]: f["published"] for f in facts}
    return {"requests": len(rows), "cves": len(chosen), "late_window": len(window),
            "window_from": published[window[0]][:10], "window_to": published[window[-1]][:10]}
