"""Request files for the frozen-model session, built from the frozen non-test bank.

    audit_requests.jsonl           200 prompts per type, n = 8, ROLLOUT_SAMPLING, 512 tokens
    rationale_requests.jsonl       one hint-conditioned request per non-test item, greedy, 512 tokens
    rationale_retry_requests.jsonl the one regeneration of surface-invalid rationales, sampled

Every request names the pinned backbone and the exact sampling `sampling_for` gives it, so the
GPU runner can refuse anything else. A request file that already has generations never changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.tokens import TokenCounter
from generators.bank.files import BankFiles, generations_for, partial_for
from generators.progress import Bar
from probe.prompts import render_qwen_chat

from . import rationale
from .files import SessionFiles

JOBS = ("audit", "rationale")
TOKEN_CHUNK = 500  # prompts per tokenizer call; only sets the progress granularity


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def seed(*parts: str) -> int:
    """A per-request vLLM seed: deterministic, fits a signed 32-bit int."""
    return pinned.stable_rank(*parts) % 2**31


def sampling_for(job: str, item_id: str, attempt: int = 1) -> dict:
    """The one sampling config each request may carry. Audit: the rollout config at n = 8 and
    AUDIT_MAX_TOKENS. Rationale attempt 1: greedy EVAL_SAMPLING. Attempt 2: the rollout config, n = 1."""
    if job == "audit" and attempt == 1:
        return {**pinned.ROLLOUT_SAMPLING, "n": pinned.ROLLOUT_N, "max_tokens": pinned.AUDIT_MAX_TOKENS,
                "seed": seed(item_id, pinned.AUDIT_SALT)}
    if job == "rationale" and attempt == 1:
        return dict(pinned.EVAL_SAMPLING)
    if job == "rationale" and attempt == 2:
        return {**pinned.ROLLOUT_SAMPLING, "n": 1, "max_tokens": pinned.RATIONALE_MAX_TOKENS,
                "seed": seed(item_id, pinned.RATIONALE_SALT, "2")}
    raise ValueError(f"no sampling config for job {job!r}, attempt {attempt}")


def request_id(job: str, item_id: str, attempt: int) -> str:
    return f"{job}:{item_id}:a{attempt}"


def audit_sample(rows: list[dict]) -> list[dict]:
    """The AUDIT_PER_TYPE non-test CVEs with the lowest stable rank: their mcq, exact_id, cvss and
    line_loc items, and both find_error items of the lowest-ranked half. Bank order is kept."""
    if any(r["pool"] != "nontest" for r in rows):
        raise ValueError("the audit draws from non-test items only")
    ranked = sorted({r["cve_id"] for r in rows}, key=lambda c: (pinned.stable_rank(c, pinned.AUDIT_SALT), c))
    if len(ranked) < pinned.AUDIT_PER_TYPE:
        raise ValueError(f"only {len(ranked)} non-test CVEs; the audit needs {pinned.AUDIT_PER_TYPE}")
    chosen = set(ranked[: pinned.AUDIT_PER_TYPE])
    paired = set(ranked[: pinned.AUDIT_PER_TYPE // 2])
    return [r for r in rows if r["cve_id"] in (paired if r["type"] == "find_error" else chosen)]


def rationale_user(item: dict) -> str:
    """The hint-conditioned generation prompt. The training prompt stays the item's own `prompt`."""
    return pinned.RATIONALE_PROMPT.format(user=item["user"], target=item["target"])


def request_row(item: dict, job: str, attempt: int, user: str, prompt_tokens: int, bank_sha: str) -> dict:
    return {
        "request_id": request_id(job, item["item_id"], attempt),
        "job": job,
        "attempt": attempt,
        "item_id": item["item_id"],
        "cve_id": item["cve_id"],
        "type": item["type"],
        "index": item["index"],
        "messages": [{"role": "user", "content": user}],
        "prompt": render_qwen_chat(user),
        "prompt_tokens": prompt_tokens,
        "model": pinned.TOKENIZER_REPO,
        "revision": pinned.TOKENIZER_REVISION,
        "sampling": sampling_for(job, item["item_id"], attempt),
        "bank_sha256": bank_sha,
    }


def max_model_len(requests: list[dict]) -> int:
    """Capacity the session needs: the longest prompt plus its own max_tokens. Changes no output."""
    return max(r["prompt_tokens"] + r["sampling"]["max_tokens"] for r in requests)


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------


def load_bank(paths: Paths) -> tuple[list[dict], str]:
    """(non-test bank rows, sha256 of bank_nontest.jsonl). Refuses a bank whose inputs have changed."""
    files = BankFiles.of(paths)
    meta = json.loads(files.meta.read_text(encoding="utf-8"))
    for key, path in (("facts_sha256", paths.facts), ("split_sha256", paths.split)):
        if meta["sources"][key] != file_sha256(path):
            raise SystemExit(f"bank_meta.json {key} does not match {path.name}: the bank is not the frozen one")
    if meta["template_version"] != pinned.BANK_TEMPLATE_VERSION:
        raise SystemExit(f"bank template {meta['template_version']} != pinned {pinned.BANK_TEMPLATE_VERSION}")
    path = files.bank("nontest")
    return read_jsonl(path), file_sha256(path)


def count_tokens(tokens: TokenCounter, prompts: list[str], label: str) -> list[int]:
    out: list[int] = []
    with Bar(label, len(prompts)) as bar:
        for start in range(0, len(prompts), TOKEN_CHUNK):
            out += tokens.count(prompts[start:start + TOKEN_CHUNK])
            bar.advance(min(TOKEN_CHUNK, len(prompts) - start))
    return out


def write_requests(path: Path, rows: list[dict]) -> None:
    body = jsonl_bytes(rows)
    started = generations_for(path).exists() or partial_for(path).exists()
    if started and path.exists() and path.read_bytes() != body:
        raise SystemExit(f"{path} already has generations and would change; the requests they came from must stay as sent")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def load_results(requests_path: Path) -> tuple[list[dict], dict[str, dict]]:
    """(requests, request_id -> generation row) for a finished request file. Exits if the generations
    are missing, incomplete, or were made from different requests or prompts."""
    out = generations_for(requests_path)
    if not out.exists():
        raise SystemExit(f"{out} is missing: run `python -m frozen_model.run_vllm --requests {requests_path}` first")
    requests = read_jsonl(requests_path)
    sha = file_sha256(requests_path)
    rows = {r["request_id"]: r for r in read_jsonl(out)}
    if set(rows) != {r["request_id"] for r in requests}:
        raise SystemExit(f"{out.name} does not cover exactly the requests in {requests_path.name}")
    for r in requests:
        g = rows[r["request_id"]]
        if g["requests_sha256"] != sha:
            raise SystemExit(f"{out.name} came from a different {requests_path.name}")
        if g["prompt_sha256"] != hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest():
            raise SystemExit(f"{r['request_id']}: generation was made from a different prompt")
        if len(g["outputs"]) != r["sampling"]["n"]:
            raise SystemExit(f"{r['request_id']}: {len(g['outputs'])} outputs, expected {r['sampling']['n']}")
    return requests, rows


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def prepare(paths: Paths, tokens: TokenCounter) -> dict:
    files = SessionFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    if tokens.sha256 != json.loads(BankFiles.of(paths).meta.read_text(encoding="utf-8"))["sources"]["tokenizer_sha256"]:
        raise SystemExit("the tokenizer differs from the one the bank was counted with")
    audit = [request_row(r, "audit", 1, r["user"], r["prompt_tokens"], bank_sha) for r in audit_sample(bank)]
    users = [rationale_user(r) for r in bank]
    counts = count_tokens(tokens, [render_qwen_chat(u) for u in users], "Rationale prompt tokens")
    rationales = [request_row(r, "rationale", 1, u, n, bank_sha) for r, u, n in zip(bank, users, counts)]
    write_requests(files.audit_requests, audit)
    write_requests(files.rationale_requests, rationales)
    return {"audit": len(audit), "rationale": len(rationales), "max_model_len": max_model_len(audit + rationales),
            "longest_rationale_prompt": max(counts)}


def prepare_retry(paths: Paths, tokens: TokenCounter) -> list[dict]:
    """Requests for the one regeneration: every item whose attempt-1 rationale is surface-invalid."""
    files = SessionFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    graph = load_cwe_graph(paths.cwe_xml)
    requests, results = load_results(files.rationale_requests)
    failed = []
    for r in requests:
        out = results[r["request_id"]]["outputs"][0]
        if rationale.validate(by_id[r["item_id"]], out["text"], out["finish_reason"], graph)[1] is not None:
            failed.append(by_id[r["item_id"]])
    users = [rationale_user(r) for r in failed]
    counts = count_tokens(tokens, [render_qwen_chat(u) for u in users], "Retry prompt tokens") if failed else []
    rows = [request_row(r, "rationale", 2, u, n, bank_sha) for r, u, n in zip(failed, users, counts)]
    write_requests(files.rationale_retry_requests, rows)
    return rows
