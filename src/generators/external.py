"""Run requests through the pinned external model (Messages API). The only module that imports anthropic.

    pip install -r requirements-external.txt
    cp .env.example .env    # then put your key in .env (git-ignored)
    PYTHONPATH=src python -m generators.external --requests data/bank/mcq_requests.jsonl [--workers 8]

Requests go out concurrently, and each result is appended to <prefix>_generations.partial.jsonl
as it arrives. An interrupted or failed run loses nothing: rerun the same command and only the
requests without a result are sent. When every request has one, the results are written, sorted, to
<prefix>_generations.jsonl and the partial file is removed. A request that fails in transport (after
the SDK's own retries) is re-sent by the next run rather than written as a result, so a network
error never uses up an item's one regeneration; a refusal is the model's answer and is kept.

The API key is read from ANTHROPIC_API_KEY in the .env file at the repository root, and only from there:
a key exported in the shell is ignored, so a run always uses the key in .env.

Also writes <prefix>_run_meta.json next to the requests file. Refuses to run if the code state is dirty, if any request departs from pinned.EXTERNAL_MODEL, or if a custom_id repeats.
The model has no dated snapshot and accepts no sampling parameters, so these files are the artifact:
they are cached and audited, never expected to regenerate identically. Server-side fallbacks are not
used, because they would switch models silently; a refusal is recorded and handled by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned
from etl.manifest import git_state
from etl.paths import ROOT

from .bank.files import generations_for, partial_for, run_meta_for
from .progress import Bar

CUSTOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
ENV_FILE = ROOT / ".env"
API_KEY_VAR = "ANTHROPIC_API_KEY"
WORKERS = 8
MAX_RETRIES = 8  # the SDK retries 429, 5xx and connection errors, with backoff, this many times
_SAMPLING = ("temperature", "top_p", "top_k", "seed")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without anthropic)
# ---------------------------------------------------------------------------


def message_params(user: str, schema: dict | None = None) -> dict:
    """The one request shape for every external job: pinned model, effort, thinking, max_tokens."""
    m = pinned.EXTERNAL_MODEL
    output_config: dict = {"effort": m["effort"]}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    return {
        "model": m["model"],
        "max_tokens": m["max_tokens"],
        "thinking": m["thinking"],
        "output_config": output_config,
        "messages": [{"role": "user", "content": user}],
    }


def custom_id(item_id: str, attempt: int) -> str:
    """The request's key in every output file: 'CVE-2021-1234:mcq:0', attempt 1 -> 'CVE-2021-1234_mcq_0_a1'.
    The [A-Za-z0-9_-] form is kept from the batch transport so the pilot's files stay valid."""
    cid = f"{item_id.replace(':', '_')}_a{attempt}"
    if not CUSTOM_ID.fullmatch(cid):
        raise ValueError(f"{cid!r} is not a valid custom_id")
    return cid


def load_requests(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    ids = [r["custom_id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate custom_id in the request file")
    return requests


def check_request_pins(requests: list[dict]) -> None:
    m = pinned.EXTERNAL_MODEL
    for r in requests:
        p, cid = r["params"], r["custom_id"]
        if not CUSTOM_ID.fullmatch(cid):
            raise SystemExit(f"{cid!r}: not a valid custom_id")
        if (p.get("model"), p.get("max_tokens"), p.get("thinking")) != (m["model"], m["max_tokens"], m["thinking"]):
            raise SystemExit(f"{cid}: model / max_tokens / thinking differ from pinned.EXTERNAL_MODEL")
        if p.get("output_config", {}).get("effort") != m["effort"]:
            raise SystemExit(f"{cid}: effort differs from pinned.EXTERNAL_MODEL")
        if bad := [k for k in (*_SAMPLING, "fallbacks") if k in p]:
            raise SystemExit(f"{cid}: {bad} must not be set (not pinned; fallbacks would switch models)")


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines. Blank lines and lines starting with # are skipped; an optional leading
    'export ' and one pair of matching surrounding quotes are removed."""
    out = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f".env line {n}: expected KEY=VALUE")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def api_key(path: Path = ENV_FILE) -> str:
    """ANTHROPIC_API_KEY from the .env file at the repository root. Never printed or recorded."""
    if not path.exists():
        raise SystemExit(f"{path} not found: copy .env.example to .env and set {API_KEY_VAR}")
    key = parse_env(path.read_text(encoding="utf-8")).get(API_KEY_VAR, "")
    if not key:
        raise SystemExit(f"{API_KEY_VAR} is missing or empty in {path}")
    return key


def require_clean_tree(root: Path = ROOT) -> dict:
    state = git_state(root)
    if state["commit"] is None or state["dirty"]:
        raise SystemExit(f"commit the code and pins before any external request (git state: {state})")
    return state


def _plain(obj):
    """SDK objects -> JSON-able values (to_dict when available; test doubles via __dict__)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {k: _plain(v) for k, v in vars(obj).items()}


def _row(custom_id_: str, requests_sha: str) -> dict:
    return {"custom_id": custom_id_, "requests_sha256": requests_sha, "result_type": None, "model": None,
            "message_id": None, "request_id": None, "stop_reason": None, "stop_category": None, "text": None,
            "usage": None, "error": None}


def message_row(custom_id_: str, msg, requests_sha: str) -> dict:
    """A reply -> one generations row. `text` joins the reply's text blocks; thinking blocks carry no
    text under the pinned display ("omitted") and are not stored. A refusal is a result, not an error."""
    details = getattr(msg, "stop_details", None)
    return {**_row(custom_id_, requests_sha),
            "result_type": "succeeded",
            "model": msg.model,
            "message_id": msg.id,
            "request_id": getattr(msg, "_request_id", None),
            "stop_reason": msg.stop_reason,
            "stop_category": getattr(details, "category", None) if details is not None else None,
            "text": "".join(b.text for b in msg.content if b.type == "text"),
            "usage": _plain(msg.usage)}


def error_row(custom_id_: str, exc: Exception, requests_sha: str) -> dict:
    """A request that failed after the SDK's retries. Never written to the final file: the next run re-sends it."""
    return {**_row(custom_id_, requests_sha),
            "result_type": "errored",
            "request_id": getattr(exc, "request_id", None),
            "error": {"type": type(exc).__name__, "status": getattr(exc, "status_code", None), "message": str(exc)[:500]}}


def summarise(rows: list[dict]) -> dict:
    usage = Counter()
    for row in rows:
        for k, v in (row["usage"] or {}).items():
            if isinstance(v, int):
                usage[k] += v
    return {
        "result_types": dict(sorted(Counter(r["result_type"] for r in rows).items())),
        "stop_reasons": dict(sorted(Counter(str(r["stop_reason"]) for r in rows).items())),
        "refusal_categories": dict(sorted(Counter(str(r["stop_category"]) for r in rows if r["stop_reason"] == "refusal").items())),
        "models": sorted({r["model"] for r in rows if r["model"]}),
        "usage_totals": dict(sorted(usage.items())),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def load_partial(path: Path, requests_sha: str) -> dict[str, dict]:
    """custom_id -> its succeeded row from an earlier, interrupted run of the same requests file."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # a line cut short by a crash is skipped; that request is simply re-sent
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("requests_sha256") != requests_sha:
                raise SystemExit(f"{path} came from a different requests file; move it aside before running")
            if row["result_type"] == "succeeded":
                done[row["custom_id"]] = row
    return done


def call(client, request: dict, requests_sha: str, errors: tuple[type[BaseException], ...]) -> dict:
    try:
        msg = client.messages.create(**request["params"])
    except errors as exc:
        return error_row(request["custom_id"], exc, requests_sha)
    return message_row(request["custom_id"], msg, requests_sha)


def run(client, requests: list[dict], partial: Path, requests_sha: str, errors: tuple[type[BaseException], ...],
        workers: int = WORKERS, label: str = "requests") -> tuple[dict[str, dict], list[dict]]:
    """Send every request without a succeeded row in `partial`, appending each result as it arrives.
    Returns (succeeded rows by custom_id, rows that failed this run). Ctrl-C stops sending new
    requests and waits for the ones in flight, so nothing already paid for is lost; a second Ctrl-C
    abandons them."""
    done = load_partial(partial, requests_sha)
    todo = [r for r in requests if r["custom_id"] not in done]
    failed: list[dict] = []
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "a", encoding="utf-8") as out, Bar(label, len(requests)) as bar:
        def record(row: dict) -> None:
            out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            out.flush()
            if row["result_type"] == "succeeded":
                done[row["custom_id"]] = row
            else:
                failed.append(row)
            bar.update(len(done), note=f"{len(failed):,} failed, will be re-sent" if failed else "")

        bar.update(len(done), note=f"resuming: {len(done):,} already done" if done else "")
        pool = ThreadPoolExecutor(max_workers=workers)
        pending = {pool.submit(call, client, r, requests_sha, errors) for r in todo}
        try:
            while pending:
                finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    record(future.result())
                bar.refresh()
        except KeyboardInterrupt:
            queued = sum(f.cancel() for f in pending)
            in_flight = [f for f in pending if not f.cancelled()]
            print(f"\ninterrupted: {queued:,} requests not sent; waiting for {len(in_flight):,} in flight "
                  "(Ctrl-C again to abandon them)", flush=True)
            try:
                for future in in_flight:
                    record(future.result())
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            raise SystemExit(f"stopped with {len(done):,}/{len(requests):,} done; rerun the same command to continue")
        pool.shutdown()
    return done, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m generators.external", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"concurrent requests (default {WORKERS})")
    args = ap.parse_args(argv)

    import anthropic

    out = generations_for(args.requests)
    if out.exists():
        raise SystemExit(f"{out} already exists: this requests file is complete")
    git = require_clean_tree()
    key = api_key()
    requests = load_requests(args.requests)
    check_request_pins(requests)
    requests_sha = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    client = anthropic.Anthropic(api_key=key, max_retries=MAX_RETRIES)
    partial = partial_for(args.requests)
    job = args.requests.stem.removesuffix("_requests")  # mcq, mcq_retry, ...
    started = datetime.now(timezone.utc)
    print(f"{job}: {len(requests):,} requests, {args.workers} at a time")
    done, failed = run(client, requests, partial, requests_sha, (anthropic.APIError,), args.workers, label=job)
    if failed:
        kinds = Counter(f"{r['error']['type']} {r['error']['status'] or ''}".strip() for r in failed)
        raise SystemExit(f"{len(failed):,} requests failed ({dict(kinds)}); {len(done):,} succeeded and are kept "
                         f"in {partial.name}. Rerun the same command to send only the failed ones.")

    rows = [done[r["custom_id"]] for r in sorted(requests, key=lambda r: r["custom_id"])]
    out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with open(partial, encoding="utf-8") as f:
        attempts = Counter(json.loads(line)["result_type"] for line in f if line.strip().startswith("{"))
    meta = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_session_started_utc": started.isoformat(timespec="seconds"),
        "transport": "messages",
        "workers": args.workers,
        "max_retries": MAX_RETRIES,
        "requests_sha256": requests_sha,
        "n_requests": len(requests),
        "transport_errors_resent": attempts.get("errored", 0),
        "git": git,
        "external_model": pinned.EXTERNAL_MODEL,
        "versions": {"anthropic": anthropic.__version__, "python": platform.python_version()},
        **summarise(rows),
    }
    run_meta_for(args.requests).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.unlink()
    print(f"wrote {len(rows):,} results to {out}: stop reasons {meta['stop_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
