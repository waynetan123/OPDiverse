"""Run requests through the pinned external model (Message Batches). The only module that imports anthropic.

    pip install -r requirements-external.txt
    cp .env.example .env    # then put your key in .env (git-ignored)
    PYTHONPATH=src python -m generators.external --requests data/bank/mcq_requests.jsonl

The API key is read from ANTHROPIC_API_KEY in the .env file at the repository root, and only from there:
a key exported in the shell is ignored, so a run always uses the key in .env.

Writes <prefix>_generations.jsonl and <prefix>_run_meta.json next to the requests file. Refuses to run
if the code state is dirty, if any request departs from pinned.EXTERNAL_MODEL, or if a custom_id repeats.
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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned
from etl.manifest import git_state
from etl.paths import ROOT

from .bank.files import generations_for, run_meta_for
from .progress import Bar, track

CUSTOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
ENV_FILE = ROOT / ".env"
API_KEY_VAR = "ANTHROPIC_API_KEY"
POLL_SECONDS = 60
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
    """Batch custom IDs allow [A-Za-z0-9_-] only: 'CVE-2021-1234:mcq:0', attempt 1 -> 'CVE-2021-1234_mcq_0_a1'."""
    cid = f"{item_id.replace(':', '_')}_a{attempt}"
    if not CUSTOM_ID.fullmatch(cid):
        raise ValueError(f"{cid!r} is not a valid batch custom_id")
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
            raise SystemExit(f"{cid!r}: not a valid batch custom_id")
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


def result_row(result) -> dict:
    """One batch result -> one generations row. `text` joins the reply's text blocks; thinking
    blocks carry no text under the pinned display ("omitted") and are not stored."""
    r = result.result
    row = {"custom_id": result.custom_id, "result_type": r.type, "model": None, "stop_reason": None,
           "stop_category": None, "text": None, "usage": None, "error": None, "message_id": None}
    if r.type == "succeeded":
        msg = r.message
        details = getattr(msg, "stop_details", None)
        row.update({
            "model": msg.model,
            "message_id": msg.id,
            "stop_reason": msg.stop_reason,
            "stop_category": getattr(details, "category", None) if details is not None else None,
            "text": "".join(b.text for b in msg.content if b.type == "text"),
            "usage": _plain(msg.usage),
        })
    elif r.type == "errored":
        row["error"] = _plain(r.error)
    return row


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
# Batch run
# ---------------------------------------------------------------------------


def finished(counts) -> int:
    """Requests the batch is done with, whatever their outcome."""
    return sum(getattr(counts, k, 0) or 0 for k in ("succeeded", "errored", "canceled", "expired"))


def progress_note(batch) -> str:
    c = batch.request_counts
    note = f"{batch.processing_status}: {c.succeeded:,} ok"
    failed = (c.errored or 0) + (c.canceled or 0) + (c.expired or 0)
    return note + (f", {failed:,} failed" if failed else "")


def wait_for_batch(client, batch_id: str, total: int, label: str = "batch", poll_seconds: int = POLL_SECONDS, sleep=time.sleep):
    """Poll until the batch ends, with a progress bar. The API is asked once per poll_seconds; the
    bar redraws every second in between so the clock keeps moving. Without a terminal, one status
    line is printed per poll instead."""
    with Bar(label, total) as bar:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            bar.update(finished(batch.request_counts), note=progress_note(batch))
            if not bar.enabled:
                print(f"  {progress_note(batch)} ({finished(batch.request_counts):,}/{total:,} done)", flush=True)
            if batch.processing_status == "ended":
                return batch
            for _ in range(poll_seconds):
                sleep(1)
                bar.refresh()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m generators.external", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--resume", metavar="BATCH_ID", help="collect an already-submitted batch instead of creating one")
    args = ap.parse_args(argv)

    import anthropic

    git = require_clean_tree()
    key = api_key()
    requests = load_requests(args.requests)
    check_request_pins(requests)
    requests_sha = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    client = anthropic.Anthropic(api_key=key)
    started = datetime.now(timezone.utc)
    if args.resume:
        batch_id = args.resume
    else:
        print(f"submitting {len(requests)} requests ...")
        batch = client.messages.batches.create(
            requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in requests])
        batch_id = batch.id
        # Written before polling so an interrupted run can be resumed with --resume.
        args.requests.with_name(args.requests.stem + ".batch_id").write_text(batch_id + "\n", encoding="utf-8")
    print(f"batch {batch_id}: {len(requests)} requests")
    job = args.requests.stem.removesuffix("_requests")  # mcq, mcq_retry, ...
    batch = wait_for_batch(client, batch_id, len(requests), label=f"{job} batch")
    results = track(client.messages.batches.results(batch_id), f"{job} results", total=len(requests))
    rows = sorted((result_row(r) for r in results), key=lambda r: r["custom_id"])
    wanted = {r["custom_id"] for r in requests}
    got = {r["custom_id"] for r in rows}
    if got != wanted:
        raise SystemExit(f"results don't match requests: {len(wanted - got)} missing, {len(got - wanted)} unexpected")
    out = generations_for(args.requests)
    out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "requests_sha256": requests_sha,
        "n_requests": len(requests),
        "git": git,
        "external_model": pinned.EXTERNAL_MODEL,
        "versions": {"anthropic": anthropic.__version__, "python": platform.python_version()},
        "request_counts": _plain(batch.request_counts),
        **summarise(rows),
    }
    run_meta_for(args.requests).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} results to {out}: {meta['result_types']}, stop reasons {meta['stop_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
