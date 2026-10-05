"""Reading the external runner's finished generation files back, matched to their requests by custom_id."""

from __future__ import annotations

from pathlib import Path

from etl.build import read_jsonl
from etl.manifest import file_sha256

from ..bank.files import generations_for


def stop_status(row: dict) -> str:
    """'ok' for a reply that ended on its own (end_turn); otherwise its stop reason ('refusal', 'max_tokens', ...)."""
    if row["result_type"] != "succeeded":
        return str(row["result_type"])
    return "ok" if row["stop_reason"] == "end_turn" else str(row["stop_reason"])


def load_results(requests_path: Path) -> tuple[list[dict], dict[str, dict]]:
    """(requests, item_id -> generation row) for a finished request file. Exits if the generations are
    missing, do not cover exactly these requests, or were made from a different request file."""
    out = generations_for(requests_path)
    if not out.exists():
        raise SystemExit(f"{out} is missing: run `python -m generators.external --requests {requests_path}` first")
    requests = read_jsonl(requests_path)
    sha = file_sha256(requests_path)
    rows = {r["custom_id"]: r for r in read_jsonl(out)}
    if set(rows) != {r["custom_id"] for r in requests}:
        raise SystemExit(f"{out.name} does not cover exactly the requests in {requests_path.name}")
    if any(g["requests_sha256"] != sha for g in rows.values()):
        raise SystemExit(f"{out.name} came from a different {requests_path.name}")
    return requests, {r["item_id"]: rows[r["custom_id"]] for r in requests}
