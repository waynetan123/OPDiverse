"""Load PrimeVul v0.1 paired files into (vulnerable, patched) pairs with integrity checks."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import pinned

FIELDS = frozenset({
    "idx", "project", "commit_id", "project_url", "commit_url", "commit_message", "target", "func",
    "func_hash", "file_name", "file_hash", "cwe", "cve", "cve_desc", "nvd_url",
})
_CVE_ID = re.compile(r"CVE-\d{4}-\d{4,}")

# Integrity drop reasons, in the order they are checked; a pair's first failure is its reason.
INTEGRITY_REASONS = (
    "bad_target_order",
    "cross_commit_or_cve",
    "malformed_cve",
    "empty_function",
    "identical_text",
    "shared_patched_function",
)


@dataclass(frozen=True)
class Pair:
    pair_id: str
    cve_id: str
    split: str
    line: int                  # 1-indexed line of the vulnerable row in its file
    idx_vuln: int
    idx_patched: int
    commit_id: str
    commit_url: str
    project: str
    file_name: str | None
    vuln_raw: str
    patched_raw: str
    vuln: str                  # clean_function(vuln_raw)
    patched: str
    failures: tuple[str, ...]  # integrity failures, in INTEGRITY_REASONS order


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != FIELDS:
                raise ValueError(f"{path}:{n}: unexpected fields {sorted(set(row) ^ FIELDS)}")
            rows.append(row)
    if len(rows) % 2:
        raise ValueError(f"{path}: odd number of rows ({len(rows)}); paired files alternate vuln/patched")
    return rows


def load_pairs(paired_files: dict[str, Path]) -> list[Pair]:
    """Rebuild pairs from consecutive rows (vulnerable, then patched) of each paired file."""
    drafts = []
    for split, path in paired_files.items():
        rows = _read_rows(path)
        for k in range(0, len(rows), 2):
            drafts.append((split, k + 1, rows[k], rows[k + 1]))

    patched_uses = Counter(pinned.sha256_text(p["func"]) for _, _, _, p in drafts)
    pairs = []
    for split, line, v, p in drafts:
        vuln, patched = pinned.clean_function(v["func"]), pinned.clean_function(p["func"])
        checks = {
            "bad_target_order": not (v["target"] == 1 and p["target"] == 0),
            "cross_commit_or_cve": v["commit_id"] != p["commit_id"] or v["cve"] != p["cve"],
            "malformed_cve": not _CVE_ID.fullmatch(v["cve"] or ""),
            "empty_function": not vuln or not patched,
            "identical_text": vuln == patched,
            "shared_patched_function": patched_uses[pinned.sha256_text(p["func"])] > 1,
        }
        pair_id = pinned.sha256_text(
            "\x1f".join((v["commit_id"], pinned.sha256_text(v["func"]), pinned.sha256_text(p["func"])))
        )
        pairs.append(Pair(
            pair_id=pair_id,
            cve_id=v["cve"],
            split=split,
            line=line,
            idx_vuln=v["idx"],
            idx_patched=p["idx"],
            commit_id=v["commit_id"],
            commit_url=v["commit_url"],
            project=v["project"],
            file_name=None if v["file_name"] in (None, "None", "") else v["file_name"],
            vuln_raw=v["func"],
            patched_raw=p["func"],
            vuln=vuln,
            patched=patched,
            failures=tuple(r for r in INTEGRITY_REASONS if checks[r]),
        ))
    ids = Counter(p.pair_id for p in pairs)
    dupes = [i for i, c in ids.items() if c > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate pair ids; the pair key would not be unique")
    return pairs
