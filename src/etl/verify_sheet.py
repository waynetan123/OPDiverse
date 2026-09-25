"""50-item manual check that gold line numbers match the numbered function the model sees.

Each item shows render_numbered() output with gold lines marked, next to a diff built from
the same opcodes that produced the gold set, plus an independent `git diff` cross-check.
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path

from . import pinned

SHEET_SIZE = 50
PER_STRATUM = 5
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.M)


def _strata(row: dict, r: pinned.PatchResult) -> list[str]:
    edits = [op for op in r.opcodes if op[0] != "equal"]
    tags = {op[0] for op in edits}
    masked_v, _ = pinned.mask_comments(row["vuln_func"])
    masked_p, _ = pinned.mask_comments(row["patched_func"])
    checks = {
        "insert_only": tags == {"insert"},
        "delete_only": tags == {"delete"},
        "multi_hunk": len(edits) >= 2,
        "insert_at_start": any(op[0] == "insert" and op[1] == 0 for op in edits),
        "cr_in_source": row["raw_had_cr"],
        "over_200_lines": row["n_lines"] > 200,
        "comment_masked": r.mask_ok and (masked_v != row["vuln_func"] or masked_p != row["patched_func"]),
    }
    return [k for k, v in checks.items() if v]


def select(rows: list[dict]) -> list[tuple[dict, pinned.PatchResult, list[str]]]:
    """Up to PER_STRATUM items from each stratum, then fill to SHEET_SIZE, all by stable_rank."""
    ranked = sorted(rows, key=lambda row: pinned.stable_rank(row["cve_id"], "line-sheet"))
    info = {row["cve_id"]: (row, pinned.patch_line_set(row["vuln_func"], row["patched_func"])) for row in ranked}
    strata = {c: _strata(row, r) for c, (row, r) in info.items()}
    chosen: list[str] = []
    for name in ("insert_only", "delete_only", "multi_hunk", "insert_at_start", "cr_in_source", "over_200_lines", "comment_masked"):
        picked = [c for c in (row["cve_id"] for row in ranked) if name in strata[c] and c not in chosen][:PER_STRATUM]
        chosen += picked
    chosen += [row["cve_id"] for row in ranked if row["cve_id"] not in chosen][: SHEET_SIZE - len(chosen)]
    return [(*info[c], strata[c]) for c in chosen[:SHEET_SIZE]]


def git_gold(vuln: str, patched: str) -> set[int] | None:
    """Gold lines under `git diff -U0 --ignore-all-space --ignore-blank-lines` (Myers diff), restricted to
    code lines, with insertions moved to the preceding code line as our convention does."""
    code = [i for i, line in enumerate(pinned.split_lines(vuln), 1) if pinned.line_key(line)]
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.c", Path(tmp) / "b.c"
        a.write_text(vuln + "\n", encoding="utf-8")
        b.write_text(patched + "\n", encoding="utf-8")
        try:
            out = subprocess.run(
                ["git", "diff", "--no-index", "--no-color", "-U0", "--ignore-all-space", "--ignore-blank-lines", str(a), str(b)],
                capture_output=True, text=True, check=False,
            ).stdout
        except OSError:
            return None
    gold: set[int] = set()
    for m in _HUNK.finditer(out):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        if count:
            gold.update(range(start, start + count))
        elif code:  # insertion after line `start` (0 = before line 1)
            prior = [c for c in code if c <= start]
            gold.add(prior[-1] if prior else code[0])
    return gold & set(code)


def _diff_view(row: dict, r: pinned.PatchResult) -> str:
    v, p = pinned.split_lines(row["vuln_func"]), pinned.split_lines(row["patched_func"])
    out = []
    for tag, i1, i2, j1, j2 in r.opcodes:
        if tag == "equal":
            continue
        if tag == "insert":
            anchor = r.vuln_code[i1 - 1] if i1 > 0 else r.vuln_code[0]
            out.append(f"@@ insert {'after' if i1 > 0 else 'before'} vulnerable line {anchor} (gold {anchor}) @@")
        else:
            out.append(f"@@ {tag} vulnerable lines {r.vuln_code[i1]}-{r.vuln_code[i2 - 1]} @@")
        out += [f"-{r.vuln_code[i]:>5}: {v[r.vuln_code[i] - 1]}" for i in range(i1, i2)]
        out += [f"+{r.patched_code[j]:>5}: {p[r.patched_code[j] - 1]}" for j in range(j1, j2)]
    return "\n".join(out)


def write_sheet(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = select(rows)
    md = [
        "# Line-numbering verification sheet", "",
        f"{len(items)} items. For each, confirm that the marked lines in the numbered function are the lines the "
        "diff below says were changed (insertions are attributed to the preceding code line). "
        "Record a verdict per item in `line_sheet.csv`. Any systematic off-by-N stops the pipeline.", "",
        "`git` is an independent Myers diff restricted to code lines; it does not mask comments, so differences on "
        "comment lines, or on which of several identical lines was matched, are expected.", "",
    ]
    csv_rows = []
    for n, (row, r, strata) in enumerate(items, 1):
        gold = set(r.lines)
        git_code = git_gold(row["vuln_func"], row["patched_func"])
        agrees = "n/a" if git_code is None else "yes" if git_code == gold else "no"
        numbered = pinned.render_numbered(row["vuln_func"]).split("\n")
        marked = "\n".join(("▶ " if i in gold else "  ") + line for i, line in enumerate(numbered, 1))
        md += [
            f"## {n}. {row['cve_id']} — {row['cwe']}", "",
            f"- strata: {', '.join(strata) or '—'}; lines {row['n_lines']}, code lines {row['n_code_lines']}",
            f"- commit: {row['commit_url']}",
            f"- **gold: {', '.join(map(str, r.lines))}**; git: {', '.join(map(str, sorted(git_code))) if git_code is not None else 'n/a'} (agrees: {agrees})",
            "", "````c", marked, "````", "", "````diff", _diff_view(row, r), "````", "",
        ]
        csv_rows.append({
            "item": n, "cve_id": row["cve_id"], "strata": ";".join(strata),
            "gold_lines": " ".join(map(str, r.lines)),
            "git_code_lines": "" if git_code is None else " ".join(map(str, sorted(git_code))),
            "git_agrees": agrees, "verdict": "", "notes": "",
        })
    (out_dir / "line_sheet.md").write_text("\n".join(md), encoding="utf-8")
    with open(out_dir / "line_sheet.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ["item"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
