"""Steps 1-2: build the fact table, then draw the frozen test window.

    PYTHONPATH=src python -m etl.build extract-nvd    # yearly NVD feeds -> data/cache/nvd_subset.jsonl.gz
    PYTHONPATH=src python -m etl.build build          # -> facts / candidates / drops / census.json
    PYTHONPATH=src python -m etl.build report         # -> census.md, parity.json, manifest.json
    PYTHONPATH=src python -m etl.build verify-sheet   # -> verification/line_sheet.{md,csv}
    PYTHONPATH=src python -m etl.build check          # invariants over facts.jsonl
    PYTHONPATH=src python -m etl.build all            # build, report, verify-sheet, check
    PYTHONPATH=src python -m etl.build test-window    # step 2 -> split.jsonl, test_window.{json,md}, baselines.json

`test-window` is not part of `all`: the window is drawn once and then frozen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import census, nvd, parity, pinned, split, verify_sheet
from .cwe_graph import CweGraph, load_cwe_graph
from .manifest import file_sha256, write_manifest
from .paths import DEFAULT, ROOT, Paths
from .primevul import INTEGRITY_REASONS, load_pairs
from .tokens import TokenCounter, pinned_counter

JOIN_REASONS = ("not_in_nvd", "rejected")
COMPLETENESS_REASONS = ("no_english_description", "bad_published")
CWE_REASONS = (
    "no_nvd_cwe", "placeholder_only", "malformed_cwe", "deprecated_no_replacement",
    "multi_cwe", "category", "view", "not_in_view_1000",
)
CVSS_REASONS = (
    "cna_only_v3", "v2_only", "v4_only", "no_cvss", "conflicting_nvd_vectors",
    "invalid_vector", "version_mismatch", "decomposed_mismatch",
)
PAIR_STAGES = (  # (stage, reason)
    ("empty_patch", "empty_patch"),
    ("patch_fraction", "patch_over_20pct"),
    ("rewrite_guard", "inserted_over_20pct"),
    ("token_cap", "over_token_cap"),
)


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes(rows))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# extract-nvd
# ---------------------------------------------------------------------------


def extract_nvd(paths: Paths) -> dict:
    needed = {p.cve_id for p in load_pairs(paths.primevul_paired) if "malformed_cve" not in p.failures}
    info = nvd.extract_subset(paths.nvd_feeds, needed, paths.nvd_subset)
    write_json(paths.nvd_feeds_info, info)
    return info


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _cve_facts(record: dict | None, graph: CweGraph) -> dict:
    if record is None:
        return {"join": "not_in_nvd"}
    facts = {"join": "rejected" if record.get("vulnStatus") == "Rejected" else "ok",
             "status": record.get("vulnStatus"), "last_modified": record.get("lastModified")}
    facts["description"] = nvd.english_description(record)
    facts["published"] = record.get("published", "")
    try:
        datetime.fromisoformat(facts["published"])
        published_ok = True
    except ValueError:
        published_ok = False
    facts["complete"] = ("no_english_description" if not facts["description"]
                         else "ok" if published_ok else "bad_published")
    has_block, values = pinned.nvd_cwe_values(record.get("weaknesses", []))
    facts["cwe_raw"] = values
    facts["cwe"], facts["cwe_reason"], facts["cwe_mapped_from"] = pinned.resolve_cwe(has_block, values, graph)
    facts["cvss"], facts["cvss_reason"] = pinned.select_nvd_cvss(record.get("metrics", {}))
    return facts


def _bucket(value: float, edges: list[tuple[float, str]]) -> str:
    for upper, label in edges:
        if value <= upper:
            return label
    return edges[-1][1]


def build(paths: Paths, tokens: TokenCounter) -> dict:
    graph = load_cwe_graph(paths.cwe_xml)
    graph.check_invariants()
    pairs = load_pairs(paths.primevul_paired)
    by_id = {p.pair_id: p for p in pairs}
    records = nvd.load_subset(paths.nvd_subset)
    ledger = census.Ledger(pairs)

    # Stage 1: pair integrity
    ledger.drop_pairs("pair_integrity", {p.pair_id: p.failures[0] for p in pairs if p.failures}, INTEGRITY_REASONS)

    # Stages 2-5: CVE-level
    cves = {c: _cve_facts(records.get(c), graph) for c in sorted({p.cve_id for p in pairs})}
    year_of = lambda c: cves[c].get("published", "")[:4] or "????"  # noqa: E731
    ledger.drop_cves("nvd_join", {c: f["join"] for c, f in cves.items() if f["join"] != "ok"}, JOIN_REASONS)
    ledger.snapshot("after_join", year_of)
    drop_alone_base = set(ledger.alive)
    cve_checks = {
        "field_completeness": ("complete", COMPLETENESS_REASONS),
        "cwe": ("cwe_reason", CWE_REASONS),
        "cvss_v3": ("cvss_reason", CVSS_REASONS),
    }
    for stage, (key, order) in cve_checks.items():
        ledger.drop_cves(stage, {c: f[key] for c, f in cves.items() if f.get(key, "ok") != "ok"}, order)
        if stage == "cwe":
            ledger.snapshot("pre_v3", year_of)
    ledger.snapshot("post_v3", year_of)
    cvss_kept = Counter()
    for c in ledger.alive_cves():
        cvss_kept[cves[c]["cvss"]["version"]] += 1
        cvss_kept["both_versions"] += cves[c]["cvss"]["both_versions"]

    # Pair features, for every pair that passed integrity
    base = [p for p in pairs if not p.failures]
    patch = {p.pair_id: pinned.patch_line_set(p.vuln, p.patched) for p in base}
    counts = tokens.count([p.vuln for p in base] + [p.patched for p in base] + [pinned.render_numbered(p.vuln) for p in base])
    n = len(base)
    tok = {p.pair_id: (counts[i], counts[n + i], counts[2 * n + i]) for i, p in enumerate(base)}
    names = {p.pair_id: pinned.extract_function_names(p.vuln) for p in base}
    norm = {p.pair_id: (pinned.norm_body_hash(p.vuln), pinned.norm_body_hash(p.patched)) for p in base}

    def pair_fails(pid: str) -> dict[str, bool]:
        r, t = patch[pid], tok[pid]
        return {
            "empty_patch": not r.lines,
            "patch_over_20pct": pinned.exceeds_fraction(len(r.lines), r.n_code_lines),
            "inserted_over_20pct": pinned.exceeds_fraction(r.n_inserted, r.n_code_lines),
            "over_token_cap": max(t[0], t[1]) > pinned.TOKEN_CAP,
        }

    fails = {pid: pair_fails(pid) for pid in patch}

    # Stages 6-9: pair-level
    token_cap_report = {}
    for stage, reason in PAIR_STAGES:
        if stage == "token_cap":
            entering = ledger.counts()
            caps = []
            for cap in (*pinned.TOKEN_CAP_REPORT, pinned.TOKEN_CAP):
                ok = {pid for pid in ledger.alive if max(tok[pid][0], tok[pid][1]) <= cap}
                caps.append({"cap": cap, "pairs": len(ok), "cves": len({ledger.cve_of[pid] for pid in ok})})
            token_cap_report = {"entering": entering, "caps": caps}
        ledger.drop_pairs(stage, {pid: reason for pid in ledger.alive if fails[pid][reason]}, (reason,))

    # Stage 10: one function per CVE
    chosen: dict[str, tuple[str, str, int]] = {}
    for cve, pids in ledger.alive_by_cve().items():
        desc = cves[cve]["description"]
        candidates = [pinned.Candidate(pid, norm[pid][0], pinned.name_in_description(names[pid], desc)) for pid in pids]
        pid, rule = pinned.choose_function(cve, candidates)
        chosen[cve] = (pid, rule, len(pids))
    selected = {pid for pid, _, _ in chosen.values()}
    ledger.drop_pairs("select_one_function", {pid: "not_selected" for pid in ledger.alive if pid not in selected},
                      ("not_selected",))
    ledger.snapshot("final", year_of)

    # Final rows
    facts = []
    for cve in sorted(chosen, key=lambda c: (cves[c]["published"], c)):
        pid, rule, n_candidates = chosen[cve]
        p, f, r = by_id[pid], cves[cve], patch[pid]
        cwe = f["cwe"][4:]
        facts.append({
            "cve_id": cve,
            "published": f["published"],
            "description": f["description"],
            "nvd_vuln_status": f["status"],
            "nvd_last_modified": f["last_modified"],
            "cwe": f["cwe"],
            "cwe_nvd_raw": f["cwe_raw"],
            "cwe_mapped_from": list(f["cwe_mapped_from"]),
            "cwe_parents": [f"CWE-{c}" for c in sorted(graph.parents(cwe), key=int)],
            "cwe_siblings": pinned.siblings(cwe, graph),
            "cvss_version": f["cvss"]["version"],
            "cvss_vector": f["cvss"]["vector"],
            "cvss_both_versions": f["cvss"]["both_versions"],
            "vuln_func": p.vuln,
            "patched_func": p.patched,
            "n_lines": r.n_lines,
            "n_code_lines": r.n_code_lines,
            "patch_lines": list(r.lines),
            "n_inserted": r.n_inserted,
            "n_deleted": r.n_deleted,
            "comment_mask_ok": r.mask_ok,
            "tokens_vuln": tok[pid][0],
            "tokens_patched": tok[pid][1],
            "tokens_vuln_numbered": tok[pid][2],
            "vuln_norm_hash": norm[pid][0],
            "patched_norm_hash": norm[pid][1],
            "func_names": list(names[pid]),
            "selection_rule": rule,
            "n_candidates": n_candidates,
            "pair_id": pid,
            "commit_id": p.commit_id,
            "commit_url": p.commit_url,
            "project": p.project,
            "file_name": p.file_name,
            "primevul_split": p.split,
            "primevul_idx": [p.idx_vuln, p.idx_patched],
            "raw_had_cr": "\r" in p.vuln_raw or "\r" in p.patched_raw,
        })
    assert len({row["cve_id"] for row in facts}) == len(facts), "cve_id must be unique"

    # Drops if each filter were applied alone, from the post-join base
    base_by_cve: dict[str, list[str]] = {}
    for pid in drop_alone_base:
        base_by_cve.setdefault(ledger.cve_of[pid], []).append(pid)
    drop_alone = []
    for stage, (key, _) in cve_checks.items():
        bad = [c for c in base_by_cve if cves[c].get(key, "ok") != "ok"]
        drop_alone.append({"filter": stage, "cves": len(bad), "pairs": sum(len(base_by_cve[c]) for c in bad)})
    for stage, reason in PAIR_STAGES:
        bad_pairs = {pid for pid in drop_alone_base if fails[pid][reason]}
        drop_alone.append({"filter": stage, "pairs": len(bad_pairs),
                           "cves": sum(all(pid in bad_pairs for pid in ps) for ps in base_by_cve.values())})

    # Candidates and drops
    candidate_rows = []
    for p in sorted(pairs, key=lambda p: (p.cve_id, p.pair_id)):
        pid = p.pair_id
        row = {
            "cve_id": p.cve_id, "pair_id": pid, "primevul_split": p.split, "primevul_line": p.line,
            "primevul_idx": [p.idx_vuln, p.idx_patched], "commit_id": p.commit_id,
            "fate": "selected" if pid in selected else ":".join(ledger.pair_fate[pid]),
            "integrity_failures": list(p.failures),
        }
        if pid in patch:
            r = patch[pid]
            row.update({
                "failed_pair_checks": [k for k, v in fails[pid].items() if v],
                "n_lines": r.n_lines, "n_code_lines": r.n_code_lines, "patch_size": len(r.lines),
                "n_inserted": r.n_inserted, "n_deleted": r.n_deleted, "comment_mask_ok": r.mask_ok,
                "tokens_vuln": tok[pid][0], "tokens_patched": tok[pid][1], "func_names": list(names[pid]),
                "name_in_description": pinned.name_in_description(names[pid], cves[p.cve_id].get("description", "")),
            })
        candidate_rows.append(row)

    drop_rows = []
    pairs_of: dict[str, list[str]] = {}
    for p in pairs:
        pairs_of.setdefault(p.cve_id, []).append(p.pair_id)
    for c in sorted(cves):
        if c in chosen:
            continue
        stage, reason = ledger.cve_fate[c]
        f = cves[c]
        cve_level = [f.get(k, "ok") for k in ("join", "complete", "cwe_reason", "cvss_reason")]
        pair_level = {
            failure
            for pid in pairs_of[c]
            for failure in (by_id[pid].failures or [k for k, v in fails.get(pid, {}).items() if v])
        }
        drop_rows.append({"cve_id": c, "stage": stage, "reason": reason,
                          "all_failures": sorted({x for x in cve_level if x != "ok"} | set(pair_level))})

    write_jsonl(paths.facts, facts)
    write_jsonl(paths.candidates, candidate_rows)
    write_jsonl(paths.drops, drop_rows)

    # Census
    finals = facts
    dates = sorted(datetime.fromisoformat(r["published"]) for r in finals)
    time_range = {}
    if dates:
        span = (dates[-1] - dates[0]).total_seconds() / (365.25 * 86400)
        p5, p95 = census.nearest_rank(dates, 0.05), census.nearest_rank(dates, 0.95)
        time_range = {
            "min": dates[0].isoformat(), "max": dates[-1].isoformat(), "span_years": span,
            "p5": p5.isoformat(), "p95": p95.isoformat(),
            "p5_p95_span_years": (p95 - p5).total_seconds() / (365.25 * 86400),
            "gate": "proceed (span >= 5 years)" if span >= 5
                    else "decide before any split: (a) accept a mild shift, or (b) extend forward",
        }
    size = len(finals)
    size_band = (">= 2,500: planned 75/10/15 split" if size >= 2500
                 else "< 2,500: fallback 70/15/15 split" if size >= 1500
                 else "< 1,500: reduce scope")
    by_year: dict[str, Counter] = {}
    for r in finals:
        by_year.setdefault(r["published"][:4], Counter())[r["cvss_version"]] += 1
    status = Counter(cves[c]["status"] for c in cves if cves[c]["join"] != "not_in_nvd")
    census_doc = {
        "pins": {
            "cwe_release": pinned.CWE_RELEASE, "token_cap": pinned.TOKEN_CAP,
            "patch_fraction": str(pinned.PATCH_FRACTION), "tokenizer_repo": pinned.TOKENIZER_REPO,
            "tokenizer_revision": pinned.TOKENIZER_REVISION, "tokenizer_sha256": tokens.sha256,
        },
        "stages": ledger.stages,
        "drop_alone_base": "nvd_join",
        "drop_alone": drop_alone,
        "year_histogram": ledger.snapshots,
        "cvss_kept": {"3.1": cvss_kept["3.1"], "3.0": cvss_kept["3.0"], "both_versions": cvss_kept["both_versions"]},
        "final_cvss_version_by_year": {y: dict(sorted(v.items())) for y, v in sorted(by_year.items())},
        "token_cap_report": token_cap_report,
        "final_tokens": {k: census.quantiles(r[k] for r in finals)
                         for k in ("tokens_vuln", "tokens_patched", "tokens_vuln_numbered")},
        "selection": {
            "cves_with_multiple_candidates": sum(n > 1 for _, _, n in chosen.values()),
            "rules": dict(sorted(Counter(rule for _, rule, _ in chosen.values()).items())),
        },
        "final": {"cves": size, "size_band": size_band, "time_range": time_range},
        "final_cwe_top": [list(x) for x in sorted(Counter(r["cwe"] for r in finals).items(),
                                                  key=lambda kv: (-kv[1], int(kv[0][4:])))[:25]],
        "final_sibling_counts": _ordered_buckets(
            (len(r["cwe_siblings"]) for r in finals),
            [(0, "0"), (1, "1"), (2, "2"), (5, "3-5"), (10, "6-10"), (float("inf"), ">10")]),
        "final_patch_size": _ordered_buckets(
            (len(r["patch_lines"]) for r in finals),
            [(1, "1"), (2, "2"), (3, "3"), (5, "4-5"), (10, "6-10"), (20, "11-20"), (float("inf"), ">20")]),
        "final_patch_fraction": _ordered_buckets(
            (len(r["patch_lines"]) / r["n_code_lines"] for r in finals),
            [(0.05, "<=5%"), (0.10, "5-10%"), (0.15, "10-15%"), (0.20, "15-20%")]),
        "final_mask_fallbacks": sum(not r["comment_mask_ok"] for r in finals),
        "nvd": {"records": len(records), "vuln_status": dict(sorted(status.items()))},
    }
    write_json(paths.census_json, census_doc)
    return census_doc


def _ordered_buckets(values, edges: list[tuple[float, str]]) -> list[list]:
    """[[label, count], ...] in edge order (a list, so sort_keys can't reorder it)."""
    counts = Counter(_bucket(v, edges) for v in values)
    return [[label, counts[label]] for _, label in edges]


# ---------------------------------------------------------------------------
# report / check
# ---------------------------------------------------------------------------


def report(paths: Paths) -> None:
    census_doc = json.loads(paths.census_json.read_text(encoding="utf-8"))
    paths.census_md.write_text(census.render_markdown(census_doc), encoding="utf-8")
    write_json(paths.parity_json, parity.parity(read_jsonl(paths.facts)))


def manifest(paths: Paths, tokenizer_path: Path) -> None:
    inputs = {f"primevul_{s}_paired": p for s, p in paths.primevul_paired.items()}
    inputs["cwe_xml"] = paths.cwe_xml
    inputs["tokenizer_json"] = tokenizer_path
    inputs["nvd_subset"] = paths.nvd_subset
    outputs = {name: getattr(paths, name) for name in (
        "facts", "candidates", "drops", "census_json", "census_md", "parity_json",
        "split", "test_window_json", "test_window_md", "baselines_json",
    )}
    for f in sorted(paths.verification.glob("*")) if paths.verification.exists() else []:
        outputs[f"verification/{f.name}"] = f
    feeds = json.loads(paths.nvd_feeds_info.read_text(encoding="utf-8")) if paths.nvd_feeds_info.exists() else None
    write_manifest(paths.manifest, ROOT, inputs, outputs, {
        "nvd_feeds": feeds,
        "tokenizer": {"repo": pinned.TOKENIZER_REPO, "revision": pinned.TOKENIZER_REVISION},
    })


def test_window(paths: Paths) -> tuple[dict, dict]:
    """Step 2. Deterministic in facts.jsonl; refuses to change an already-drawn window."""
    facts = read_jsonl(paths.facts)
    facts_sha = file_sha256(paths.facts)
    if paths.test_window_json.exists():
        drawn_from = json.loads(paths.test_window_json.read_text(encoding="utf-8"))["facts_sha256"]
        if drawn_from != facts_sha:
            raise SystemExit(f"facts.jsonl changed since the test window was drawn (from {drawn_from[:12]}…, now "
                             f"{facts_sha[:12]}…). The window is frozen: restore the facts, or delete the step-2 files "
                             "deliberately and record why.")
    tw = split.draw_test_window(facts)
    assignments = tw.pop("assignments")
    new_split = jsonl_bytes(assignments)
    if paths.split.exists() and paths.split.read_bytes() != new_split:
        raise SystemExit("split.jsonl exists and the redrawn window differs. The window is frozen: "
                         "delete the step-2 files deliberately and record why before redrawing.")
    pool_of = {a["cve_id"]: a["pool"] for a in assignments}
    graph = load_cwe_graph(paths.cwe_xml)
    base = split.baselines([r for r in facts if pool_of[r["cve_id"]] == "nontest"], graph)
    decision = base["exact_id_hierarchy"]["decision"]
    if decision != pinned.EXACT_ID_SCHEDULE:
        raise SystemExit(f"the constant-answer baseline selects the {decision} schedule but pinned.EXACT_ID_SCHEDULE "
                         f"is {pinned.EXACT_ID_SCHEDULE!r}; the computed number decides, so update the pin and the record")
    tw = {"facts_sha256": facts_sha, "test_fraction": str(pinned.TEST_FRACTION),
          "near_dup_jaccard": str(pinned.NEAR_DUP_JACCARD), "shingle_n": pinned.SHINGLE_N,
          "backbone_released": pinned.BACKBONE_RELEASED, **tw, "pools": split.pool_stats(facts, assignments)}
    paths.split.write_bytes(new_split)
    write_json(paths.test_window_json, tw)
    write_json(paths.baselines_json, base)
    paths.test_window_md.write_text(split.render_markdown(tw, base), encoding="utf-8")
    return tw, base


def check(paths: Paths) -> list[str]:
    """Row invariants over facts.jsonl. Returns a list of violations (empty = pass)."""
    graph = load_cwe_graph(paths.cwe_xml)
    problems = []
    rows = read_jsonl(paths.facts)
    if len({r["cve_id"] for r in rows}) != len(rows):
        problems.append("cve_id not unique")
    for r in rows:
        cve, lines, funcs = r["cve_id"], r["patch_lines"], pinned.split_lines(r["vuln_func"])
        if not lines:
            problems.append(f"{cve}: empty patch_lines")
        if lines != sorted(set(lines)) or any(not 1 <= ln <= r["n_lines"] for ln in lines):
            problems.append(f"{cve}: patch_lines not sorted/unique/in range")
        if any(not pinned.line_key(funcs[ln - 1]) for ln in lines if 1 <= ln <= len(funcs)):
            problems.append(f"{cve}: gold line on a blank line")
        if len(funcs) != r["n_lines"]:
            problems.append(f"{cve}: n_lines mismatch")
        if pinned.exceeds_fraction(len(lines), r["n_code_lines"]) or pinned.exceeds_fraction(r["n_inserted"], r["n_code_lines"]):
            problems.append(f"{cve}: violates 20% rule or rewrite guard")
        if max(r["tokens_vuln"], r["tokens_patched"]) > pinned.TOKEN_CAP:
            problems.append(f"{cve}: over token cap")
        if not graph.in_view(r["cwe"][4:]):
            problems.append(f"{cve}: {r['cwe']} not a live view-{pinned.CWE_VIEW} weakness")
        if r["cwe"] in r["cwe_siblings"]:
            problems.append(f"{cve}: CWE listed as its own sibling")
        parsed = pinned.parse_cvss_v3(r["cvss_vector"])
        if parsed is None or parsed[0] is not None or pinned.canonical_cvss(parsed[1]) != r["cvss_vector"]:
            problems.append(f"{cve}: CVSS vector not canonical")
        if r["cvss_version"] not in pinned.CVSS_VERSIONS:
            problems.append(f"{cve}: bad CVSS version")
        if pinned.patch_line_set(r["vuln_func"], r["patched_func"]).lines != tuple(lines):
            problems.append(f"{cve}: patch_lines not reproducible from stored functions")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m etl.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("extract-nvd", "build", "report", "verify-sheet", "check", "all", "test-window"))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--tokenizer", type=Path, help="override tokenizer.json (tests)")
    ap.add_argument("--unpinned-tokenizer", action="store_true", help="skip the tokenizer sha256 check (tests)")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())
    tokenizer_path = args.tokenizer or paths.tokenizer_json

    def counter() -> TokenCounter:
        return TokenCounter(tokenizer_path, None) if args.unpinned_tokenizer else pinned_counter(tokenizer_path)

    if args.command == "extract-nvd":
        info = extract_nvd(paths)
        print(f"NVD: {info['found']:,} of {info['needed']:,} CVEs found in {len(info['feeds'])} feeds "
              f"({len(info['missing'])} missing, {info['cross_feed_duplicates']} cross-feed duplicates)")
        return 0
    if args.command == "test-window":
        tw, base = test_window(paths)
        manifest(paths, tokenizer_path)
        h = base["exact_id_hierarchy"]
        print(f"test window: {tw['final']['test']:,} of {tw['n']:,} CVEs from {tw['boundary_day']} "
              f"({len(tw['moved'])} moved to non-test); exact-ID schedule: {h['decision']} "
              f"(symmetric best {h['symmetric_top'][0]['cwe']} {h['symmetric_top'][0]['mean']:.4f})")
        return 0
    if args.command in ("build", "all"):
        doc = build(paths, counter())
        print(f"facts: {doc['final']['cves']:,} CVEs ({doc['final']['size_band']}); "
              f"time-range gate: {doc['final']['time_range'].get('gate')}")
    if args.command in ("report", "all"):
        report(paths)
    if args.command in ("verify-sheet", "all"):
        verify_sheet.write_sheet(read_jsonl(paths.facts), paths.verification)
    if args.command in ("report", "verify-sheet", "all"):
        manifest(paths, tokenizer_path)
    if args.command in ("check", "all"):
        problems = check(paths)
        print(f"check: {len(problems)} problems")
        for p in problems[:50]:
            print("  " + p)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
