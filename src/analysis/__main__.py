"""The frozen analysis CLI (run at step 19, after training; written and frozen at step 7).

    PYTHONPATH=src python -m analysis primary-test --scores m2_scores.jsonl --seeds 0 1 2 --out data/analysis [--flagged sft:line_loc ...]
    PYTHONPATH=src python -m analysis mde --scores m2_scores.jsonl --seeds 0 1 2 --out data/analysis

--scores rows are {"arm", "seed", "item_id", "metric", "dropped_type"}: test scores of M2 runs. Only rows
whose item type is the run's dropped type enter the test (that column is M2; the others are diagnostics).
--flagged marks floored or substituted (arm, type) cells; their columns are dropped in the sensitivity run
reported alongside. --arms runs the three-arm test if distill-self left the primary test. The scores must
cover exactly the test pool of --split and exactly --seeds.

Refuses to run unless src/analysis/{primary_test,mde}.py hash to pinned.PRIMARY_TEST_SHA256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from etl import pinned
from etl.build import read_jsonl, write_json
from etl.paths import DEFAULT

from . import mde, primary_test, source_sha256


def require_frozen() -> None:
    actual = source_sha256()
    if pinned.PRIMARY_TEST_SHA256 is None or actual != pinned.PRIMARY_TEST_SHA256:
        raise SystemExit(f"the analysis source hashes to {actual}, not pinned.PRIMARY_TEST_SHA256 "
                         f"{pinned.PRIMARY_TEST_SHA256}: the frozen test has been changed")


def m2_rows(path: Path) -> list[dict]:
    return [r for r in read_jsonl(path) if r["item_id"].split(":")[1] == r["dropped_type"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("primary-test", "mde"))
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True, help="the seeds every M2 cell was trained with")
    ap.add_argument("--split", type=Path, default=DEFAULT.split, help="split.jsonl: the test CVEs the scores must cover")
    ap.add_argument("--arms", nargs="+", default=list(pinned.PRIMARY_ARMS))
    ap.add_argument("--flagged", nargs="*", default=[], help="arm:type cells that were floored or substituted")
    ap.add_argument("--exclude-cves", type=Path, help="a jsonl of {cve_id} to leave out (the probe's sensitivity run)")
    args = ap.parse_args(argv)
    require_frozen()
    if not set(args.arms) <= set(pinned.PRIMARY_ARMS) or len(args.arms) < 3:
        raise SystemExit(f"--arms must be the primary arms, or three of them; got {args.arms}")
    flagged = tuple(tuple(f.split(":")) for f in args.flagged)
    exclude = frozenset(r["cve_id"] for r in read_jsonl(args.exclude_cves)) if args.exclude_cves else frozenset()
    test_cves = tuple(r["cve_id"] for r in read_jsonl(args.split) if r["pool"] == "test")
    scores = primary_test.collect(m2_rows(args.scores), tuple(args.arms), exclude_cves=exclude, cves=test_cves,
                                  seeds=tuple(args.seeds))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "primary-test":
        out = {"primary": primary_test.primary_test(scores, flagged_cells=flagged)}
        dropped = tuple(sorted({t for _, t in flagged}, key=pinned.BANK_TYPES.index))
        if dropped:
            out["sensitivity_leave_flagged_columns_out"] = primary_test.primary_test(scores, drop_types=dropped, flagged_cells=flagged)
        out["source_sha256"] = source_sha256()
        write_json(args.out / "primary_test.json", out)
        print(f"T = {out['primary']['T']:.6g}, p = {out['primary']['p']:.4f}" + "".join(
            f"; leave-flagged-columns-out p = {out['sensitivity_leave_flagged_columns_out']['p']:.4f}" for _ in [0] if dropped))
    else:
        out = {**mde.simulate(scores), "source_sha256": source_sha256()}
        write_json(args.out / "mde.json", out)
        print(f"MDE overall: {out['overall']['mde_points']} points; per type "
              + json.dumps({t: v["mde_points"] for t, v in out["per_type"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
