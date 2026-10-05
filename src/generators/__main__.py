"""Generator CLI. Steps 4 and 6 (the external runs themselves are `python -m generators.external`).

Step 4, the question bank:

    PYTHONPATH=src python -m generators bank prepare-mcq --pilot  # -> data/bank/pilot/mcq_requests.jsonl
    PYTHONPATH=src python -m generators bank pilot-report         # pilot generations -> pilot_report.{json,md}
    PYTHONPATH=src python -m generators bank prepare-mcq          # -> data/bank/mcq_requests.jsonl (all CVEs)
    PYTHONPATH=src python -m generators bank prepare-mcq --retry  # the one regeneration -> mcq_retry_requests.jsonl
    PYTHONPATH=src python -m generators bank build                # -> bank_{nontest,test}.jsonl, frozen
    PYTHONPATH=src python -m generators bank check                # invariants; exit 1 on any violation
    PYTHONPATH=src python -m generators bank report               # -> bank_report.{json,md}

--dry-mcq on build / check / report works in data/bank/dry with every MCQ from the prior-matched draw:
no external request, never frozen.

Step 6, the DPO rejected answers over the frozen non-test bank (data/teacher):

    PYTHONPATH=src python -m generators teacher prepare --pilot  # -> data/teacher/pilot/dpo_requests.jsonl
    PYTHONPATH=src python -m generators teacher pilot-report     # pilot generations -> pilot_report.{json,md}
    PYTHONPATH=src python -m generators teacher prepare          # -> dpo_requests.jsonl
    PYTHONPATH=src python -m generators teacher prepare-retry    # invalid first proposals -> dpo_retry_requests.jsonl
    PYTHONPATH=src python -m generators teacher build            # -> dpo.jsonl, step6_report.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths
from etl.tokens import TokenCounter, pinned_counter

from .bank import build, check, prepare, report
from .teacher import build as teacher_build
from .teacher import prepare as teacher_prepare
from .teacher import report as teacher_report

COMMANDS = {
    "bank": ("prepare-mcq", "pilot-report", "build", "check", "report"),
    "teacher": ("prepare", "pilot-report", "prepare-retry", "build"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m generators", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", choices=tuple(COMMANDS))
    ap.add_argument("command", choices=sorted({c for cs in COMMANDS.values() for c in cs}))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--tokenizer", type=Path, default=None, help="default: the pinned tokenizer.json")
    ap.add_argument("--unpinned-tokenizer", action="store_true", help="skip the tokenizer sha256 check (tests only)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--dry-mcq", action="store_true")
    ap.add_argument("--skip-git", action="store_true", help="report: skip the whole-table git cross-check")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())
    if args.command not in COMMANDS[args.group]:
        ap.error(f"{args.group} has no command {args.command!r}; choose from {COMMANDS[args.group]}")

    def tokens() -> TokenCounter:
        tok_path = args.tokenizer or paths.tokenizer_json
        return TokenCounter(tok_path, None) if args.unpinned_tokenizer else pinned_counter(tok_path)

    if args.group == "teacher":
        return teacher(args, paths)
    if args.command == "prepare-mcq":
        files, rows = prepare.prepare_mcq(paths, pilot=args.pilot, retry=args.retry)
        print(f"wrote {len(rows)} MCQ requests to {files.mcq_retry_requests if args.retry else files.mcq_requests}")
    elif args.command == "pilot-report":
        out = report.pilot_report(paths)
        print(f"pilot: {out['statuses']}, shortcut {out['shortcut_model_first_attempt_draw_filled']:.3f}")
    elif args.command == "build":
        meta = build.build(paths, tokens(), dry=args.dry_mcq)
        print(f"built {meta['counts']}; MCQ guard {meta['mcq_guard']}")
    elif args.command == "check":
        problems = check.check(paths, dry=args.dry_mcq)
        for p in problems[:50]:
            print(p)
        print(f"{len(problems)} violations")
        return 1 if problems else 0
    else:
        out = report.report(paths, dry=args.dry_mcq, git=not args.skip_git)
        print(f"report written; longest prompt {out['tokens']['max_prompt_tokens']:,} tokens")
    return 0


def teacher(args, paths: Paths) -> int:
    if args.command == "prepare":
        n = teacher_prepare.prepare(paths, pilot=args.pilot)
        print(f"wrote {n:,} DPO requests" + (" (pilot)" if args.pilot else ""))
    elif args.command == "pilot-report":
        out = teacher_report.pilot_report(paths)
        print(f"pilot: full run estimated at ${out['cost_usd']['full_run_estimate']:,.2f}; see pilot_report.md")
    elif args.command == "prepare-retry":
        n = teacher_prepare.prepare_retry(paths)
        print(f"{n:,} DPO regenerations" + ("" if n else "; no retry file written, go straight to build"))
    else:
        out = teacher_report.write(paths, teacher_build.build(paths))
        print(f"built dpo.jsonl for {out['n_items']:,} items; total cost ${out['total_cost_usd']:,.2f}; see step6_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
