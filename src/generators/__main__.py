"""Generator CLI. Step 4, the question bank (the external runs are `python -m generators.external`):

    PYTHONPATH=src python -m generators bank prepare-mcq --pilot  # -> data/bank/pilot/mcq_requests.jsonl
    PYTHONPATH=src python -m generators bank pilot-report         # pilot generations -> pilot_report.{json,md}
    PYTHONPATH=src python -m generators bank prepare-mcq          # -> data/bank/mcq_requests.jsonl (all CVEs)
    PYTHONPATH=src python -m generators bank prepare-mcq --retry  # the one regeneration -> mcq_retry_requests.jsonl
    PYTHONPATH=src python -m generators bank build                # -> bank_{nontest,test}.jsonl, frozen
    PYTHONPATH=src python -m generators bank check                # invariants; exit 1 on any violation
    PYTHONPATH=src python -m generators bank report               # -> bank_report.{json,md}

--dry-mcq on build / check / report works in data/bank/dry with every MCQ from the prior-matched draw:
no external request, never frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths
from etl.tokens import TokenCounter, pinned_counter

from .bank import build, check, prepare, report

COMMANDS = ("prepare-mcq", "pilot-report", "build", "check", "report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m generators", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", choices=("bank",))
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--tokenizer", type=Path, default=None, help="default: the pinned tokenizer.json")
    ap.add_argument("--unpinned-tokenizer", action="store_true", help="skip the tokenizer sha256 check (tests only)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--dry-mcq", action="store_true")
    ap.add_argument("--skip-git", action="store_true", help="report: skip the whole-table git cross-check")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())

    if args.command == "prepare-mcq":
        files, rows = prepare.prepare_mcq(paths, pilot=args.pilot, retry=args.retry)
        print(f"wrote {len(rows)} MCQ requests to {files.mcq_retry_requests if args.retry else files.mcq_requests}")
    elif args.command == "pilot-report":
        out = report.pilot_report(paths)
        print(f"pilot: {out['statuses']}, shortcut {out['shortcut_model_first_attempt_draw_filled']:.3f}")
    elif args.command == "build":
        tok_path = args.tokenizer or paths.tokenizer_json
        tokens = TokenCounter(tok_path, None) if args.unpinned_tokenizer else pinned_counter(tok_path)
        meta = build.build(paths, tokens, dry=args.dry_mcq)
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


if __name__ == "__main__":
    raise SystemExit(main())
