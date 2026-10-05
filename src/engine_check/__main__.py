"""Step 7 CLI. The GPU runs are `python -m evaluate.run_vllm` (vLLM passes) and `python -m engine_check.run_hf` (HF).

    PYTHONPATH=src python -m engine_check prepare        # -> data/engine_check/vllm_{a,b}_requests.jsonl
    PYTHONPATH=src python -m engine_check compare        # all three runs -> engine_report.{json,md}, the decision
    PYTHONPATH=src python -m engine_check parser-review  # base-model non-test replies -> parser_review.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths
from etl.tokens import TokenCounter, pinned_counter

from . import compare, parser_review, prepare

COMMANDS = ("prepare", "compare", "parser-review")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m engine_check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--tokenizer", type=Path, default=None, help="default: the pinned tokenizer.json")
    ap.add_argument("--unpinned-tokenizer", action="store_true", help="skip the tokenizer sha256 check (tests only)")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())

    if args.command == "prepare":
        out = prepare.prepare(paths)
        print(f"wrote {out['requests']:,} requests ({out['cves']} CVEs from a late window of {out['late_window']} "
              f"non-test CVEs, {out['window_from']} → {out['window_to']}) for vLLM passes A and B")
    elif args.command == "compare":
        path = args.tokenizer or paths.tokenizer_json
        tokens = TokenCounter(path, None) if args.unpinned_tokenizer else pinned_counter(path)
        out = compare.run(paths, tokens)
        print(f"engine agreement: {out['outcome']}" + (f" on {', '.join(out['material_types'])}" if out["material_types"] else "")
              + f" (see {paths.engine_check / 'engine_report.md'})")
    else:
        out = parser_review.run(paths)
        print(f"parser review written ({len(out['listed']):,} replies listed); see {paths.engine_check / 'parser_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
