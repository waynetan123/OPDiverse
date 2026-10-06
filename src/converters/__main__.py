"""Step 9: the converters. One training file per arm, per matrix configuration, per seed, selected from the
frozen bank, the seed partitions and the cached step-5 and step-6 files. No GPU, no generation.

    PYTHONPATH=src python -m converters build    # -> data/converters/seed{s}/{config}/{arm}.jsonl.gz, manifest, meta
    PYTHONPATH=src python -m converters check    # invariants over every file; exit 1 on any violation
    PYTHONPATH=src python -m converters report   # -> converters_report.{json,md} (needs the pinned tokenizer)

`build` is frozen once written: a rebuild that would change any file's content is refused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths
from etl.tokens import TokenCounter, pinned_counter

from . import build, check, masks, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m converters", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "check", "report"))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--tokenizer", type=Path, default=None, help="default: the pinned tokenizer.json")
    ap.add_argument("--unpinned-tokenizer", action="store_true", help="skip the tokenizer sha256 check (tests only)")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())

    if args.command == "build":
        meta = build.build(paths)
        print(f"wrote {len(meta['outputs']):,} training files over {len(masks.CONFIGS)} configurations; "
              f"see {paths.converters / 'manifest.json'}")
    elif args.command == "check":
        problems = check.check(paths)
        for p in problems[:50]:
            print(p)
        print(f"{len(problems)} violations")
        return 1 if problems else 0
    else:
        tok_path = args.tokenizer or paths.tokenizer_json
        tokens = TokenCounter(tok_path, None) if args.unpinned_tokenizer else pinned_counter(tok_path)
        out = report.run(paths, tokens)
        print("report written; longest sequence per arm: "
              + ", ".join(f"{a} {n:,}" for a, n in out["max_sequence_tokens"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
