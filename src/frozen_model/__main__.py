"""Step 5 CLI (the vLLM runs themselves are `python -m frozen_model.run_vllm`, on the GPU machine).

    PYTHONPATH=src python -m frozen_model prepare        # -> audit_requests.jsonl, rationale_requests.jsonl
    PYTHONPATH=src python -m frozen_model prepare-retry  # invalid attempt-1 rationales -> rationale_retry_requests.jsonl
    PYTHONPATH=src python -m frozen_model audit          # audit generations -> GRPO caps and bands
    PYTHONPATH=src python -m frozen_model rationales     # -> distill_self.jsonl, substitution.json
    PYTHONPATH=src python -m frozen_model report         # -> step5_report.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths
from etl.tokens import TokenCounter, pinned_counter

from . import audit, distill, prepare, report

COMMANDS = ("prepare", "prepare-retry", "audit", "rationales", "report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m frozen_model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--tokenizer", type=Path, default=None, help="default: the pinned tokenizer.json")
    ap.add_argument("--unpinned-tokenizer", action="store_true", help="skip the tokenizer sha256 check (tests only)")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())

    def tokens() -> TokenCounter:
        path = args.tokenizer or paths.tokenizer_json
        return TokenCounter(path, None) if args.unpinned_tokenizer else pinned_counter(path)

    if args.command == "prepare":
        out = prepare.prepare(paths, tokens())
        print(f"wrote {out['audit']:,} audit and {out['rationale']:,} rationale requests; "
              f"longest rationale prompt {out['longest_rationale_prompt']:,} tokens, session max_model_len {out['max_model_len']:,}")
    elif args.command == "prepare-retry":
        rows = prepare.prepare_retry(paths, tokens())
        print(f"wrote {len(rows):,} regeneration requests")
    elif args.command == "audit":
        out = audit.run(paths, tokens())
        print(f"GRPO caps {out['grpo_caps']}; bands {out['bands']}")
    elif args.command == "rationales":
        out = distill.run(paths)
        print(f"substituted types: {out['substituted_types'] or 'none'}; "
              f"drop distill-self from the primary test: {out['drop_distill_self_from_primary_test']}")
    else:
        report.run(paths)
        print("step5_report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
