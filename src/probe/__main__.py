"""Step 3 CLI (the vLLM run itself is `python -m probe.run_vllm`, on the GPU machine).

    PYTHONPATH=src python -m probe prepare   # -> data/probe/requests.jsonl
    PYTHONPATH=src python -m probe score     # data/probe/generations.jsonl -> scores, recalled, report
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths

from . import prompts, score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m probe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("prepare", "score"))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())
    if args.command == "prepare":
        requests = prompts.write_requests(paths)
        print(f"wrote {len(requests)} requests ({len(requests) // 2} CVEs) to {paths.probe_requests}")
    else:
        report = score.run(paths)
        print(f"probe outcome: {report['outcome']} (see {score.versioned(paths.probe_report_md)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
