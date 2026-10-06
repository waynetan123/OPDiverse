"""Step 10 CLI: the LR sweep at M1, seed 0. Training is `python -m train.run`, merging `python -m train.merge`,
evaluation `python -m evaluate.run_vllm --model ... --out ...`, all on the GPU machine.

    PYTHONPATH=src python -m sweep prepare   # -> data/sweep/{dev,checkpoint}_requests.jsonl
    PYTHONPATH=src python -m sweep scale     # every scale evaluation -> scale.json (frozen)
    PYTHONPATH=src python -m sweep select    # -> selection.json: chosen checkpoints and LRs so far, and what is missing
    PYTHONPATH=src python -m sweep report    # -> sweep_report.{json,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etl.paths import DEFAULT, Paths

from . import prepare, report, select

COMMANDS = ("prepare", "scale", "select", "report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())

    if args.command == "prepare":
        out = prepare.prepare(paths)
        print(f"seed {out['seed']}: {out['dev_items']:,} dev requests ({out['dev_cves']} CVEs), "
              f"{out['checkpoint_items']:,} checkpoint requests ({out['checkpoint_cves']} CVEs)")
    elif args.command == "scale":
        out = select.freeze_scale(paths)
        print("type scale: " + ", ".join(f"{t} {v:.4f}" for t, v in out["scale"].items())
              + f" (from {len(out['evaluations'])} evaluations); commit scale.json before selecting")
    elif args.command == "select":
        out = select.select(paths)
        for arm, a in out["arms"].items():
            print(f"{arm}: LR {a['lr']:.0e} at step {a['step']}" + (" (grid edge)" if a["edge"] else ""))
        for item in out["todo"]:
            print(f"missing: {item}")
    else:
        report.run(paths)
        print(f"report written to {paths.sweep / 'sweep_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
