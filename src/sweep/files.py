"""File names for the LR sweep, all in data/sweep.

    dev_requests.jsonl                 seed 0's dev items (full dev), evaluation requests
    checkpoint_requests.jsonl          seed 0's checkpoint subsample, a subset of dev
    runs/{arm}_lr{lr}/                 one training run (train.run): run_meta.json, train_log.jsonl
        trainer/checkpoint-{step}/     the adapter (and optimizer state) at each of the CHECKPOINTS steps
        step{step}/merged/             the merged bf16 checkpoint, deleted after evaluation
        step{step}/{checkpoint,dev}_generations.jsonl, *_run_meta.json   evaluations (evaluate.run_vllm --out)
    backbone/dev_generations.jsonl     the raw backbone on full dev (reference only)
    scale.json                         the frozen type scale
    selection.json                     chosen checkpoints and winning LRs
    sweep_report.{json,md}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etl.paths import Paths
from generators.bank.files import generations_for, run_meta_for
from train.config import run_name

EVALS = ("checkpoint", "dev")


@dataclass(frozen=True)
class SweepFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths) -> SweepFiles:
        return cls(paths.sweep)

    def requests(self, which: str) -> Path:
        if which not in EVALS:
            raise ValueError(f"unknown evaluation {which!r}")
        return self.dir / f"{which}_requests.jsonl"

    @property
    def runs(self) -> Path:
        return self.dir / "runs"

    def run_dir(self, arm: str, lr: float) -> Path:
        return self.runs / run_name(arm, lr)

    def adapter(self, arm: str, lr: float, step: int) -> Path:
        return self.run_dir(arm, lr) / "trainer" / f"checkpoint-{step}"

    def step_dir(self, arm: str, lr: float, step: int) -> Path:
        return self.run_dir(arm, lr) / f"step{step}"

    def merged(self, arm: str, lr: float, step: int) -> Path:
        return self.step_dir(arm, lr, step) / "merged"

    def generations(self, arm: str, lr: float, step: int, which: str) -> Path:
        return self.step_dir(arm, lr, step) / generations_for(self.requests(which)).name

    def eval_meta(self, arm: str, lr: float, step: int, which: str) -> Path:
        return self.step_dir(arm, lr, step) / run_meta_for(self.requests(which)).name

    @property
    def backbone(self) -> Path:
        return self.dir / "backbone"

    @property
    def backbone_generations(self) -> Path:
        return self.backbone / generations_for(self.requests("dev")).name

    @property
    def scale(self) -> Path:
        return self.dir / "scale.json"

    @property
    def selection(self) -> Path:
        return self.dir / "selection.json"

    @property
    def report_json(self) -> Path:
        return self.dir / "sweep_report.json"

    @property
    def report_md(self) -> Path:
        return self.dir / "sweep_report.md"
