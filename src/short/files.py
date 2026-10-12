"""File names for the one-day run, all in data/short.

    test_requests.jsonl                    every test item, evaluation requests
    pilot/{mem,cost}_<arm>_lr<lr>/         the pilots (train.launch --out-root data/short/pilot/cost)
    step_count.json                        TRAIN_STEPS from the cost pilot, and whether the fallback fired (frozen)
    queues.json                            the shared run queue, longest first, and its per-GPU forecast
    runs/<arm>_<config>_s<seed>/           one M2 run (short.job): run_meta.json, train_log.jsonl
        trainer/checkpoint-{step}/         the adapter at each checkpoint step
        final/merged/                      the last checkpoint merged, deleted after evaluation
        final/test_generations.jsonl, test_run_meta.json   its test evaluation (evaluate.run_vllm --out)
        final/merge_meta.json              the merge's record, kept when merged/ is deleted
    m2_scores.jsonl                        the analysis input: one row per (run, test item)
    score_summary.{json,md}                per run and type: score, parse, strict and cut-off rates, monitor events
    analysis/                              python -m analysis --out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etl import pinned
from etl.paths import Paths
from generators.bank.files import generations_for, run_meta_for


def run_name(arm: str, config: str, seed: int) -> str:
    """'grpo_m2-cvss_s1': an M2 run's directory name (the LR is the arm's one SWEEP_LR)."""
    return f"{arm}_{config}_s{seed}"


def lr(arm: str) -> float:
    """The arm's learning rate: pinned.SWEEP_LR, which the one-day run fixes without a sweep."""
    if pinned.SWEEP_LR is None or arm not in pinned.SWEEP_LR:
        raise SystemExit(f"pinned.SWEEP_LR has no LR for {arm}: pin 5e-5 for every arm (docs/OPDiverse_short.md) first")
    return pinned.SWEEP_LR[arm]


@dataclass(frozen=True)
class ShortFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths) -> ShortFiles:
        return cls(paths.short)

    @property
    def test_requests(self) -> Path:
        return self.dir / "test_requests.jsonl"

    @property
    def pilot(self) -> Path:
        return self.dir / "pilot"

    @property
    def runs(self) -> Path:
        return self.dir / "runs"

    def run_dir(self, arm: str, config: str, seed: int) -> Path:
        return self.runs / run_name(arm, config, seed)

    def adapter(self, arm: str, config: str, seed: int, step: int) -> Path:
        return self.run_dir(arm, config, seed) / "trainer" / f"checkpoint-{step}"

    def final(self, arm: str, config: str, seed: int) -> Path:
        return self.run_dir(arm, config, seed) / "final"

    def merged(self, arm: str, config: str, seed: int) -> Path:
        return self.final(arm, config, seed) / "merged"

    def generations(self, arm: str, config: str, seed: int) -> Path:
        return self.final(arm, config, seed) / generations_for(self.test_requests).name

    def eval_meta(self, arm: str, config: str, seed: int) -> Path:
        return self.final(arm, config, seed) / run_meta_for(self.test_requests).name

    def merge_meta(self, arm: str, config: str, seed: int) -> Path:
        """merged/merge_meta.json, kept in final/ when short.job deletes the merged weights."""
        return self.final(arm, config, seed) / "merge_meta.json"

    @property
    def step_count(self) -> Path:
        return self.dir / "step_count.json"

    @property
    def queues(self) -> Path:
        return self.dir / "queues.json"

    @property
    def scores(self) -> Path:
        return self.dir / "m2_scores.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.dir / "score_summary.json"

    @property
    def summary_md(self) -> Path:
        return self.dir / "score_summary.md"

    @property
    def analysis(self) -> Path:
        return self.dir / "analysis"
