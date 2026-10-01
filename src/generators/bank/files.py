"""File names for one bank directory: data/bank (frozen), data/bank/dry, or data/bank/pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etl.paths import Paths


@dataclass(frozen=True)
class BankFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths, dry: bool = False, pilot: bool = False) -> BankFiles:
        return cls(paths.bank_pilot if pilot else paths.bank_dry if dry else paths.bank)

    # External-model requests and results, first attempt and the one regeneration
    @property
    def mcq_requests(self) -> Path:
        return self.dir / "mcq_requests.jsonl"

    @property
    def mcq_generations(self) -> Path:
        return self.dir / "mcq_generations.jsonl"

    @property
    def mcq_run_meta(self) -> Path:
        return self.dir / "mcq_run_meta.json"

    @property
    def mcq_retry_requests(self) -> Path:
        return self.dir / "mcq_retry_requests.jsonl"

    @property
    def mcq_retry_generations(self) -> Path:
        return self.dir / "mcq_retry_generations.jsonl"

    @property
    def mcq_retry_run_meta(self) -> Path:
        return self.dir / "mcq_retry_run_meta.json"

    # Bank
    @property
    def mcq_decisions(self) -> Path:
        return self.dir / "mcq_decisions.jsonl"

    def bank(self, pool: str) -> Path:
        return self.dir / f"bank_{pool}.jsonl"

    @property
    def meta(self) -> Path:
        return self.dir / "bank_meta.json"

    @property
    def report_json(self) -> Path:
        return self.dir / "bank_report.json"

    @property
    def report_md(self) -> Path:
        return self.dir / "bank_report.md"

    @property
    def pilot_report_json(self) -> Path:
        return self.dir / "pilot_report.json"

    @property
    def pilot_report_md(self) -> Path:
        return self.dir / "pilot_report.md"


def generations_for(requests: Path) -> Path:
    """mcq_requests.jsonl -> mcq_generations.jsonl (the runner's default output)."""
    return requests.with_name(requests.name.replace("requests.jsonl", "generations.jsonl"))


def partial_for(requests: Path) -> Path:
    """mcq_requests.jsonl -> mcq_generations.partial.jsonl, the runner's checkpoint while it runs."""
    return requests.with_name(requests.name.replace("requests.jsonl", "generations.partial.jsonl"))


def run_meta_for(requests: Path) -> Path:
    return requests.with_name(requests.name.replace("requests.jsonl", "run_meta.json"))
