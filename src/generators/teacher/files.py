"""File names for the external-model jobs: data/teacher, or data/teacher/pilot.

Each request file gets the runner's outputs next to it (generators.bank.files): <prefix>_generations.jsonl,
<prefix>_generations.partial.jsonl while running, and <prefix>_run_meta.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etl.paths import Paths


@dataclass(frozen=True)
class TeacherFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths, pilot: bool = False) -> TeacherFiles:
        return cls(paths.teacher_pilot if pilot else paths.teacher)

    def requests(self, job: str, retry: bool = False) -> Path:
        """job is 'trace' or 'dpo'; retry is the one regeneration."""
        return self.dir / f"{job}{'_retry' if retry else ''}_requests.jsonl"

    # Outputs, one row per non-test item, keyed by item_id
    @property
    def distill_external(self) -> Path:
        return self.dir / "distill_external.jsonl"

    @property
    def dpo(self) -> Path:
        return self.dir / "dpo.jsonl"

    @property
    def report_json(self) -> Path:
        return self.dir / "step6_report.json"

    @property
    def report_md(self) -> Path:
        return self.dir / "step6_report.md"

    @property
    def pilot_report_json(self) -> Path:
        return self.dir / "pilot_report.json"

    @property
    def pilot_report_md(self) -> Path:
        return self.dir / "pilot_report.md"
