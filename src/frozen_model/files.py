"""File names for the frozen-model session, all in data/frozen_model.

Each request file gets its runner outputs next to it: <prefix>_generations.jsonl,
<prefix>_generations.partial.jsonl while running, and <prefix>_run_meta.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etl.paths import Paths


@dataclass(frozen=True)
class SessionFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths) -> SessionFiles:
        return cls(paths.frozen_model)

    # Requests (one engine load serves audit + rationale attempt 1; the retry is a second, smaller run)
    @property
    def audit_requests(self) -> Path:
        return self.dir / "audit_requests.jsonl"

    @property
    def rationale_requests(self) -> Path:
        return self.dir / "rationale_requests.jsonl"

    @property
    def rationale_retry_requests(self) -> Path:
        return self.dir / "rationale_retry_requests.jsonl"

    # Scored outputs
    @property
    def audit_scores(self) -> Path:
        return self.dir / "audit_scores.jsonl"

    @property
    def audit_report_json(self) -> Path:
        return self.dir / "audit_report.json"

    @property
    def audit_report_md(self) -> Path:
        return self.dir / "audit_report.md"

    @property
    def distill_self(self) -> Path:
        """The cached distill-self targets, one row per non-test item, keyed by item_id."""
        return self.dir / "distill_self.jsonl"

    @property
    def substitution(self) -> Path:
        return self.dir / "substitution.json"

    @property
    def rationale_report_json(self) -> Path:
        return self.dir / "rationale_report.json"

    @property
    def rationale_report_md(self) -> Path:
        return self.dir / "rationale_report.md"

    @property
    def report_json(self) -> Path:
        return self.dir / "step5_report.json"

    @property
    def report_md(self) -> Path:
        return self.dir / "step5_report.md"
