"""Input and output locations for every step, all derived from one data directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Paths:
    data: Path

    # Inputs
    @property
    def primevul_paired(self) -> dict[str, Path]:
        return {s: self.data / "primevul" / f"primevul_{s}_paired.jsonl" for s in SPLITS}

    @property
    def nvd_feeds(self) -> Path:
        return self.data / "nvd" / "feeds"

    @property
    def cwe_xml(self) -> Path:
        return self.data / "mitre_cwe" / "cwec_v4.20.xml"

    @property
    def tokenizer_json(self) -> Path:
        return self.data / "tokenizer" / "qwen2.5-7b-instruct" / "tokenizer.json"

    # Cache
    @property
    def nvd_subset(self) -> Path:
        return self.data / "cache" / "nvd_subset.jsonl.gz"

    @property
    def nvd_feeds_info(self) -> Path:
        return self.data / "cache" / "nvd_feeds.json"

    # Outputs
    @property
    def out(self) -> Path:
        return self.data / "combined_dataset"

    @property
    def facts(self) -> Path:
        return self.out / "facts.jsonl"

    @property
    def candidates(self) -> Path:
        return self.out / "candidates.jsonl"

    @property
    def drops(self) -> Path:
        return self.out / "drops.jsonl"

    @property
    def census_json(self) -> Path:
        return self.out / "census.json"

    @property
    def census_md(self) -> Path:
        return self.out / "census.md"

    @property
    def parity_json(self) -> Path:
        return self.out / "parity.json"

    @property
    def verification(self) -> Path:
        return self.out / "verification"

    @property
    def manifest(self) -> Path:
        return self.out / "manifest.json"

    # Step 2
    @property
    def split(self) -> Path:
        return self.out / "split.jsonl"

    @property
    def test_window_json(self) -> Path:
        return self.out / "test_window.json"

    @property
    def test_window_md(self) -> Path:
        return self.out / "test_window.md"

    @property
    def baselines_json(self) -> Path:
        return self.out / "baselines.json"

    # Step 3: contamination probe
    @property
    def probe(self) -> Path:
        return self.data / "probe"

    @property
    def probe_requests(self) -> Path:
        return self.probe / "requests.jsonl"

    @property
    def probe_generations(self) -> Path:
        return self.probe / "generations.jsonl"

    @property
    def probe_run_meta(self) -> Path:
        return self.probe / "run_meta.json"

    @property
    def probe_scores(self) -> Path:
        return self.probe / "scores.jsonl"

    @property
    def probe_recalled(self) -> Path:
        return self.probe / "recalled.jsonl"

    @property
    def probe_report_json(self) -> Path:
        return self.probe / "report.json"

    @property
    def probe_report_md(self) -> Path:
        return self.probe / "report.md"

    # Step 4: question bank (file names are in generators.bank.files, shared by the frozen and dry runs)
    @property
    def bank(self) -> Path:
        return self.data / "bank"

    @property
    def bank_pilot(self) -> Path:
        return self.bank / "pilot"

    @property
    def bank_dry(self) -> Path:
        """--dry-mcq output: MCQ options from the prior-matched draw only, never frozen."""
        return self.bank / "dry"

    # Step 5: frozen-model session (file names are in frozen_model.files)
    @property
    def frozen_model(self) -> Path:
        return self.data / "frozen_model"

    # Step 6: external-model jobs (file names are in generators.teacher.files)
    @property
    def teacher(self) -> Path:
        return self.data / "teacher"

    @property
    def teacher_pilot(self) -> Path:
        return self.teacher / "pilot"

    # Step 7: engine-agreement check (file names are in engine_check.files)
    @property
    def engine_check(self) -> Path:
        return self.data / "engine_check"


DEFAULT = Paths(ROOT / "data")
