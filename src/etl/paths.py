"""Input and output locations for step 1, all derived from one data directory."""

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


DEFAULT = Paths(ROOT / "data")
