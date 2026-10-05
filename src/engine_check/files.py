"""File names for the engine-agreement check, all in data/engine_check.

vLLM passes use the runner's naming (generators.bank.files): vllm_a_requests.jsonl ->
vllm_a_generations.jsonl and vllm_a_run_meta.json. The HF reference reads vllm_a_requests.jsonl and
writes one file per shard: hf_shard{k}of{n}_generations.jsonl (with a .partial file while running)
and hf_shard{k}of{n}_run_meta.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from etl.paths import Paths

PASSES = ("a", "b")  # a: bank order, enters the decision; b: seeded shuffle, the vLLM noise reference
_SHARD = re.compile(r"hf_shard(\d+)of(\d+)_generations\.jsonl")


@dataclass(frozen=True)
class EngineFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths) -> EngineFiles:
        return cls(paths.engine_check)

    def vllm_requests(self, pass_: str) -> Path:
        if pass_ not in PASSES:
            raise ValueError(f"unknown pass {pass_!r}")
        return self.dir / f"vllm_{pass_}_requests.jsonl"

    def hf_generations(self, shard: int, shards: int) -> Path:
        return self.dir / f"hf_shard{shard}of{shards}_generations.jsonl"

    def hf_partial(self, shard: int, shards: int) -> Path:
        return self.dir / f"hf_shard{shard}of{shards}_generations.partial.jsonl"

    def hf_run_meta(self, shard: int, shards: int) -> Path:
        return self.dir / f"hf_shard{shard}of{shards}_run_meta.json"

    def hf_shard_files(self) -> list[Path]:
        """Every finished HF shard; exits unless they are exactly shards 0..n-1 of one n."""
        found = sorted((int(m.group(2)), int(m.group(1)), p) for p in self.dir.glob("hf_shard*_generations.jsonl")
                       if (m := _SHARD.fullmatch(p.name)))
        if not found:
            raise SystemExit(f"no HF generations in {self.dir}: run `python -m engine_check.run_hf` on the GPU machine")
        counts = {n for n, _, _ in found}
        if len(counts) != 1 or sorted(k for _, k, _ in found) != list(range(next(iter(counts)))):
            raise SystemExit(f"HF shard files are incomplete or mixed: {[p.name for *_, p in found]}")
        return [p for *_, p in found]

    @property
    def report_json(self) -> Path:
        return self.dir / "engine_report.json"

    @property
    def report_md(self) -> Path:
        return self.dir / "engine_report.md"

    @property
    def parser_review_json(self) -> Path:
        return self.dir / "parser_review.json"

    @property
    def parser_review_md(self) -> Path:
        return self.dir / "parser_review.md"
