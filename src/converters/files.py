"""File names for the converters, all in data/converters.

    seed{s}/{config}/{arm}.jsonl.gz   one training file (gzipped JSONL, deterministic bytes)
    manifest.json                     (arm, config, seed) -> file, dev types, checkpoint subsample
    converters_meta.json              input and output sha256s, the pins
    converters_report.{json,md}       row counts, tokens seen, flags, masks
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from etl.paths import Paths


@dataclass(frozen=True)
class ConverterFiles:
    dir: Path

    @classmethod
    def of(cls, paths: Paths) -> ConverterFiles:
        return cls(paths.converters)

    def rel(self, seed: int, config: str, arm: str) -> str:
        """The path of one training file relative to data/converters, as the manifest records it."""
        return f"seed{seed}/{config}/{arm}.jsonl.gz"

    def training(self, seed: int, config: str, arm: str) -> Path:
        return self.dir / self.rel(seed, config, arm)

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def meta(self) -> Path:
        return self.dir / "converters_meta.json"

    @property
    def report_json(self) -> Path:
        return self.dir / "converters_report.json"

    @property
    def report_md(self) -> Path:
        return self.dir / "converters_report.md"


def gzip_bytes(content: bytes) -> bytes:
    """Deterministic gzip: no file name, zeroed timestamp (as etl.nvd's cache)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(content)
    return buf.getvalue()


def read_content(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in read_content(path).decode("utf-8").splitlines()]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
