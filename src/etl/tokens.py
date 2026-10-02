"""Token counts under the pinned backbone tokenizer (tokenizer.json only; no transformers/torch)."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from . import pinned
from .manifest import file_sha256


class TokenCounter:
    def __init__(self, path: Path, expected_sha256: str | None):
        """expected_sha256=None skips verification; only tests should do that."""
        self.path = path
        self.sha256 = file_sha256(path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError(f"{path}: sha256 {self.sha256} != pinned {expected_sha256}")
        self._tokenizer = Tokenizer.from_file(str(path))

    def count(self, texts: list[str]) -> list[int]:
        return [len(e.ids) for e in self._tokenizer.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text, special tokens dropped (as vLLM's skip_special_tokens)."""
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def token_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)


def pinned_counter(path: Path) -> TokenCounter:
    if pinned.TOKENIZER_REVISION is None or pinned.TOKENIZER_SHA256 is None:
        raise SystemExit(
            f"Pin the tokenizer first: download tokenizer.json for {pinned.TOKENIZER_REPO} at a fixed commit "
            f"to {path}, then set TOKENIZER_REVISION and TOKENIZER_SHA256 in src/etl/pinned.py."
        )
    return TokenCounter(path, pinned.TOKENIZER_SHA256)
