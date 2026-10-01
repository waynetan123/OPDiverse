"""Progress bars on stderr for the long generator steps. Stdlib only.

A bar is drawn only when stderr is a terminal, so tests, logs and redirected runs stay clean, and
it never touches an output file: builds stay byte-identical with or without it.

    for f in track(facts, "MCQ options"):
        ...

    with Bar("mcq", total=2276) as bar:
        bar.update(done, note="3 failed, will be re-sent")
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from typing import TextIO, TypeVar

T = TypeVar("T")
WIDTH = 30
MIN_INTERVAL = 0.1  # seconds between redraws


def clock(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Bar:
    def __init__(self, label: str, total: int, stream: TextIO | None = None, enabled: bool | None = None):
        self.label, self.total = label, max(0, total)
        self.stream = sys.stderr if stream is None else stream
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self.done = 0
        self.note = ""
        self.start = time.monotonic()
        self._last = float("-inf")

    def __enter__(self) -> Bar:
        self.refresh(force=True)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def advance(self, n: int = 1) -> None:
        self.update(self.done + n)

    def update(self, done: int, note: str | None = None) -> None:
        self.done = min(done, self.total)
        if note is not None:
            self.note = note
        self.refresh()

    def render(self) -> str:
        elapsed = time.monotonic() - self.start
        frac = self.done / self.total if self.total else 1.0
        filled = int(WIDTH * frac)
        eta = f", ~{clock(elapsed * (1 - frac) / frac)} left" if 0 < frac < 1 else ""
        note = f"  {self.note}" if self.note else ""
        return (f"{self.label} [{'#' * filled}{'-' * (WIDTH - filled)}] {frac:4.0%} "
                f"{self.done:,}/{self.total:,}  {clock(elapsed)}{eta}{note}")

    def refresh(self, force: bool = False) -> None:
        """Redraw; also keeps the elapsed clock moving while waiting on something else."""
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last < MIN_INTERVAL:
            return
        self._last = now
        self.stream.write("\r\x1b[2K" + self.render())
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.refresh(force=True)
            self.stream.write("\n")
            self.stream.flush()
            self.enabled = False


def track(items: Iterable[T], label: str, total: int | None = None, **kwargs) -> Iterator[T]:
    """Yield every item of `items`, advancing a bar after each. `total` defaults to len(items)."""
    with Bar(label, len(items) if total is None else total, **kwargs) as bar:
        for item in items:
            yield item
            bar.advance()
