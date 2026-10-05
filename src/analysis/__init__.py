"""The pre-registered analysis (frozen at step 7): the primary test and the minimum detectable effect."""

from __future__ import annotations

import hashlib
from pathlib import Path

FROZEN = ("primary_test.py", "mde.py")


def source_sha256() -> str:
    """sha256 over the frozen modules' bytes; pinned.PRIMARY_TEST_SHA256 must equal it."""
    h = hashlib.sha256()
    for name in FROZEN:
        h.update(name.encode() + b"\0" + (Path(__file__).parent / name).read_bytes() + b"\0")
    return h.hexdigest()
