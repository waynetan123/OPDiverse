"""Provenance: hashes of every input and output, versions and git state."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    """sha256 over every file under `root` (relative path and content), in sorted path order: a model directory's
    identity."""
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h.update(str(p.relative_to(root)).encode("utf-8") + b"\0" + file_sha256(p).encode() + b"\0")
    return h.hexdigest()


def git_state(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        # Untracked files (e.g. fresh outputs) don't make the code state dirty.
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_manifest(path: Path, root: Path, inputs: dict[str, Path], outputs: dict[str, Path], extra: dict) -> None:
    """The only step-1 file carrying a run timestamp, so every other output is reproducible byte for byte."""

    def entry(p: Path) -> dict:
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        return {"path": rel, "sha256": file_sha256(p), "bytes": p.stat().st_size}

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(root),
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in ("tokenizers",)},
        "inputs": {k: entry(p) for k, p in sorted(inputs.items())},
        "outputs": {k: entry(p) for k, p in sorted(outputs.items()) if p.exists()},
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
