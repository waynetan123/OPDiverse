"""Extract the CVEs we need from NVD's yearly JSON 2.0 feeds into a small, sorted cache."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

_YEARLY_FEED = re.compile(r"nvdcve-2\.0-(\d{4})\.json(\.gz)?")
_KEEP = ("id", "published", "lastModified", "vulnStatus", "descriptions", "metrics", "weaknesses")


def yearly_feeds(feeds_dir: Path) -> list[Path]:
    """Yearly feed files only; the rolling 'modified'/'recent' feeds would duplicate records."""
    feeds = sorted(p for p in feeds_dir.iterdir() if _YEARLY_FEED.fullmatch(p.name))
    if not feeds:
        raise FileNotFoundError(f"no nvdcve-2.0-YYYY.json[.gz] feeds in {feeds_dir}")
    return feeds


def _meta_sha256(feed: Path) -> str | None:
    """sha256 of the uncompressed JSON, from NVD's companion .meta file if present."""
    year = _YEARLY_FEED.fullmatch(feed.name).group(1)
    meta = feed.with_name(f"nvdcve-2.0-{year}.meta")
    if not meta.exists():
        return None
    for line in meta.read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "sha256":
            return value.strip().lower()
    return None


def _read_feed(feed: Path) -> tuple[dict, dict]:
    raw = feed.read_bytes()
    body = gzip.decompress(raw) if feed.suffix == ".gz" else raw
    body_sha = hashlib.sha256(body).hexdigest()
    expected = _meta_sha256(feed)
    if expected is not None and expected != body_sha:
        raise ValueError(f"{feed.name}: sha256 {body_sha} does not match .meta {expected}")
    data = json.loads(body)
    info = {
        "file": feed.name,
        "sha256_file": hashlib.sha256(raw).hexdigest(),
        "sha256_json": body_sha,
        "meta_verified": expected is not None,
        "timestamp": data.get("timestamp"),
        "n_vulnerabilities": len(data.get("vulnerabilities", [])),
    }
    return data, info


def extract_subset(feeds_dir: Path, needed: set[str], out_path: Path) -> dict:
    """Keep only the needed CVE IDs, one feed in memory at a time.

    A CVE present in two feeds keeps the record with the later lastModified. Output is
    gzip'd JSONL sorted by CVE ID, with a zeroed gzip timestamp so reruns are byte-identical.
    """
    records: dict[str, dict] = {}
    feeds_info = []
    duplicates = 0
    for feed in yearly_feeds(feeds_dir):
        data, info = _read_feed(feed)
        feeds_info.append(info)
        for vuln in data.get("vulnerabilities", []):
            cve = vuln["cve"]
            if cve["id"] not in needed:
                continue
            record = {k: cve[k] for k in _KEEP if k in cve}
            record["descriptions"] = [d for d in record.get("descriptions", []) if d.get("lang") == "en"]
            prior = records.get(cve["id"])
            if prior is not None:
                duplicates += 1
                if prior["lastModified"] >= record["lastModified"]:
                    continue
            records[cve["id"]] = record
        del data

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        for cve_id in sorted(records):
            gz.write((json.dumps(records[cve_id], sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    out_path.write_bytes(buf.getvalue())
    return {
        "feeds": feeds_info,
        "needed": len(needed),
        "found": len(records),
        "missing": sorted(needed - records.keys()),
        "cross_feed_duplicates": duplicates,
    }


def load_subset(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def english_description(record: dict) -> str:
    for d in record.get("descriptions", []):
        if d.get("lang") == "en" and d.get("value", "").strip():
            return d["value"].strip()
    return ""
