import gzip
import hashlib
import json

import pytest

from etl import nvd


def feed_bytes(records):
    return json.dumps({"format": "NVD_CVE", "version": "2.0", "timestamp": "2026-09-25T00:00:00.000",
                       "vulnerabilities": [{"cve": r} for r in records]}).encode()


def record(cve_id, last_modified="2020-01-01T00:00:00.000", **extra):
    return {"id": cve_id, "published": "2016-01-01T00:00:00.000", "lastModified": last_modified,
            "vulnStatus": "Analyzed", "descriptions": [{"lang": "en", "value": "d"}, {"lang": "es", "value": "e"}],
            "metrics": {}, "weaknesses": [], "configurations": [{"big": True}], "references": [], **extra}


def write_feed(path, records, meta=True, corrupt_meta=False):
    body = feed_bytes(records)
    path.write_bytes(gzip.compress(body))
    if meta:
        sha = hashlib.sha256(body).hexdigest().upper()
        if corrupt_meta:
            sha = "0" * 64
        path.with_name(path.name.replace(".json.gz", ".meta")).write_text(f"size:{len(body)}\nsha256:{sha}\n")


def test_extract_keeps_needed_and_latest(tmp_path):
    feeds = tmp_path / "feeds"
    feeds.mkdir()
    write_feed(feeds / "nvdcve-2.0-2016.json.gz", [record("CVE-2016-0001"), record("CVE-2016-0002"), record("CVE-2016-0003")])
    write_feed(feeds / "nvdcve-2.0-2017.json.gz", [record("CVE-2016-0002", "2021-01-01T00:00:00.000")], meta=False)
    (feeds / "nvdcve-2.0-modified.json").write_bytes(feed_bytes([record("CVE-2016-0001", "2099-01-01T00:00:00.000")]))
    out = tmp_path / "subset.jsonl.gz"
    info = nvd.extract_subset(feeds, {"CVE-2016-0001", "CVE-2016-0002", "CVE-2016-9999"}, out)
    assert info["found"] == 2 and info["missing"] == ["CVE-2016-9999"] and info["cross_feed_duplicates"] == 1
    assert [f["file"] for f in info["feeds"]] == ["nvdcve-2.0-2016.json.gz", "nvdcve-2.0-2017.json.gz"]
    assert [f["meta_verified"] for f in info["feeds"]] == [True, False]
    subset = nvd.load_subset(out)
    assert subset["CVE-2016-0002"]["lastModified"] == "2021-01-01T00:00:00.000"
    assert subset["CVE-2016-0001"]["lastModified"] == "2020-01-01T00:00:00.000"  # modified feed ignored
    assert "configurations" not in subset["CVE-2016-0001"]
    assert subset["CVE-2016-0001"]["descriptions"] == [{"lang": "en", "value": "d"}]
    first = out.read_bytes()
    nvd.extract_subset(feeds, {"CVE-2016-0001", "CVE-2016-0002", "CVE-2016-9999"}, out)
    assert out.read_bytes() == first


def test_meta_mismatch_raises(tmp_path):
    write_feed(tmp_path / "nvdcve-2.0-2016.json.gz", [record("CVE-2016-0001")], corrupt_meta=True)
    with pytest.raises(ValueError, match="does not match"):
        nvd.extract_subset(tmp_path, {"CVE-2016-0001"}, tmp_path / "out.jsonl.gz")


def test_english_description():
    assert nvd.english_description({"descriptions": [{"lang": "es", "value": "x"}, {"lang": "en", "value": " y "}]}) == "y"
    assert nvd.english_description({"descriptions": []}) == ""
