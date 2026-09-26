"""End to end on a small fixture: every drop path once, and byte-identical reruns across hash seeds."""

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from etl import build, pinned
from etl.paths import DEFAULT, Paths

ROOT = Path(__file__).resolve().parents[1]
NVD = pinned.NVD_SOURCE
V31 = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
V30 = "CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N"

ALPHA = """static int alpha_parse(char *buf, int len)
{
    char tmp[16];
    int i;
    for (i = 0; i < len; i++)
        tmp[i] = buf[i];
    return tmp[0];
}"""
BETA = """int beta_read(struct ctx *c, size_t n)
{
    memcpy(c->buf, c->src, n);
    c->len = n;
    return 0;
}"""
HDR = """/* read a header */
static int hdr_read(const unsigned char *p, int n)
{
    int v = p[n]; // unchecked
    return v;
}"""
HDR_FIXED = """/* read a header safely */
static int hdr_read(const unsigned char *p, int n)
{
    if (n < 0) return -1;
    int v = p[n]; // now checked
    return v;
}"""
CLEANUP = """void cleanup(struct s *p)
{
    free(p->a);
    p->a = NULL;
    free(p->a);
}"""
TEN = """int g(int a)
{
    int r = 0;
    r += a;
    r *= 2;
    r -= 1;
    r /= 3;
    r ^= a;
    return r;
}"""
SMALL = "int f(void)\n{\n    return 0;\n}"
SMALL_FIXED = "int f(void)\n{\n    return 1;\n}"
BIG = "int big(void)\n{\n" + "    x;\n" * 6000 + "    return 0;\n}"


def row(cve, commit, target, func, idx):
    return {"idx": idx, "project": "p", "commit_id": commit, "project_url": "u", "commit_url": f"https://x/{commit}",
            "commit_message": "m", "target": target, "func": func, "func_hash": idx, "file_name": "None",
            "file_hash": None, "cwe": ["CWE-20"], "cve": cve, "cve_desc": "d", "nvd_url": "n"}


PAIRS = [  # (cve, commit, vulnerable, patched); patched commit/cve override for the cross pair
    ("CVE-2016-0001", "c01", ALPHA, ALPHA.replace("    int i;\n", "    int i;\n    if (len > 16) return -1;\n")),
    ("CVE-2016-0001", "c01", BETA, BETA.replace("c->src, n);", "c->src, MIN(n, sizeof(c->buf)));")),
    ("CVE-2017-0002", "c02", HDR, HDR_FIXED),
    ("CVE-2018-0003", "c03", SMALL, SMALL_FIXED),
    ("CVE-2018-0004", "c04", SMALL, SMALL_FIXED.replace("return 1", "return 2")),
    ("CVE-2019-0005", "c05", SMALL, SMALL.replace("    return", "\treturn  ") + "\n\n"),
    ("CVE-2019-0006", "c06", SMALL, SMALL_FIXED.replace("return 1", "return 3")),
    ("CVE-2020-0007", "c07", BIG, BIG.replace("return 0", "return 1")),
    ("CVE-2020-0008", "c08", SMALL, SMALL_FIXED.replace("return 1", "return 4")),
    ("CVE-2021-0009", "c09", SMALL, SMALL_FIXED.replace("return 1", "return 5")),
    ("CVE-2021-0010", "c10", CLEANUP, CLEANUP.replace("    free(p->a);\n}", "}")),
    ("CVE-2022-0011", "c11", TEN, TEN.replace("    r *= 2;\n", "    r *= 2;\n" + "".join(f"    check{i}();\n" for i in range(5)))),
    ("CVE-2022-0012", "c12", TEN, TEN.replace("r += a;", "r += a + 1;").replace("r *= 2;", "r *= 3;").replace("r -= 1;", "r -= 2;")),
]


def metric(source, version, vector):
    return {"source": source, "type": "Primary", "cvssData": {"version": version, "vectorString": vector}}


def weakness(source, *values):
    return {"source": source, "type": "Primary", "description": [{"lang": "en", "value": v} for v in values]}


def nvd_record(cve, published, cwes=("CWE-787",), v31=V31, v30=None, cna31=None, status="Analyzed",
               desc="A flaw.", last_modified="2024-01-01T00:00:00.000"):
    metrics = {}
    if v31 or cna31:
        metrics["cvssMetricV31"] = ([metric(NVD, "3.1", v31)] if v31 else []) + ([metric("cna@x", "3.1", cna31)] if cna31 else [])
    if v30:
        metrics["cvssMetricV30"] = [metric(NVD, "3.0", v30)]
    return {"id": cve, "published": published, "lastModified": last_modified, "vulnStatus": status,
            "descriptions": [{"lang": "en", "value": desc}], "metrics": metrics,
            "weaknesses": [weakness(NVD, *cwes), weakness("cna@x", "CWE-20")]}


RECORDS_2016 = [
    nvd_record("CVE-2016-0001", "2016-03-01T10:00:00.000", desc="Overflow in beta_read in libfoo.", cna31=V30.replace("3.0", "3.1")),
    nvd_record("CVE-2017-0002", "2017-06-01T10:00:00.000", cwes=("CWE-125", "NVD-CWE-Other"), v30=V30),
]
RECORDS_2018 = [
    nvd_record("CVE-2017-0002", "2017-06-01T10:00:00.000", cwes=("CWE-20",), last_modified="2019-01-01T00:00:00.000"),
    nvd_record("CVE-2018-0003", "2018-01-01T00:00:00.000", v31=None, cna31=V31),
    nvd_record("CVE-2018-0004", "2018-02-01T00:00:00.000", cwes=("CWE-399",)),
    nvd_record("CVE-2019-0005", "2019-01-01T00:00:00.000"),
    nvd_record("CVE-2019-0006", "2019-02-01T00:00:00.000"),
    nvd_record("CVE-2020-0007", "2020-01-01T00:00:00.000"),
    nvd_record("CVE-2021-0009", "2021-01-01T00:00:00.000", status="Rejected"),
    nvd_record("CVE-2021-0010", "2021-12-01T00:00:00.000", cwes=("CWE-1187",), v31=None, v30=V30),
    nvd_record("CVE-2022-0011", "2022-01-01T00:00:00.000"),
    nvd_record("CVE-2022-0012", "2022-02-01T00:00:00.000"),
]


def make_fixture(data: Path) -> Path:
    rows = []
    for k, (cve, commit, vuln, patched) in enumerate(PAIRS):
        rows.append(row(cve, commit, 1, vuln, 2 * k))
        other = ("CVE-2019-9999", "c99") if cve == "CVE-2019-0006" else (cve, commit)
        rows.append(row(other[0], other[1], 0, patched, 2 * k + 1))
    (data / "primevul").mkdir(parents=True)
    splits = {"train": rows[:16], "valid": rows[16:20], "test": rows[20:]}
    for split, part in splits.items():
        (data / "primevul" / f"primevul_{split}_paired.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    feeds = data / "nvd" / "feeds"
    feeds.mkdir(parents=True)
    for year, records in (("2016", RECORDS_2016), ("2018", RECORDS_2018)):
        body = json.dumps({"timestamp": "2026-09-25T00:00:00", "vulnerabilities": [{"cve": r} for r in records]}).encode()
        (feeds / f"nvdcve-2.0-{year}.json.gz").write_bytes(gzip.compress(body, mtime=0))
    (data / "mitre_cwe").mkdir()
    (data / "mitre_cwe" / "cwec_v4.20.xml").symlink_to(DEFAULT.cwe_xml)
    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_path = data / "tokenizer.json"
    tok.save(str(tok_path))
    return tok_path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    tok = make_fixture(data)
    args = ["--data-dir", str(data), "--tokenizer", str(tok), "--unpinned-tokenizer"]
    assert build.main(["extract-nvd", *args]) == 0
    assert build.main(["all", *args]) == 0  # includes `check`, which returns 1 on any violation
    return data, tok, args


def test_facts(built):
    data, _, _ = built
    facts = {r["cve_id"]: r for r in build.read_jsonl(Paths(data).facts)}
    assert list(facts) == ["CVE-2016-0001", "CVE-2017-0002", "CVE-2021-0010"]
    beta = facts["CVE-2016-0001"]
    assert beta["func_names"] == ["beta_read"] and beta["selection_rule"] == "name_in_description"
    assert beta["patch_lines"] == [3] and beta["n_candidates"] == 2
    assert (beta["cwe"], beta["cvss_version"], beta["cvss_vector"]) == ("CWE-787", "3.1", V31[9:])
    assert "CWE-125" in beta["cwe_siblings"] and beta["cwe_parents"] == ["CWE-119"]
    hdr = facts["CVE-2017-0002"]
    assert hdr["cwe"] == "CWE-125" and hdr["cvss_version"] == "3.1" and hdr["cvss_both_versions"]
    assert hdr["patch_lines"] == [3]  # insertion after '{'; comment edits on lines 1 and 4 ignored
    assert hdr["n_code_lines"] == 5 and hdr["comment_mask_ok"]  # 1 of 5 lines: exactly 20%, kept
    cleanup = facts["CVE-2021-0010"]
    assert cleanup["cwe"] == "CWE-908" and cleanup["cwe_mapped_from"] == ["CWE-1187"] and cleanup["cvss_version"] == "3.0"
    assert cleanup["patch_lines"] == [5]


def test_drops(built):
    data, _, _ = built
    drops = {r["cve_id"]: (r["stage"], r["reason"]) for r in build.read_jsonl(Paths(data).drops)}
    assert drops == {
        "CVE-2018-0003": ("cvss_v3", "cna_only_v3"),
        "CVE-2018-0004": ("cwe", "category"),
        "CVE-2019-0005": ("empty_patch", "empty_patch"),
        "CVE-2019-0006": ("pair_integrity", "cross_commit_or_cve"),
        "CVE-2020-0007": ("token_cap", "over_token_cap"),
        "CVE-2020-0008": ("nvd_join", "not_in_nvd"),
        "CVE-2021-0009": ("nvd_join", "rejected"),
        "CVE-2022-0011": ("rewrite_guard", "inserted_over_20pct"),
        "CVE-2022-0012": ("patch_fraction", "patch_over_20pct"),
    }


def test_census(built):
    data, _, _ = built
    doc = json.loads(Paths(data).census_json.read_text())
    stages = [s["stage"] for s in doc["stages"]]
    assert stages == ["loaded", "pair_integrity", "nvd_join", "field_completeness", "cwe", "cvss_v3",
                      "empty_patch", "patch_fraction", "rewrite_guard", "token_cap", "select_one_function"]
    assert doc["stages"][0]["after"] == {"pairs": 13, "cves": 12, "commits": 12}
    assert doc["stages"][-1]["after"] == {"pairs": 3, "cves": 3, "commits": 3}
    assert doc["year_histogram"]["pre_v3"]["2018"] == 1 and doc["year_histogram"]["post_v3"].get("2018", 0) == 0
    assert doc["cvss_kept"] == {"3.1": 6, "3.0": 1, "both_versions": 1}
    assert doc["selection"] == {"cves_with_multiple_candidates": 1, "rules": {"name_in_description": 1, "single": 2}}
    assert doc["token_cap_report"]["caps"][0] == {"cap": 1024, "cves": 3, "pairs": 4}
    assert {"filter": "cvss_v3", "cves": 1, "pairs": 1} in doc["drop_alone"]
    assert doc["final"]["time_range"]["gate"].startswith("proceed")
    assert Paths(data).census_md.read_text().startswith("# Step 1 filtering census")


def test_parity_and_sheet(built):
    data, _, _ = built
    par = json.loads(Paths(data).parity_json.read_text())
    assert par["all_years"]["3.1"]["vs_pooled_majority"]["n"] == 2
    assert par["all_years"]["3.0"]["vs_pooled_majority"]["n"] == 1
    sheet = (Paths(data).verification / "line_sheet.csv").read_text().splitlines()
    assert len(sheet) == 4 and sheet[0].startswith("item,cve_id")
    manifest = json.loads(Paths(data).manifest.read_text())
    assert "facts" in manifest["outputs"] and "nvd_subset" in manifest["inputs"]


def test_rerun_is_byte_identical_across_hash_seeds(built, tmp_path):
    data, tok, args = built
    out = Paths(data)
    names = (out.facts, out.candidates, out.drops, out.census_json)
    digests = []
    for seed in ("0", "999"):
        subprocess.run([sys.executable, "-m", "etl.build", "build", *args], check=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": seed})
        digests.append([hashlib.sha256(p.read_bytes()).hexdigest() for p in names])
    assert digests[0] == digests[1]


# --- Step 2 ------------------------------------------------------------------


@pytest.fixture(scope="module")
def windowed(built):
    data, tok, args = built
    assert build.main(["test-window", *args]) == 0
    return data, tok, args


def test_test_window(windowed):
    data, _, _ = windowed
    out = Paths(data)
    split_rows = build.read_jsonl(out.split)
    assert [(r["cve_id"], r["pool"], r["moved"]) for r in split_rows] == [
        ("CVE-2016-0001", "nontest", False), ("CVE-2017-0002", "nontest", False), ("CVE-2021-0010", "test", False)]
    tw = json.loads(out.test_window_json.read_text())
    assert (tw["k"], tw["boundary_day"], tw["final"]) == (1, "2021-12-01", {"test": 1, "nontest": 2})
    assert tw["facts_sha256"] == hashlib.sha256(out.facts.read_bytes()).hexdigest()
    base = json.loads(out.baselines_json.read_text())
    h = base["exact_id_hierarchy"]
    assert h["symmetric_top"][0]["cwe"] == "CWE-125" and h["symmetric_top"][0]["mean_exact"] == "5/8"
    assert h["decision"] == pinned.EXACT_ID_SCHEDULE == "direction_aware"
    assert out.test_window_md.read_text().startswith("# Step 2: frozen test window")
    assert "split" in json.loads(out.manifest.read_text())["outputs"]


def test_test_window_rerun_is_identical(windowed):
    data, _, args = windowed
    out = Paths(data)
    before = [p.read_bytes() for p in (out.split, out.test_window_json, out.baselines_json)]
    assert build.main(["test-window", *args]) == 0
    assert [p.read_bytes() for p in (out.split, out.test_window_json, out.baselines_json)] == before


def test_test_window_is_frozen(windowed, tmp_path):
    import shutil

    data, tok, _ = windowed
    changed_facts = tmp_path / "facts_changed"
    shutil.copytree(data, changed_facts, symlinks=True)
    facts = Paths(changed_facts).facts
    facts.write_text("".join(facts.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(SystemExit, match="facts.jsonl changed"):
        build.main(["test-window", "--data-dir", str(changed_facts), "--tokenizer", str(tok), "--unpinned-tokenizer"])

    changed_split = tmp_path / "split_changed"
    shutil.copytree(data, changed_split, symlinks=True)
    s = Paths(changed_split).split
    s.write_text(s.read_text().replace('"pool": "test"', '"pool": "nontest"'))
    with pytest.raises(SystemExit, match="redrawn window differs"):
        build.main(["test-window", "--data-dir", str(changed_split), "--tokenizer", str(tok), "--unpinned-tokenizer"])
