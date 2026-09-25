import os
import subprocess
import sys
from pathlib import Path

import pytest

from etl import pinned
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


# --- Function text ---------------------------------------------------------


def test_clean_function_line_endings_and_blank_edges():
    assert pinned.clean_function("\n\n  int f()\r\n{\r  return 0;\r\n}\n\n  \n") == "  int f()\n{\n  return 0;\n}"


def test_split_lines_ignores_form_feed_and_vertical_tab():
    assert pinned.split_lines("a\fb\vc\nd") == ["a\fb\vc", "d"]


def test_render_numbered_is_one_indexed():
    assert pinned.render_numbered("int f()\n{\n}") == "1: int f()\n2: {\n3: }"


# --- Comment masking -------------------------------------------------------


@pytest.mark.parametrize("src, expected", [
    ("a = 1; // note\nb = 2;", "a = 1;        \nb = 2;"),
    ("a /* x\ny */ b", "a     \n     b"),
    ('s = "// not a comment";', 's = "// not a comment";'),
    ("c = '\"'; // q", "c = '\"';     "),
    ('s = "esc \\" /* still string */";', 's = "esc \\" /* still string */";'),
    ('r = R"x(/* raw */)x"; // c', 'r = R"x(/* raw */)x";     '),
    ("n = 1'000'000; // sep", "n = 1'000'000;       "),
    ("// cont \\\nstill comment\nx;", " " * 9 + "\n" + " " * 13 + "\nx;"),
])
def test_mask_comments(src, expected):
    masked, ok = pinned.mask_comments(src)
    assert ok and masked == expected
    assert masked.count("\n") == src.count("\n")


@pytest.mark.parametrize("src", ["a /* never closed", 's = "unterminated\nx;', "#error don't\nx;"])
def test_mask_comments_unterminated(src):
    assert pinned.mask_comments(src) == (src, False)


# --- Patch line set --------------------------------------------------------

BASE = "int f(int a)\n{\n    int r = a;\n    r += 1;\n    return r;\n}"


def lines(vuln, patched):
    return pinned.patch_line_set(vuln, patched).lines


def test_identical_reindent_blank_and_comment_changes_are_empty():
    assert lines(BASE, BASE) == ()
    assert lines(BASE, BASE.replace("    ", "\t")) == ()
    assert lines(BASE, BASE.replace("{\n", "{\n\n\n")) == ()
    assert lines(BASE, BASE.replace("r += 1;", "r += 1; /* bump */")) == ()
    assert lines(BASE, BASE.replace("r += 1;", "r+=1;")) == ()  # all whitespace removed from the key


def test_replace_and_delete():
    assert lines(BASE, BASE.replace("r += 1;", "r += 2;")) == (4,)
    assert lines(BASE, BASE.replace("    r += 1;\n", "")) == (4,)


def test_insert_attribution():
    mid = BASE.replace("    r += 1;\n", "    r += 1;\n    if (r < 0) r = 0;\n")
    assert lines(BASE, mid) == (4,)
    after_blank = "int f(int a)\n{\n    int r = a;\n\n    return r;\n}"
    patched = after_blank.replace("\n\n", "\n\n    r++;\n")
    assert lines(after_blank, patched) == (3,)  # blank line 4 is skipped; attributed to line 3
    assert lines(BASE, "#include <x.h>\n" + BASE) == (1,)  # before all code lines
    assert lines(BASE, BASE + "\nint g;") == (6,)  # after the last line


def test_insertion_counts_and_multi_hunk():
    patched = BASE.replace("    int r = a;\n", "    int r = a + 0;\n    int s;\n").replace("return r;", "return r + 0;")
    r = pinned.patch_line_set(BASE, patched)
    assert r.lines == (3, 5)
    assert (r.n_inserted, r.n_deleted, r.n_code_lines, r.n_lines) == (3, 2, 6, 6)


def test_autojunk_disabled_on_long_functions():
    body = []
    for i in range(120):
        body += [f"    case {i}:", "        x++;", "        break;"]
    vuln = "void f(int x)\n{\n    switch (x) {\n" + "\n".join(body) + "\n    }\n}"
    patched = vuln.replace("    case 60:\n        x++;", "    case 60:\n        x += 2;")
    assert len(pinned.split_lines(vuln)) > 200
    target = pinned.split_lines(vuln).index("    case 60:") + 2  # the "x++;" after case 60
    assert lines(vuln, patched) == (target,)


def test_mask_failure_on_either_side_disables_masking_for_both():
    vuln = "int f(void)\n{\n    return 0; /* ok */\n}"
    patched = "int f(void)\n{\n    return 0; /* ok */\n    #error don't\n}"
    r = pinned.patch_line_set(vuln, patched)
    assert not r.mask_ok and r.lines == (3,)


def test_exceeds_fraction_boundary():
    assert not pinned.exceeds_fraction(2, 10)
    assert pinned.exceeds_fraction(3, 10)
    assert not pinned.exceeds_fraction(1, 5)
    assert pinned.exceeds_fraction(2, 9)


def test_norm_body_hash_ignores_whitespace_and_comments():
    assert pinned.norm_body_hash(BASE) == pinned.norm_body_hash(BASE.replace("    ", "\t") + "\n// tail")
    assert pinned.norm_body_hash(BASE) != pinned.norm_body_hash(BASE.replace("1", "2"))


# --- CVSS ------------------------------------------------------------------


def test_parse_cvss_any_order_and_case():
    version, comps = pinned.parse_cvss_v3("cvss:3.1/a:h/i:h/c:h/s:u/ui:n/pr:n/ac:l/av:n")
    assert version == "3.1"
    assert pinned.canonical_cvss(comps) == "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert pinned.parse_cvss_v3("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")[0] is None


@pytest.mark.parametrize("vector", [
    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H",           # missing A
    "CVSS:3.1/AV:N/AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # duplicate
    "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",       # bad value
    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H/E:P",   # temporal metric
    "CVSS:2.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",       # wrong version
    "AV:N/AC:L/Au:N/C:P/I:P/A:P",                          # v2
])
def test_parse_cvss_rejects(vector):
    assert pinned.parse_cvss_v3(vector) is None


def metric(source, version, vector, **fields):
    return {"source": source, "type": "Primary", "cvssData": {"version": version, "vectorString": vector, **fields}}


V31 = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
V30 = "CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N"


def test_select_prefers_nvd_then_v31():
    both = {"cvssMetricV31": [metric(pinned.NVD_SOURCE, "3.1", V31)], "cvssMetricV30": [metric(pinned.NVD_SOURCE, "3.0", V30)]}
    got, reason = pinned.select_nvd_cvss(both)
    assert reason == "ok" and got["version"] == "3.1" and got["both_versions"]
    nvd30_cna31 = {"cvssMetricV31": [metric("cna@example.com", "3.1", V31)], "cvssMetricV30": [metric(pinned.NVD_SOURCE, "3.0", V30)]}
    got, reason = pinned.select_nvd_cvss(nvd30_cna31)
    assert reason == "ok" and got["version"] == "3.0" and got["vector"] == V30[9:]


@pytest.mark.parametrize("metrics, reason", [
    ({"cvssMetricV31": [metric("cna@example.com", "3.1", V31)]}, "cna_only_v3"),
    ({"cvssMetricV2": [metric(pinned.NVD_SOURCE, "2.0", "AV:N/AC:L/Au:N/C:P/I:P/A:P")]}, "v2_only"),
    ({"cvssMetricV40": [metric("cna@example.com", "4.0", "CVSS:4.0/AV:N")]}, "v4_only"),
    ({}, "no_cvss"),
    ({"cvssMetricV31": [metric(pinned.NVD_SOURCE, "3.1", V31), metric(pinned.NVD_SOURCE, "3.1", V31.replace("A:H", "A:L"))]}, "conflicting_nvd_vectors"),
    ({"cvssMetricV31": [metric(pinned.NVD_SOURCE, "3.1", V30)]}, "version_mismatch"),
    ({"cvssMetricV31": [metric(pinned.NVD_SOURCE, "3.1", V31, attackVector="LOCAL")]}, "decomposed_mismatch"),
])
def test_select_drop_reasons(metrics, reason):
    assert pinned.select_nvd_cvss(metrics) == (None, reason)


def test_decomposed_fields_accepted_when_consistent():
    m = metric(pinned.NVD_SOURCE, "3.1", V31, attackVector="NETWORK", attackComplexity="LOW", privilegesRequired="NONE",
               userInteraction="NONE", scope="UNCHANGED", confidentialityImpact="HIGH", integrityImpact="HIGH",
               availabilityImpact="HIGH")
    assert pinned.select_nvd_cvss({"cvssMetricV31": [m]})[1] == "ok"


# --- CWE -------------------------------------------------------------------


def weakness(source, *values, type_="Primary"):
    return {"source": source, "type": type_, "description": [{"lang": "en", "value": v} for v in values]}


def test_nvd_cwe_values_filters_on_source_not_type():
    blocks = [weakness(pinned.NVD_SOURCE, "CWE-787", type_="Secondary"), weakness("cna@example.com", "CWE-20"),
              weakness(pinned.NVD_SOURCE, "CWE-787", "NVD-CWE-Other")]
    assert pinned.nvd_cwe_values(blocks) == (True, ["CWE-787", "NVD-CWE-Other"])
    assert pinned.nvd_cwe_values([weakness("cna@example.com", "CWE-20")]) == (False, [])


@pytest.mark.parametrize("has_block, values, expected", [
    (True, ["CWE-787", "NVD-CWE-Other"], ("CWE-787", "ok", ())),
    (True, ["NVD-CWE-noinfo"], (None, "placeholder_only", ())),
    (False, [], (None, "no_nvd_cwe", ())),
    (True, ["CWE-1187", "CWE-908"], ("CWE-908", "ok", ("CWE-1187",))),
    (True, ["CWE-1187"], ("CWE-908", "ok", ("CWE-1187",))),
    (True, ["CWE-216"], (None, "deprecated_no_replacement", ("CWE-216",))),
    (True, ["CWE-787", "CWE-125"], (None, "multi_cwe", ())),
    (True, ["CWE-399"], (None, "category", ())),
    (True, ["CWE-225"], (None, "category", ("CWE-225",))),
    (True, ["CWE-1000"], (None, "view", ())),
    (True, ["CWE-99999"], (None, "not_in_view_1000", ())),
    (True, ["cwe 787"], (None, "malformed_cwe", ())),
])
def test_resolve_cwe(graph, has_block, values, expected):
    assert pinned.resolve_cwe(has_block, values, graph) == expected


def test_siblings(graph):
    assert "CWE-125" in pinned.siblings("787", graph)
    assert pinned.siblings("119", graph) == []
    assert "CWE-672" not in pinned.siblings("415", graph)  # 672 is also 415's grandparent via 825
    sibs = pinned.siblings("476", graph)
    assert sibs == sorted(sibs, key=lambda s: int(s[4:])) and "CWE-476" not in sibs


# --- Function names and selection -----------------------------------------


@pytest.mark.parametrize("src, expected", [
    ("static int foo(int a)\n{\n}", ("foo",)),
    ("static Image *ReadOneJNGImage(MngInfo *m,\n    const ImageInfo *i)\n{", ("ReadOneJNGImage",)),
    ("int Foo::Bar::baz(int x) const\n{", ("Foo::Bar::baz", "baz")),
    ("Foo::~Foo()\n{", ("Foo::~Foo", "Foo")),
    ("Foo::Foo(int x) : m_x(x), m_y{0}\n{", ("Foo::Foo", "Foo", "m_x")),
    ("PHP_FUNCTION(array_walk)\n{", ("PHP_FUNCTION", "array_walk")),
    ("SYSCALL_DEFINE4(epoll_ctl, int, epfd, int, op)\n{", ("SYSCALL_DEFINE4", "epoll_ctl")),
    ("int\nold_style(a, b)\n    int a; char *b;\n{", ("old_style",)),
    ("static __attribute__((unused)) int quiet(void)\n{", ("quiet",)),
    ("/* helper(x) */\nint real_name(void)\n{", ("real_name",)),
    ("    while (1) {\n        x++;\n    }", ()),
])
def test_extract_function_names(src, expected):
    assert pinned.extract_function_names(src) == expected


def test_name_in_description_whole_word_case_sensitive():
    desc = "A heap overflow in the png_read_row function of libpng."
    assert pinned.name_in_description(("png_read_row",), desc)
    assert not pinned.name_in_description(("png_read",), desc)
    assert not pinned.name_in_description(("PNG_READ_ROW",), desc)


def test_stable_rank_golden():
    assert pinned.stable_rank("CVE-2016-0001", "abc") == 0x73B757FBC6BCC229


def test_choose_function_rules():
    c = pinned.Candidate
    assert pinned.choose_function("CVE-1", [c("p1", "h1", False)]) == ("p1", "single")
    assert pinned.choose_function("CVE-1", [c("p1", "h1", False), c("p2", "h2", True)]) == ("p2", "name_in_description")
    pid, rule = pinned.choose_function("CVE-1", [c("p1", "h1", True), c("p2", "h2", True), c("p3", "h3", False)])
    assert rule == "name_in_description+hash" and pid in ("p1", "p2")
    pid, rule = pinned.choose_function("CVE-1", [c("p1", "h1", False), c("p2", "h2", False)])
    expected = min(("p1", "h1"), ("p2", "h2"), key=lambda t: pinned.stable_rank("CVE-1", t[1]))[0]
    assert (pid, rule) == (expected, "hash")


def test_choice_independent_of_hash_seed():
    code = (
        "from etl import pinned\n"
        "c = [pinned.Candidate(f'p{i}', f'h{i}', False) for i in range(50)]\n"
        "print([pinned.choose_function(f'CVE-{k}', c)[0] for k in range(30)])\n"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": seed}).stdout
        for seed in ("0", "999")
    }
    assert len(outs) == 1
