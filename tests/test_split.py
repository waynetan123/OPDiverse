from fractions import Fraction

import pytest

from etl import pinned, split
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


def func(tag: str, n: int = 12) -> str:
    """A function with n distinct code lines."""
    return "void f_{0}(void)\n{{\n".format(tag) + "".join(f"    {tag}_{i} = {i};\n" for i in range(n - 3)) + "}"


def fact(cve: str, published: str, vuln: str | None = None, patched: str | None = None, cwe: str = "CWE-787") -> dict:
    vuln = vuln if vuln is not None else func(cve.replace("-", "_"))
    patched = patched if patched is not None else vuln.replace("= 0;", "= 100;", 1) + "\n// fixed"
    return {
        "cve_id": cve, "published": published, "cwe": cwe, "cvss_version": "3.1",
        "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", "primevul_split": "train",
        "vuln_func": vuln, "patched_func": patched,
        "vuln_norm_hash": pinned.norm_body_hash(vuln), "patched_norm_hash": pinned.norm_body_hash(patched),
    }


def timeline(n: int, *, same_day_tail: int = 0) -> list[dict]:
    """n CVEs one per day from 2020-01-01; the last `same_day_tail` share one day."""
    rows = []
    for i in range(n):
        day = min(i, n - same_day_tail) if same_day_tail else i
        rows.append(fact(f"CVE-2020-{1000 + i}", f"2020-{1 + day // 28:02d}-{1 + day % 28:02d}T00:00:{i:02d}.000"))
    return rows


# --- Boundary --------------------------------------------------------------


def test_k_is_ceiling_and_day_ties_go_to_test():
    rows = timeline(20)
    tw = split.draw_test_window(rows)
    assert tw["k"] == 3 and tw["final"]["test"] == 3
    tied = timeline(20, same_day_tail=4)  # the last 4 share a day; k = 3 lands inside that day
    tw = split.draw_test_window(tied)
    assert tw["k"] == 3 and tw["final"]["test"] == 4
    assert {a["cve_id"] for a in tw["assignments"] if a["pool"] == "test"} == {r["cve_id"] for r in tied[-4:]}


def test_requires_sorted_unique_facts():
    rows = timeline(5)
    with pytest.raises(AssertionError):
        split.draw_test_window(rows[::-1])
    with pytest.raises(AssertionError):
        split.draw_test_window(rows + [rows[-1]])


# --- Links -----------------------------------------------------------------


def test_exact_link_across_sides():
    a = fact("CVE-2020-0001", "2020-01-01T00:00:00.000")
    b = fact("CVE-2020-0002", "2020-02-01T00:00:00.000", vuln=a["patched_func"].replace("    ", "\t"))
    links = split.near_dup_links([a, b])
    assert links == [{"a": "CVE-2020-0001", "b": "CVE-2020-0002", "kind": "exact", "sides": "patched-vuln", "jaccard": 1.0}]


def test_fuzzy_threshold_is_inclusive_and_exact():
    long = func("x", 12)                                  # 12 code lines -> 10 shingles
    lines = long.split("\n")
    at_threshold = "\n".join(lines[:10])                  # 10 code lines -> 8 shingles, all shared: J = 8/10
    below = "\n".join(lines[:9])                          # 9 code lines -> 7 shingles: J = 7/10
    unrelated = func("u", 12)
    for other, linked in ((at_threshold, True), (below, False)):
        a = fact("CVE-2020-0001", "2020-01-01T00:00:00.000", vuln=long, patched=unrelated)
        b = fact("CVE-2020-0002", "2020-02-01T00:00:00.000", vuln=other, patched=func("v", 12))
        links = split.near_dup_links([a, b])
        assert bool(links) == linked
        if linked:
            assert links[0]["kind"] == "fuzzy" and links[0]["jaccard"] == 0.8
    assert pinned.jaccard_at_least(frozenset(range(8)), frozenset(range(10)))
    assert not pinned.jaccard_at_least(frozenset(range(7)), frozenset(range(10)))


def test_same_cve_never_links_to_itself():
    row = fact("CVE-2020-0001", "2020-01-01T00:00:00.000", vuln=func("s"), patched=func("s"))
    assert split.near_dup_links([row]) == []


# --- Moves -----------------------------------------------------------------


def test_straddling_cluster_moves_to_the_earlier_side():
    rows = timeline(20, same_day_tail=5)  # k = 3, but 5 share the boundary day -> 5 in test initially
    # The latest CVE's vulnerable function duplicates an early CVE's patched function: it moves.
    rows[-1] = fact(rows[-1]["cve_id"], rows[-1]["published"], vuln=rows[2]["patched_func"])
    # Two test CVEs duplicating each other stay in test together.
    rows[-2] = fact(rows[-2]["cve_id"], rows[-2]["published"], vuln=rows[-3]["vuln_func"] + "\n")
    tw = split.draw_test_window(rows)
    pool = {a["cve_id"]: a["pool"] for a in tw["assignments"]}
    assert tw["k"] == 3 and tw["initial"]["test"] == 5
    assert [m["cve_id"] for m in tw["moved"]] == [rows[-1]["cve_id"]]
    assert pool[rows[-1]["cve_id"]] == "nontest" and pool[rows[2]["cve_id"]] == "nontest"
    assert pool[rows[-2]["cve_id"]] == pool[rows[-3]["cve_id"]] == "test"
    assert tw["final"]["test"] == 4
    clusters = {c["cluster_id"]: c for c in tw["multi_cve_clusters"]}
    assert clusters[rows[2]["cve_id"]]["members"] == [rows[2]["cve_id"], rows[-1]["cve_id"]]
    assert clusters[rows[-3]["cve_id"]]["pool"] == "test"


def test_refuses_to_shave_test():
    rows = timeline(20)
    for i in (17, 18, 19):  # every test CVE duplicates an early one
        rows[i] = fact(rows[i]["cve_id"], rows[i]["published"], vuln=rows[i - 17]["vuln_func"])
    with pytest.raises(SystemExit, match="never shave test"):
        split.draw_test_window(rows)


# --- Hierarchy credit ------------------------------------------------------


def test_distances(graph):
    assert pinned.cwe_distance("119", "787", graph) == 1
    assert pinned.cwe_distance("787", "125", graph) == 2
    assert pinned.cwe_distance("118", "787", graph) == 2
    assert pinned.cwe_distance("787", "79", graph) is None


@pytest.mark.parametrize("pred, gold, relation, direction_aware, symmetric", [
    ("CWE-787", "CWE-787", "exact", Fraction(1), Fraction(1)),
    ("CWE-787", "CWE-119", "child", Fraction(1, 2), Fraction(1, 2)),
    ("CWE-119", "CWE-787", "parent", Fraction(1, 4), Fraction(1, 2)),
    ("CWE-125", "CWE-787", "sibling", Fraction(1, 4), Fraction(1, 4)),
    ("CWE-118", "CWE-787", "ancestor_2", Fraction(1, 8), Fraction(1, 4)),
    ("CWE-672", "CWE-415", "ancestor_2", Fraction(1, 8), Fraction(1, 4)),  # also a sibling; ancestor wins
    ("CWE-121", "CWE-119", "descendant_2", Fraction(1, 4), Fraction(1, 4)),
    ("CWE-79", "CWE-787", "far", Fraction(0), Fraction(0)),
])
def test_relations_and_scores(graph, pred, gold, relation, direction_aware, symmetric):
    assert pinned.cwe_relation(pred[4:], gold[4:], graph) == relation
    assert pinned.hierarchy_score(pred, gold, graph, "direction_aware") == direction_aware
    assert pinned.hierarchy_score(pred, gold, graph, "symmetric") == symmetric


def test_coparent_scores_quarter(graph):
    live = sorted((c for c in graph.weaknesses if graph.in_view(c)), key=int)
    pair = next((p, g) for g in live for p in live if pinned.cwe_relation(p, g, graph) == "coparent")
    p, g = pair
    assert any(p in graph.parents(c) for c in graph.children(g))
    assert pinned.hierarchy_score(f"CWE-{p}", f"CWE-{g}", graph, "direction_aware") == Fraction(1, 4)


def test_invalid_predictions_and_gold(graph):
    assert pinned.hierarchy_score("CWE-399", "CWE-787", graph) == 0   # category
    assert pinned.hierarchy_score("CWE-1187", "CWE-908", graph) == 0  # deprecated
    assert pinned.hierarchy_score("787", "CWE-787", graph) == 0       # not a CWE string
    assert pinned.hierarchy_score(" cwe-787 ", "CWE-787", graph) == 1
    with pytest.raises(ValueError):
        pinned.hierarchy_score("CWE-787", "CWE-399", graph)


def test_scores_stay_in_unit_interval(graph):
    golds = ["CWE-787", "CWE-125", "CWE-119", "CWE-476", "CWE-416", "CWE-20", "CWE-415"]
    preds = [f"CWE-{c}" for c in sorted(graph.weaknesses, key=int)[:300]]
    for schedule in pinned.SCHEDULES:
        for g in golds:
            for p in preds:
                assert 0 <= pinned.hierarchy_score(p, g, graph, schedule) <= 1


# --- Baselines -------------------------------------------------------------


def test_baselines_pick_best_constant_and_break_ties_low(graph):
    rows = [fact("CVE-2020-0001", "2020-01-01T00:00:00.000", cwe="CWE-787"),
            fact("CVE-2020-0002", "2020-01-02T00:00:00.000", cwe="CWE-125")]
    base = split.baselines(rows, graph)
    h = base["exact_id_hierarchy"]
    assert h["symmetric_top"][0] == {"cwe": "CWE-125", "mean": 0.625, "mean_exact": "5/8"}  # ties with 787
    assert h["decision"] == "direction_aware"
    assert h["adopted_schedule_top"][0]["cwe"] == "CWE-125"
    assert base["most_frequent_cwe"]["cwe"] == "CWE-125"
    assert base["cvss_majority"]["mean_component_agreement"] == 1.0
