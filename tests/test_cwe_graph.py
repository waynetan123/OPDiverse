import pytest

from etl import pinned
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


def test_release_and_invariants(graph):
    assert graph.release == pinned.CWE_RELEASE
    graph.check_invariants()


def test_view_1000_membership_counts(graph):
    live = [c for c, w in graph.weaknesses.items() if w.status != "Deprecated"]
    assert len(live) == 944
    assert sum(c in graph.parents_map for c in live) == 934
    assert sum(graph.weaknesses[c].abstraction == "Pillar" for c in live) == 10


def test_deprecated_map_covers_release_exactly(graph):
    assert set(graph.deprecated_descriptions) == set(pinned.DEPRECATED_REPLACEMENT)
    for old, new in pinned.DEPRECATED_REPLACEMENT.items():
        if new is not None:
            assert f"CWE-{new}" in graph.deprecated_descriptions[old], old  # the successor is named in MITRE's text
            assert graph.in_view(new) or graph.is_category(new)


def test_known_edges(graph):
    assert graph.parents("787") == {"119"}
    assert graph.parents("119") == {"118"}
    assert graph.children("118") == {"119"}
    assert graph.parents("415") == {"1341", "666", "825"}
    assert "672" in graph.ancestors("415")
    assert "787" in graph.descendants("118")
    assert graph.is_category("399") and not graph.in_view("399")
