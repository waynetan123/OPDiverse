"""MITRE CWE XML -> the view-1000 ChildOf hierarchy. The only code that reads the XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from . import pinned

_NS = "{http://cwe.mitre.org/cwe-7}"


@dataclass(frozen=True)
class Weakness:
    name: str
    abstraction: str
    status: str


@dataclass(frozen=True)
class CweGraph:
    release: str
    weaknesses: dict[str, Weakness]
    categories: frozenset[str]
    views: frozenset[str]
    parents_map: dict[str, frozenset[str]]
    children_map: dict[str, frozenset[str]]
    deprecated_descriptions: dict[str, str]

    def is_weakness(self, cwe: str) -> bool:
        return cwe in self.weaknesses

    def is_deprecated_weakness(self, cwe: str) -> bool:
        return cwe in self.weaknesses and self.weaknesses[cwe].status == "Deprecated"

    def is_category(self, cwe: str) -> bool:
        return cwe in self.categories

    def is_view(self, cwe: str) -> bool:
        return cwe in self.views

    def in_view(self, cwe: str) -> bool:
        """A non-deprecated weakness placed in view 1000 (has a parent there, or is a pillar)."""
        w = self.weaknesses.get(cwe)
        return w is not None and w.status != "Deprecated" and (cwe in self.parents_map or w.abstraction == "Pillar")

    def parents(self, cwe: str) -> frozenset[str]:
        return self.parents_map.get(cwe, frozenset())

    def children(self, cwe: str) -> frozenset[str]:
        return self.children_map.get(cwe, frozenset())

    def _closure(self, cwe: str, step) -> frozenset[str]:
        seen: set[str] = set()
        frontier = list(step(cwe))
        while frontier:
            node = frontier.pop()
            if node not in seen:
                seen.add(node)
                frontier.extend(step(node))
        return frozenset(seen)

    def ancestors(self, cwe: str) -> frozenset[str]:
        return self._closure(cwe, self.parents)

    def descendants(self, cwe: str) -> frozenset[str]:
        return self._closure(cwe, self.children)

    def check_invariants(self) -> None:
        """Every non-deprecated weakness is in view 1000, edges point at live weaknesses, no cycles."""
        live = {c for c, w in self.weaknesses.items() if w.status != "Deprecated"}
        missing = sorted((c for c in live if not self.in_view(c)), key=int)
        assert not missing, f"non-deprecated weaknesses outside view {pinned.CWE_VIEW}: {missing[:10]}"
        for child, parents in self.parents_map.items():
            assert child in live, f"CWE-{child} has ChildOf edges but is not a live weakness"
            bad = sorted(p for p in parents if p not in live)
            assert not bad, f"CWE-{child} ChildOf non-live {bad}"
        cyclic = sorted((c for c in self.parents_map if c in self.ancestors(c)), key=int)
        assert not cyclic, f"ChildOf cycle through {cyclic[:10]}"
        uncovered = sorted(c for c in self.deprecated_descriptions if c not in pinned.DEPRECATED_REPLACEMENT)
        assert not uncovered, f"DEPRECATED_REPLACEMENT lacks {uncovered}"


def load_cwe_graph(path: Path) -> CweGraph:
    root = ET.parse(path).getroot()
    release = root.get("Version", "")
    assert release == pinned.CWE_RELEASE, f"CWE XML is release {release}, pinned {pinned.CWE_RELEASE}"
    weaknesses: dict[str, Weakness] = {}
    parents: dict[str, set[str]] = {}
    deprecated: dict[str, str] = {}
    for w in root.iter(f"{_NS}Weakness"):
        cwe = w.get("ID")
        weaknesses[cwe] = Weakness(w.get("Name", ""), w.get("Abstraction", ""), w.get("Status", ""))
        if w.get("Status") == "Deprecated":
            desc = w.find(f"{_NS}Description")
            deprecated[cwe] = " ".join("".join(desc.itertext()).split()) if desc is not None else ""
        for rel in w.iter(f"{_NS}Related_Weakness"):
            if rel.get("Nature") == "ChildOf" and rel.get("View_ID") == pinned.CWE_VIEW:
                parents.setdefault(cwe, set()).add(rel.get("CWE_ID"))
    children: dict[str, set[str]] = {}
    for child, ps in parents.items():
        for p in ps:
            children.setdefault(p, set()).add(child)
    return CweGraph(
        release=release,
        weaknesses=weaknesses,
        categories=frozenset(c.get("ID") for c in root.iter(f"{_NS}Category")),
        views=frozenset(v.get("ID") for v in root.iter(f"{_NS}View")),
        parents_map={k: frozenset(v) for k, v in parents.items()},
        children_map={k: frozenset(v) for k, v in children.items()},
        deprecated_descriptions=deprecated,
    )
