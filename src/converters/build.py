"""Step 9: write every training file. A selection over frozen inputs: nothing is generated, no GPU is used.

Inputs are checked against the hashes their own steps recorded, so a converter can never read a bank, a
partition, a distill-self file or a DPO file other than the frozen ones. Outputs are frozen: a rebuild that would
change any file's content is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from etl.cwe_graph import CweGraph, load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from frozen_model.files import SessionFiles
from frozen_model.prepare import load_bank
from generators.progress import track
from generators.teacher.files import TeacherFiles

from . import arms, masks
from .files import ConverterFiles, gzip_bytes, read_content, sha256_bytes


@dataclass(frozen=True)
class Inputs:
    bank: list[dict]                 # non-test items, bank order
    facts: dict[str, dict]           # cve_id -> fact
    graph: CweGraph
    partition: list[dict]
    distill: dict[str, dict]         # item_id -> distill_self.jsonl row
    dpo: dict[str, dict]             # item_id -> dpo.jsonl row
    grpo_types: dict[str, dict]      # type -> {cap, band}, from step5_report.json
    sources: dict[str, str]

    @cached_property
    def by_id(self) -> dict[str, dict]:
        return {r["item_id"]: r for r in self.bank}

    def train_cves(self, seed: int) -> set[str]:
        return {r["cve_id"] for r in self.partition if r["role"][str(seed)] == "train"}

    def train_items(self, seed: int) -> list[dict]:
        cves = self.train_cves(seed)
        items = [r for r in self.bank if r["cve_id"] in cves]
        if len(items) != len(cves) * pinned.ITEMS_PER_CVE:
            raise SystemExit(f"seed {seed}: {len(items)} bank items for {len(cves)} train CVEs")
        return items


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(paths: Paths) -> Inputs:
    bank, bank_sha = load_bank(paths)
    session, teacher = SessionFiles.of(paths), TeacherFiles.of(paths)
    step5, step6, sub = _json(session.report_json), _json(teacher.report_json), _json(session.substitution)
    distill_sha, dpo_sha = file_sha256(session.distill_self), file_sha256(teacher.dpo)

    checks = (
        (step5["sources"]["bank_nontest_sha256"] == bank_sha, "step5_report.json was built from a different bank"),
        (step5["sources"]["distill_self_sha256"] == distill_sha, "distill_self.jsonl is not the file step 5 reported"),
        (step6["sources"]["bank_nontest_sha256"] == bank_sha, "step6_report.json was built from a different bank"),
        (step6["sources"]["dpo_sha256"] == dpo_sha, "dpo.jsonl is not the file step 6 reported"),
        (sub["substituted_types"] == [] and step5["substituted_types"] == [],
         "step 5 substituted a type: its distill-self rationales would come from the external model, which this "
         "converter does not read"),
    )
    for ok, message in checks:
        if not ok:
            raise SystemExit(message)

    part = _json(paths.partition_json)
    if part["facts_sha256"] != file_sha256(paths.facts) or part["split_sha256"] != file_sha256(paths.split):
        raise SystemExit("partition.json was drawn from different facts or split files")
    partition = read_jsonl(paths.partition)
    drawn = sorted({int(s) for r in partition for s in r["role"]})
    if drawn != list(pinned.PARTITION_SEEDS):
        raise SystemExit(f"partition.jsonl has seeds {drawn}, pinned {list(pinned.PARTITION_SEEDS)}")

    ids = [r["item_id"] for r in bank]
    distill = {r["item_id"]: r for r in read_jsonl(session.distill_self)}
    dpo = {r["item_id"]: r for r in read_jsonl(teacher.dpo)}
    for name, rows in (("distill_self.jsonl", distill), ("dpo.jsonl", dpo)):
        if sorted(rows) != sorted(ids):
            raise SystemExit(f"{name} does not cover exactly the non-test bank")
    if {r["cve_id"] for r in partition} != {r["cve_id"] for r in bank}:
        raise SystemExit("partition.jsonl does not cover exactly the non-test bank's CVEs")

    grpo_types = {t: {"cap": step5["grpo"][t]["cap"], "band": step5["grpo"][t]["band"]} for t in pinned.BANK_TYPES}
    sources = {
        "bank_nontest_sha256": bank_sha,
        "facts_sha256": file_sha256(paths.facts),
        "split_sha256": file_sha256(paths.split),
        "partition_sha256": file_sha256(paths.partition),
        "partition_json_sha256": file_sha256(paths.partition_json),
        "distill_self_sha256": distill_sha,
        "substitution_sha256": file_sha256(session.substitution),
        "step5_report_sha256": file_sha256(session.report_json),
        "dpo_sha256": dpo_sha,
        "step6_report_sha256": file_sha256(teacher.report_json),
        "cwe_xml_sha256": file_sha256(paths.cwe_xml),
    }
    return Inputs(bank, {f["cve_id"]: f for f in read_jsonl(paths.facts)}, load_cwe_graph(paths.cwe_xml), partition,
                  distill, dpo, grpo_types, sources)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def selection(inputs: Inputs, seed: int, config: str) -> list[tuple[str, int]]:
    return masks.selection([(r["item_id"], r["type"]) for r in inputs.train_items(seed)], config, seed)


def file_rows(inputs: Inputs, seed: int, config: str, arm: str, chosen: list[tuple[str, int]] | None = None) -> list[dict]:
    """One training file's rows, in file order."""
    if arm == "base":
        if config != masks.M1:
            raise ValueError("the base arm is built for M1 only")
        cves = sorted(inputs.train_cves(seed), key=lambda c: (
            pinned.stable_rank(c, "0", pinned.CONVERTER_SALTS["order"], config, str(seed)), c))
        return [arms.base(inputs.facts[c], inputs.graph) for c in cves]
    by_id = inputs.by_id
    chosen = selection(inputs, seed, config) if chosen is None else chosen
    if arm == "sft":
        return [arms.sft(by_id[i], c) for i, c in chosen]
    if arm == "distill_self":
        return [arms.distill_self(by_id[i], c, inputs.distill[i]) for i, c in chosen]
    if arm == "dpo":
        return [arms.dpo(by_id[i], c, inputs.dpo[i]) for i, c in chosen]
    if arm == "grpo":
        return [arms.grpo(by_id[i], c, inputs.grpo_types) for i, c in chosen]
    raise ValueError(f"unknown arm {arm!r}")


def specs() -> list[tuple[int, str, str]]:
    return [(s, c, a) for s in pinned.PARTITION_SEEDS for c in masks.CONFIGS for a in masks.arms(c)]


def contents(inputs: Inputs):
    """Yield (seed, config, arm, content bytes) for every training file; one selection per (seed, config)."""
    for seed in pinned.PARTITION_SEEDS:
        for config in masks.CONFIGS:
            chosen = selection(inputs, seed, config)
            for arm in masks.arms(config):
                yield seed, config, arm, jsonl_bytes(file_rows(inputs, seed, config, arm, chosen))


# ---------------------------------------------------------------------------
# Manifest and meta
# ---------------------------------------------------------------------------


def manifest(inputs: Inputs, files: ConverterFiles) -> dict:
    configs = {}
    for c in masks.CONFIGS:
        configs[c] = {
            "dropped_type": masks.dropped_type(c),
            "m1v_fraction": None if masks.m1v_fraction(c) is None else str(masks.m1v_fraction(c)),
            "m1v_comparator": masks.comparator(c),
            "dev_types": masks.dev_types(c),
            "arms": [*masks.arms(c), "sft_dpo"],  # sft_dpo reads the dpo file; whether it is trained is step 11's call
        }
    seeds = {}
    for s in pinned.PARTITION_SEEDS:
        dev = [r["cve_id"] for r in inputs.partition if r["role"][str(s)] == "dev"]
        seeds[str(s)] = {
            "train_cves": len(inputs.train_cves(s)),
            "train_items": len(inputs.train_cves(s)) * pinned.ITEMS_PER_CVE,
            "dev_cves": len(dev),
            "checkpoint_cves": sorted(r["cve_id"] for r in inputs.partition if s in r["checkpoint_seeds"]),
            "files": {c: {**{a: files.rel(s, c, a) for a in masks.arms(c)},
                          "sft_dpo": files.rel(s, c, pinned.SFT_DPO_READS)} for c in masks.CONFIGS},
        }
    return {
        "arms": {**{a: {"reads": a} for a in pinned.CONVERTER_ARMS}, "sft_dpo": {"reads": pinned.SFT_DPO_READS}},
        "configs": configs,
        "seeds": seeds,
        "completion_end": pinned.COMPLETION_END,
        "base_doc_end": pinned.BASE_DOC_END,
        "partition": "data/combined_dataset/partition.jsonl (dev and checkpoint CVEs per seed)",
    }


def pins() -> dict:
    return {
        "converter_arms": list(pinned.CONVERTER_ARMS),
        "question_arms": list(pinned.QUESTION_ARMS),
        "sft_dpo_reads": pinned.SFT_DPO_READS,
        "configs": list(masks.CONFIGS),
        "m1v_fractions": [str(f) for f in pinned.M1V_FRACTIONS],
        "salts": pinned.CONVERTER_SALTS,
        "completion_end": pinned.COMPLETION_END,
        "base_doc_end": pinned.BASE_DOC_END,
        "base_doc_template_sha256": pinned.sha256_text(pinned.BASE_DOC_TEMPLATE),
        "seeds": list(pinned.PARTITION_SEEDS),
    }


def _dumps(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(paths: Paths) -> dict:
    """Write every training file, the manifest and the meta. Refuses if any existing file would change."""
    inputs = load_inputs(paths)
    files = ConverterFiles.of(paths)
    recorded = _json(files.meta)["outputs"] if files.meta.exists() else {}

    # Pass 1, only if anything exists already: compare every existing file's content before writing anything.
    known: dict[str, dict] = {}
    if recorded or any(files.training(*spec).exists() for spec in specs()):
        changed = []
        for seed, config, arm, content in track(contents(inputs), "Converters (compare)", total=len(specs())):
            rel, path = files.rel(seed, config, arm), files.training(seed, config, arm)
            sha = sha256_bytes(content)
            on_disk = sha256_bytes(read_content(path)) if path.exists() else None
            if (on_disk is not None and on_disk != sha) or (rel in recorded and recorded[rel]["content_sha256"] != sha):
                changed.append(rel)
            elif on_disk is not None:
                known[rel] = {"content_sha256": sha, "sha256": file_sha256(path), "rows": content.count(b"\n")}
        if changed:
            raise SystemExit(f"the training files are frozen and this build differs in {len(changed)}, e.g. "
                             f"{changed[:3]}; a change needs a decision-record entry and a deliberate reset")

    outputs = dict(known)
    if len(known) < len(specs()):
        for seed, config, arm, content in track(contents(inputs), "Converters", total=len(specs())):
            rel = files.rel(seed, config, arm)
            if rel in known:
                continue
            path = files.training(seed, config, arm)
            path.parent.mkdir(parents=True, exist_ok=True)
            body = gzip_bytes(content)
            path.write_bytes(body)
            outputs[rel] = {"content_sha256": sha256_bytes(content), "sha256": sha256_bytes(body),
                            "rows": content.count(b"\n")}

    files.manifest.write_bytes(_dumps(manifest(inputs, files)))
    meta = {"sources": inputs.sources, "pins": pins(), "grpo_types": inputs.grpo_types,
            "manifest_sha256": file_sha256(files.manifest), "outputs": dict(sorted(outputs.items()))}
    files.meta.write_bytes(_dumps(meta))
    return meta
