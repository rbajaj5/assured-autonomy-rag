"""Build and query a lightweight assurance concept graph.

This is a deterministic GraphRAG/HippoRAG-style memory layer over artifacts the
project already produces. It does not replace vector retrieval; it gives the
RAG a persistent map of claims, methods, assumptions, metrics, failure modes,
bridge concepts, and project hooks that can be retrieved before synthesis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path("concept_graph_runs")
STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "because", "before",
    "between", "could", "does", "during", "from", "have", "into", "more",
    "must", "only", "other", "should", "such", "than", "that", "their",
    "then", "there", "these", "this", "through", "under", "using", "when",
    "where", "which", "while", "with", "without", "would",
}


@dataclass
class ConceptNode:
    id: str
    kind: str
    label: str
    description: str = ""
    sources: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    score: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptEdge:
    source: str
    relation: str
    target: str
    evidence: str = ""
    sources: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class ConceptGraph:
    nodes: dict[str, ConceptNode] = field(default_factory=dict)
    edges: list[ConceptEdge] = field(default_factory=list)

    def add_node(
        self,
        *,
        kind: str,
        label: str,
        description: str = "",
        source: str = "",
        module: str = "",
        score: float = 1.0,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        label = _clean(label)
        if not label:
            return ""
        node_id = f"{kind}:{_slug(label)}"
        existing = self.nodes.get(node_id)
        if existing:
            if description and description not in existing.description:
                existing.description = _merge_text(existing.description, description)
            if source and source not in existing.sources:
                existing.sources.append(source)
            if module and module not in existing.modules:
                existing.modules.append(module)
            existing.score += score
            if attrs:
                existing.attrs.update({k: v for k, v in attrs.items() if v not in (None, "", [])})
        else:
            self.nodes[node_id] = ConceptNode(
                id=node_id,
                kind=kind,
                label=label,
                description=description,
                sources=[source] if source else [],
                modules=[module] if module else [],
                score=score,
                attrs=attrs or {},
            )
        return node_id

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        evidence: str = "",
        source_path: str = "",
        weight: float = 1.0,
    ) -> None:
        if not source or not target or source == target:
            return
        self.edges.append(
            ConceptEdge(
                source=source,
                relation=relation,
                target=target,
                evidence=_clean(evidence),
                sources=[source_path] if source_path else [],
                weight=weight,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in sorted(self.nodes.values(), key=lambda n: (n.kind, n.label))],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConceptGraph":
        graph = cls()
        for row in data.get("nodes", []):
            node = ConceptNode(**row)
            graph.nodes[node.id] = node
        graph.edges = [ConceptEdge(**row) for row in data.get("edges", [])]
        return graph


def _clean(text: Any, limit: int = 900) -> str:
    if isinstance(text, dict):
        bits = []
        for key, value in text.items():
            if isinstance(value, (str, int, float)):
                bits.append(f"{key}: {value}")
        text = "; ".join(bits)
    elif isinstance(text, list):
        text = "; ".join(_clean(x, limit=180) for x in text[:8])
    else:
        text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text[:90] or "item"


def _merge_text(a: str, b: str, limit: int = 1600) -> str:
    if not a:
        return b[:limit]
    if not b or b in a:
        return a
    return (a + " | " + b)[:limit]


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if t not in STOPWORDS
    ]


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_json_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file() and root.suffix.lower() == ".json":
            yield root
        elif root.is_dir():
            yield from root.rglob("*.json")


def build_concept_graph(
    *,
    artifact_roots: list[Path],
    max_files: int | None = None,
) -> ConceptGraph:
    graph = ConceptGraph()
    count = 0
    for path in _iter_json_files(artifact_roots):
        data = _load_json(path)
        if data is None:
            continue
        rel = str(path)
        _ingest_artifact(graph, data, rel)
        count += 1
        if max_files and count >= max_files:
            break
    return graph


def _ingest_artifact(graph: ConceptGraph, data: Any, source_path: str) -> None:
    if not isinstance(data, dict):
        return
    if "assurance_case_version" in data or (
        "claim_layers" in data and "assurance_role" in data and "hidden_witnesses" in data
    ):
        _ingest_assurance_case_card(graph, data, source_path)
    elif "evidence_gate" in data or "lift" in data or "corpus_recommendations" in data:
        _ingest_council_run(graph, data, source_path)
    elif "bridge_concepts" in data or "abstract_assurance_principle" in data:
        _ingest_bridge_card(graph, data, source_path)
    elif "paper_central_claim" in data or "method_or_framework" in data:
        _ingest_paper_card(graph, data, source_path)
    elif "chunk_claims" in data or "methods_or_evidence" in data:
        _ingest_chunk_card(graph, data, source_path)


def _paper_node(graph: ConceptGraph, data: dict[str, Any], source_path: str) -> str:
    label = data.get("short_path") or data.get("filename") or Path(source_path).stem
    return graph.add_node(
        kind="paper",
        label=str(label),
        description=_clean(data.get("paper_central_claim") or data.get("bridge_summary") or ""),
        source=source_path,
        module=str(data.get("module_id") or ""),
        attrs={
            "module_name": data.get("module_name"),
            "filename": data.get("filename"),
            "status": data.get("status"),
        },
    )


def _ingest_paper_card(graph: ConceptGraph, data: dict[str, Any], source_path: str) -> None:
    paper = _paper_node(graph, data, source_path)
    module = str(data.get("module_id") or "")
    for field, kind, relation in [
        ("paper_central_claim", "claim", "makes_claim"),
        ("method_or_framework", "method", "uses_method"),
        ("capacity_allocation_relevance", "project_hook", "has_project_hook"),
        ("techtrack_upgrade_relevance", "project_hook", "has_project_hook"),
    ]:
        node = graph.add_node(kind=kind, label=_clean(data.get(field)), source=source_path, module=module)
        graph.add_edge(paper, relation, node, evidence=field, source_path=source_path)
    _ingest_list_field(graph, paper, data, "specific_concepts", "concept", "mentions_concept", source_path, module)
    _ingest_list_field(graph, paper, data, "assurance_tests_or_metrics", "metric", "supports_metric", source_path, module)
    _ingest_list_field(graph, paper, data, "assumptions_limits", "assumption", "is_limited_by", source_path, module)
    _ingest_list_field(graph, paper, data, "misuse_or_overclaim_warnings", "overclaim_warning", "warns_against", source_path, module)
    _ingest_list_field(graph, paper, data, "citation_snippets", "citation_anchor", "has_citation_anchor", source_path, module)


def _ingest_chunk_card(graph: ConceptGraph, data: dict[str, Any], source_path: str) -> None:
    paper = _paper_node(graph, data, source_path)
    module = str(data.get("module_id") or "")
    chunk = graph.add_node(
        kind="chunk",
        label=f"{data.get('short_path') or Path(source_path).stem} {data.get('chunk_id') or ''}".strip(),
        description=f"pages {data.get('page_start', '?')}-{data.get('page_end', '?')}",
        source=source_path,
        module=module,
    )
    graph.add_edge(paper, "has_chunk", chunk, source_path=source_path)
    _ingest_list_field(graph, chunk, data, "chunk_claims", "claim", "makes_claim", source_path, module)
    _ingest_list_field(graph, chunk, data, "methods_or_evidence", "method", "uses_method", source_path, module)
    _ingest_list_field(graph, chunk, data, "assumptions_and_limits", "assumption", "is_limited_by", source_path, module)
    _ingest_list_field(graph, chunk, data, "specific_concepts", "concept", "mentions_concept", source_path, module)
    _ingest_list_field(graph, chunk, data, "capacity_allocation_hooks", "project_hook", "has_project_hook", source_path, module)
    _ingest_list_field(graph, chunk, data, "techtrack_upgrade_hooks", "project_hook", "has_project_hook", source_path, module)
    _ingest_list_field(graph, chunk, data, "assurance_tests_or_metrics", "metric", "supports_metric", source_path, module)
    _ingest_list_field(graph, chunk, data, "overclaim_warnings", "overclaim_warning", "warns_against", source_path, module)
    _ingest_list_field(graph, chunk, data, "page_anchors", "citation_anchor", "has_citation_anchor", source_path, module)


def _ingest_bridge_card(graph: ConceptGraph, data: dict[str, Any], source_path: str) -> None:
    bridge = graph.add_node(
        kind="bridge_card",
        label=data.get("title") or data.get("id") or Path(source_path).stem,
        description=_clean(data.get("abstract_assurance_principle")),
        source=source_path,
        attrs={"source_url": data.get("source_url"), "kind": data.get("kind")},
    )
    for module in data.get("modules", []) or []:
        mod = graph.add_node(kind="module", label=str(module), source=source_path)
        graph.add_edge(bridge, "maps_to_module", mod, source_path=source_path)
    _ingest_list_field(graph, bridge, data, "bridge_concepts", "concept", "bridges_concept", source_path, "")
    _ingest_list_field(graph, bridge, data, "autonomy_instantiations", "project_hook", "instantiates_as", source_path, "")
    _ingest_list_field(graph, bridge, data, "threat_models", "failure_mode", "has_threat_model", source_path, "")
    _ingest_list_field(graph, bridge, data, "retrieval_expansions", "retrieval_expansion", "suggests_retrieval", source_path, "")
    _ingest_list_field(graph, bridge, data, "what_not_to_claim", "overclaim_warning", "warns_against", source_path, "")


def _ingest_assurance_case_card(graph: ConceptGraph, data: dict[str, Any], source_path: str) -> None:
    card = graph.add_node(
        kind="assurance_case",
        label=data.get("artifact_name") or Path(source_path).stem,
        description=_clean(data.get("assurance_role") or data.get("recommended_project_role")),
        source=source_path,
        attrs={
            "artifact_kind": data.get("artifact_kind"),
            "confidence": data.get("confidence"),
            "source_run": data.get("source_run"),
        },
    )
    role = graph.add_node(
        kind="project_hook",
        label=_clean(data.get("recommended_project_role")),
        source=source_path,
    )
    graph.add_edge(card, "recommended_as", role, source_path=source_path)
    claim_layers = data.get("claim_layers") or {}
    if isinstance(claim_layers, dict):
        for layer, claim in claim_layers.items():
            node = graph.add_node(
                kind="claim_layer",
                label=f"{layer}: {_clean(claim, 240)}",
                description=_clean(claim),
                source=source_path,
                attrs={"layer": layer},
            )
            graph.add_edge(card, "has_claim_layer", node, evidence=layer, source_path=source_path)
    _ingest_list_field(graph, card, data, "hidden_witnesses", "hidden_witness", "has_hidden_witness", source_path, "")
    _ingest_list_field(graph, card, data, "cyberphysical_escalation_points", "escalation_point", "has_escalation_point", source_path, "")
    _ingest_list_field(graph, card, data, "physical_world_assumptions", "assumption", "depends_on_assumption", source_path, "")
    _ingest_list_field(graph, card, data, "missing_evidence", "evidence_gap", "missing_evidence", source_path, "")
    _ingest_list_field(graph, card, data, "needed_evidence", "needed_evidence", "needs_evidence", source_path, "")
    _ingest_list_field(graph, card, data, "runtime_monitoring_hooks", "runtime_hook", "has_runtime_hook", source_path, "")
    _ingest_list_field(graph, card, data, "tests_or_metrics", "metric", "supports_metric", source_path, "")
    risk = data.get("risk_portfolio") or {}
    if isinstance(risk, dict):
        role = graph.add_node(kind="risk_portfolio_role", label=_clean(risk.get("portfolio_role")), source=source_path)
        graph.add_edge(card, "has_risk_portfolio_role", role, source_path=source_path)
        for field, kind, relation in [
            ("risky_assets", "risky_asset", "protects_asset"),
            ("risk_factors", "risk_factor", "exposed_to"),
            ("controls_or_hedges", "risk_control", "controlled_by"),
            ("evidence_assets", "evidence_asset", "measured_by"),
            ("residual_risk", "residual_risk", "has_residual_risk"),
        ]:
            _ingest_list_field(graph, card, risk, field, kind, relation, source_path, "")
        exposure = risk.get("exposure_vector") or {}
        if isinstance(exposure, dict):
            for name, level in exposure.items():
                node = graph.add_node(
                    kind="risk_exposure",
                    label=f"{name}: {level}",
                    source=source_path,
                    attrs={"risk_factor": name, "level": level},
                )
                graph.add_edge(card, "has_exposure", node, source_path=source_path)
    _ingest_list_field(graph, card, data, "trust_transfer_questions", "trust_transfer_question", "asks_trust_transfer", source_path, "")
    _ingest_list_field(graph, card, data, "overclaim_warnings", "overclaim_warning", "warns_against", source_path, "")
    _ingest_list_field(graph, card, data, "next_actions", "project_hook", "suggests_next_action", source_path, "")
    for grounding in data.get("course_grounding", []) or []:
        if isinstance(grounding, dict):
            module = grounding.get("module") or grounding.get("module_id") or ""
            paper_label = grounding.get("paper") or grounding.get("filename") or module
            paper = graph.add_node(
                kind="paper",
                label=str(paper_label),
                description=_clean(grounding.get("supports") or grounding.get("rationale")),
                source=source_path,
                module=str(module),
            )
            graph.add_edge(card, "grounded_by", paper, evidence=_clean(grounding.get("supports") or grounding), source_path=source_path)
            if module:
                mod = graph.add_node(kind="module", label=str(module), source=source_path)
                graph.add_edge(card, "maps_to_module", mod, source_path=source_path)
        else:
            node = graph.add_node(kind="citation_anchor", label=_clean(grounding), source=source_path)
            graph.add_edge(card, "grounded_by", node, source_path=source_path)
    for tag in data.get("code_annotation_tags", []) or []:
        if isinstance(tag, dict):
            label = tag.get("tag") or tag.get("name") or _clean(tag)
            desc = tag.get("meaning") or tag.get("use_when") or ""
        else:
            label = _clean(tag)
            desc = ""
        node = graph.add_node(kind="code_tag", label=str(label), description=_clean(desc), source=source_path)
        graph.add_edge(card, "suggests_code_tag", node, evidence=desc, source_path=source_path)


def _ingest_council_run(graph: ConceptGraph, data: dict[str, Any], source_path: str) -> None:
    query_label = _clean(data.get("query") or Path(source_path).parent.name, limit=240)
    run = graph.add_node(
        kind="run",
        label=query_label,
        description=_clean(data.get("final_answer") or data.get("scout_answer") or ""),
        source=source_path,
        attrs={"models": data.get("models")},
    )
    lift = data.get("lift") or {}
    if isinstance(lift, dict):
        _ingest_list_field(graph, run, lift, "lifted_concepts", "concept", "lifts_to", source_path, "")
        _ingest_list_field(graph, run, lift, "threat_models", "failure_mode", "has_threat_model", source_path, "")
        _ingest_list_field(graph, run, lift, "possible_le_cps_instantiations", "project_hook", "has_project_hook", source_path, "")
        _ingest_list_field(graph, run, lift, "what_not_to_overclaim", "overclaim_warning", "warns_against", source_path, "")
    gate = data.get("evidence_gate") or {}
    if isinstance(gate, dict):
        for module in gate.get("recommended_modules", []) or []:
            mod = graph.add_node(kind="module", label=str(module), source=source_path)
            graph.add_edge(run, "retrieved_module", mod, source_path=source_path)
        for rec in gate.get("recommendation_summary", []) or []:
            paper = graph.add_node(
                kind="paper",
                label=f"{rec.get('module_id')}/{rec.get('filename')}",
                description=_clean(rec.get("rationale")),
                source=source_path,
                module=str(rec.get("module_id") or ""),
                score=float(rec.get("relevance_score") or 0.0),
                attrs={"required": rec.get("required"), "module_name": rec.get("module_name")},
            )
            graph.add_edge(run, "retrieved_paper", paper, evidence=rec.get("rationale", ""), source_path=source_path)
        _ingest_list_field(graph, run, gate, "uncovered_lifted_concepts", "concept_gap", "has_gap", source_path, "")
        _ingest_list_field(graph, run, gate, "suggested_expansion_terms", "retrieval_expansion", "suggests_retrieval", source_path, "")
    for card in data.get("matched_bridge_cards", []) or []:
        if isinstance(card, dict):
            card_node = graph.add_node(
                kind="bridge_card",
                label=card.get("title") or card.get("id") or "matched bridge card",
                description=_clean(card.get("abstract_assurance_principle") or card.get("source_context")),
                source=source_path,
            )
            graph.add_edge(run, "matched_bridge_card", card_node, source_path=source_path)


def _ingest_list_field(
    graph: ConceptGraph,
    parent: str,
    data: dict[str, Any],
    field_name: str,
    kind: str,
    relation: str,
    source_path: str,
    module: str,
) -> None:
    value = data.get(field_name)
    if not value:
        return
    if not isinstance(value, list):
        value = [value]
    for item in value:
        label = _clean(item)
        if not label:
            continue
        node = graph.add_node(kind=kind, label=label, source=source_path, module=module)
        graph.add_edge(parent, relation, node, evidence=field_name, source_path=source_path)


def query_graph(graph: ConceptGraph, query: str, *, top_k: int = 20) -> dict[str, Any]:
    q_terms = Counter(_tokens(query))
    scored: list[tuple[float, ConceptNode]] = []
    for node in graph.nodes.values():
        text = f"{node.kind} {node.label} {node.description} {' '.join(node.modules)}"
        terms = Counter(_tokens(text))
        overlap = sum(min(q_terms[t], terms[t]) for t in q_terms)
        if overlap == 0:
            continue
        score = overlap / math.sqrt(max(sum(terms.values()), 1)) + math.log1p(node.score) * 0.08
        scored.append((score, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    top_ids = {node.id for _, node in top}
    edges = [
        edge for edge in graph.edges
        if edge.source in top_ids or edge.target in top_ids
    ][: top_k * 4]
    return {
        "query": query,
        "nodes": [
            {
                "score": round(score, 4),
                **asdict(node),
            }
            for score, node in top
        ],
        "edges": [asdict(edge) for edge in edges],
    }


def render_query_markdown(result: dict[str, Any]) -> str:
    lines = ["# Concept Graph Query", "", f"**Query:** {result['query']}", ""]
    lines.append("## Top Nodes")
    for row in result["nodes"]:
        sources = ", ".join(row.get("sources", [])[:2])
        modules = ", ".join(row.get("modules", [])[:4])
        lines.append(f"- **{row['kind']}** `{row['label']}` score={row['score']}")
        if modules:
            lines.append(f"  - Modules: {modules}")
        if row.get("description"):
            lines.append(f"  - Note: {row['description'][:360]}")
        if sources:
            lines.append(f"  - Source: {sources}")
    lines.append("")
    lines.append("## Nearby Edges")
    for edge in result["edges"][:40]:
        s = edge["source"].split(":", 1)[-1]
        t = edge["target"].split(":", 1)[-1]
        lines.append(f"- `{s}` --{edge['relation']}--> `{t}`")
    lines.append("")
    return "\n".join(lines)


def render_summary_markdown(graph: ConceptGraph, out_path: Path) -> str:
    by_kind = Counter(node.kind for node in graph.nodes.values())
    by_module = Counter(m for node in graph.nodes.values() for m in node.modules if m)
    relations = Counter(edge.relation for edge in graph.edges)
    lines = ["# Assurance Concept Graph", ""]
    lines.append(f"- Nodes: {len(graph.nodes)}")
    lines.append(f"- Edges: {len(graph.edges)}")
    lines.append(f"- JSON: `{out_path.name}`")
    lines.append("")
    lines.append("## Node Types")
    for kind, count in by_kind.most_common():
        lines.append(f"- {kind}: {count}")
    lines.append("")
    lines.append("## Module Coverage")
    for module, count in by_module.most_common(20):
        lines.append(f"- {module}: {count}")
    lines.append("")
    lines.append("## Relation Types")
    for relation, count in relations.most_common():
        lines.append(f"- {relation}: {count}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build concept graph JSON from existing artifacts")
    build.add_argument("--artifact-root", action="append", required=True,
                       help="Artifact directory or JSON file. Repeatable.")
    build.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    build.add_argument("--max-files", type=int)

    query = sub.add_parser("query", help="Query an existing concept graph JSON")
    query.add_argument("--graph", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=20)
    query.add_argument("--out")

    args = parser.parse_args(argv)
    if args.cmd == "build":
        roots = [Path(p) for p in args.artifact_root]
        graph = build_concept_graph(artifact_roots=roots, max_files=args.max_files)
        out_dir = Path(args.out_dir) / f"concept_graph_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "concept_graph.json"
        json_path.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path = out_dir / "concept_graph.md"
        md_path.write_text(render_summary_markdown(graph, json_path), encoding="utf-8")
        print(f"Saved concept graph directory: {out_dir}")
        print(f"Nodes: {len(graph.nodes)} Edges: {len(graph.edges)}")
        return 0
    if args.cmd == "query":
        graph = ConceptGraph.from_dict(json.loads(Path(args.graph).read_text(encoding="utf-8")))
        result = query_graph(graph, args.query, top_k=args.top_k)
        md = render_query_markdown(result)
        if args.out:
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(md)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
