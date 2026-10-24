"""Map the reading list with paper-level embeddings and PCA-style reduction.

This is a cheap triage layer before expensive LLM digestion. It averages the
existing chunk embeddings into paper embeddings, projects papers into a small
space with SVD/PCA, clusters them, and selects "governing" papers for a query
or project direction.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUT_DIR = Path("reading_map_runs")


def _load_index(index_dir: Path) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    with (index_dir / "chunks.pkl").open("rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    return chunks, embeddings, config


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def _paper_key(chunk: Any) -> str:
    return getattr(chunk, "relpath", None) or getattr(chunk, "filename", "unknown")


def build_paper_matrix(index_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    chunks, embeddings, config = _load_index(index_dir)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        groups[_paper_key(chunk)].append(i)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for relpath, idxs in sorted(groups.items()):
        first = chunks[idxs[0]]
        vec = embeddings[idxs].mean(axis=0)
        rows.append({
            "relpath": relpath,
            "filename": getattr(first, "filename", Path(relpath).name),
            "module_id": getattr(first, "module_id", ""),
            "module_name": getattr(first, "module_name", ""),
            "chunk_count": len(idxs),
            "pages_sampled": sorted({int(getattr(chunks[i], "page", 0)) for i in idxs if int(getattr(chunks[i], "page", 0)) > 0})[:8],
        })
        vectors.append(vec)
    mat = _l2_normalize(np.vstack(vectors).astype(np.float32))
    return rows, mat, config


def pca_project(mat: np.ndarray, dims: int = 3) -> tuple[np.ndarray, list[float]]:
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:dims].T
    eig = s ** 2
    explained = (eig[:dims] / max(eig.sum(), 1e-12)).tolist()
    return coords, explained


def kmeans(mat: np.ndarray, k: int, *, seed: int = 13, iters: int = 80) -> np.ndarray:
    n = mat.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(0, n))]
    dist = np.full(n, np.inf)
    for _ in range(1, k):
        dist = np.minimum(dist, ((mat - mat[centers[-1]]) ** 2).sum(axis=1))
        probs = dist / max(dist.sum(), 1e-12)
        centers.append(int(rng.choice(n, p=probs)))
    centroids = mat[centers].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((mat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            members = mat[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return labels


def _embed_query(query: str, config: dict[str, Any]) -> np.ndarray | None:
    if not query.strip():
        return None
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config["embedding_model"])
    q = model.encode([query], convert_to_numpy=True).astype(np.float32)
    return _l2_normalize(q)[0]


def _cluster_medoids(mat: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    medoids: dict[int, int] = {}
    for label in sorted(set(int(x) for x in labels)):
        idxs = np.where(labels == label)[0]
        centroid = mat[idxs].mean(axis=0)
        sims = mat[idxs] @ centroid
        medoids[label] = int(idxs[int(np.argmax(sims))])
    return medoids


def build_reading_map(args: argparse.Namespace) -> int:
    rows, mat, config = build_paper_matrix(Path(args.index_dir))
    coords, explained = pca_project(mat, dims=args.dims)
    labels = kmeans(mat, args.clusters)
    centroid = _l2_normalize(mat.mean(axis=0, keepdims=True))[0]
    centrality = mat @ centroid
    qvec = _embed_query(args.query or "", config)
    query_scores = mat @ qvec if qvec is not None else np.zeros(mat.shape[0], dtype=np.float32)
    medoids = _cluster_medoids(mat, labels)

    for i, row in enumerate(rows):
        row["cluster"] = int(labels[i])
        row["centrality"] = float(centrality[i])
        row["query_score"] = float(query_scores[i])
        for d in range(args.dims):
            row[f"pc{d + 1}"] = float(coords[i, d])
        row["is_cluster_medoid"] = i in set(medoids.values())

    query_rank = sorted(range(len(rows)), key=lambda i: query_scores[i], reverse=True)
    central_rank = sorted(range(len(rows)), key=lambda i: centrality[i], reverse=True)
    selected: list[int] = []
    for pool in [query_rank[: args.top_query], central_rank[: args.top_central], list(medoids.values())]:
        for idx in pool:
            if idx not in selected:
                selected.append(idx)
    for i, row in enumerate(rows):
        row["selected_for_deep_read"] = i in selected

    out_dir = Path(args.out_dir) / f"reading_map_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reading_map.json").write_text(
        json.dumps({
            "query": args.query,
            "index_dir": args.index_dir,
            "embedding_model": config.get("embedding_model"),
            "paper_count": len(rows),
            "dims": args.dims,
            "clusters": args.clusters,
            "explained_variance": explained,
            "papers": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "reading_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = render_map_markdown(rows, explained, args)
    (out_dir / "reading_map.md").write_text(md, encoding="utf-8")
    print(f"Saved reading map directory: {out_dir}")
    print(f"Papers: {len(rows)} Selected: {sum(1 for r in rows if r['selected_for_deep_read'])}")
    return 0


def render_map_markdown(rows: list[dict[str, Any]], explained: list[float], args: argparse.Namespace) -> str:
    lines = [
        "# Reading List Map",
        "",
        f"- Query: {args.query or '(none)'}",
        f"- Papers: {len(rows)}",
        f"- PCA explained variance: {', '.join(f'{x:.3f}' for x in explained)}",
        "",
        "## Selected For Deep Read",
    ]
    selected = [r for r in rows if r["selected_for_deep_read"]]
    selected.sort(key=lambda r: (r["query_score"], r["centrality"]), reverse=True)
    for row in selected:
        lines.append(
            f"- **{row['module_id']} / {row['filename']}** "
            f"cluster={row['cluster']} query={row['query_score']:.3f} centrality={row['centrality']:.3f}"
        )
    lines.extend(["", "## Cluster Medoids"])
    for row in sorted([r for r in rows if r["is_cluster_medoid"]], key=lambda r: r["cluster"]):
        lines.append(
            f"- Cluster {row['cluster']}: **{row['module_id']} / {row['filename']}** "
            f"centrality={row['centrality']:.3f}"
        )
    lines.extend(["", "## Top Query-Relevant Papers"])
    for row in sorted(rows, key=lambda r: r["query_score"], reverse=True)[:20]:
        lines.append(
            f"- **{row['module_id']} / {row['filename']}** "
            f"query={row['query_score']:.3f} cluster={row['cluster']}"
        )
    lines.extend(["", "## Most Central Papers"])
    for row in sorted(rows, key=lambda r: r["centrality"], reverse=True)[:20]:
        lines.append(
            f"- **{row['module_id']} / {row['filename']}** "
            f"centrality={row['centrality']:.3f} cluster={row['cluster']}"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=".aate_index")
    parser.add_argument("--query", default=(
        "sparse-test trust transfer for autonomous driving; cyber-physical escalation; "
        "scenario coverage; runtime assurance; safe control; dataset shift; adversarial perception"
    ))
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--top-query", type=int, default=14)
    parser.add_argument("--top-central", type=int, default=8)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return build_reading_map(args)


if __name__ == "__main__":
    raise SystemExit(main())
