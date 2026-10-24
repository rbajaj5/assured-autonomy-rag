"""End-to-end pipeline test with a deterministic offline encoder.

The real system uses sentence-transformers; here we patch in a small
hash-based encoder so we can verify ingest -> index -> retrieve -> analyze
-> compare without network access to HuggingFace.
"""
from __future__ import annotations
import hashlib
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Inject a fake sentence_transformers module BEFORE importing aate.
# This avoids the HF download path entirely.
# ---------------------------------------------------------------------------
class _FakeEncoder:
    """Deterministic 64-dim 'embedding' from rolling SHA256 over tokens.

    Just barely good enough to produce non-degenerate cosine similarities
    that prefer keyword overlap. Not a real semantic embedder.
    """
    EMB_DIM = 64

    def __init__(self, *_args, **_kwargs):
        pass

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.EMB_DIM, dtype=np.float32)
        for tok in text.lower().split():
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            # Map hash bytes onto vector dims, additive
            for i, b in enumerate(h[: self.EMB_DIM]):
                v[i] += (b / 255.0) - 0.5
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


fake_st = types.ModuleType("sentence_transformers")
fake_st.SentenceTransformer = _FakeEncoder  # type: ignore
sys.modules["sentence_transformers"] = fake_st

# ---------------------------------------------------------------------------
from aate import (  # noqa: E402
    ingest_course_corpus, build_index, load_new_item,
    analyze_baseline, analyze_with_rag, analyze_with_rubric,
    compare_modes, get_llm,
)
from aate.compare import render_comparison_markdown  # noqa: E402


def test_full_pipeline():
    corpus_dir = ROOT / "tests" / "mini_corpus"
    chunks = ingest_course_corpus(corpus_dir, verbose=False)
    assert len(chunks) >= 3, f"expected at least 3 chunks, got {len(chunks)}"
    modules = {c.module_id for c in chunks}
    assert {"M03", "M05", "M11"}.issubset(modules), f"missing modules: {modules}"
    print(f"[ingest] {len(chunks)} chunks across {len(modules)} modules ✓")

    index = build_index(chunks, verbose=False)

    # Pasted CISA-style advisory snippet about adversarial ML attacks
    advisory = (
        "CISA Advisory AA26-101 (synthetic). A threat actor was observed "
        "exploiting a deployed image-classification model used for security "
        "screening. The actor crafted adversarial inputs that caused targeted "
        "misclassification at decision boundaries. The agency recommends "
        "monitoring input distributions, applying randomized smoothing where "
        "feasible, and adding a runtime envelope that bounds the policy's "
        "authority on out-of-distribution inputs. The actor's capabilities "
        "included query access to the production model."
    )
    item = load_new_item(advisory)
    assert item.source_kind == "text"
    print(f"[loader] item_kind={item.source_kind} len={len(item.text)} ✓")

    # Should retrieve M11 (adversarial) and M05 (dataset shift) most strongly.
    hits = index.query(item.text, k=4)
    print("[retrieval] top hits:")
    for h in hits:
        print(f"   {h.chunk.citation():<60s} score={h.score:.3f}")
    retrieved_modules = {h.chunk.module_id for h in hits}
    assert "M11" in retrieved_modules, (
        f"expected M11 in top-4, got {retrieved_modules}"
    )

    # Stub LLM exercises the prompt machinery
    llm = get_llm("stub")

    a = analyze_baseline(item, llm)
    b = analyze_with_rag(item, index, llm)
    c = analyze_with_rubric(item, index, llm)

    assert a.mode == "baseline" and a.retrieved == []
    assert b.mode == "rag" and len(b.retrieved) > 0
    assert c.mode == "rag_rubric" and c.rubric_scores is not None
    assert all(1 <= v <= 5 for v in c.rubric_scores.values())
    print("[analyze] all 3 modes returned valid results ✓")
    print(f"   baseline.raw_output starts with: {a.raw_output[:60]!r}")
    print(f"   rag retrieved {len(b.retrieved)} passages")
    print(f"   rag_rubric scores: {c.rubric_scores}")

    # Compare end-to-end
    report = compare_modes(item, index, llm)
    md = render_comparison_markdown(report)
    assert "Retrieved passages" in md
    assert "RUBRIC SELF-SCORES".lower() in md.lower() or "rubric self-scores" in md.lower()
    assert "Baseline" in md and "RAG" in md
    print(f"[compare] markdown report = {len(md)} chars ✓")

    # JSON round-trip
    d = report.to_dict()
    assert "baseline" in d and "rag" in d and "rag_rubric" in d
    print("[compare] json serialization ✓")

    # Persistence round-trip
    save_dir = ROOT / "tests" / ".aate_index_test"
    index.save(save_dir)
    index2 = type(index).load(save_dir)
    hits2 = index2.query(item.text, k=2)
    assert len(hits2) > 0
    print("[index] save/load round-trip ✓")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test_full_pipeline()
