# Public Release Validation

Run from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
python test_pipeline.py
python -m aate --help
```

The 95-test historical assurance suite contains four integration tests requiring the privately retained paper corpus and repaired/distilled cards. Those four now have an explicit `ASSURED_RAG_PRIVATE_CORPUS_TESTS=1` gate. No assertions were removed. A skipped test is not a passing test; the public release does not claim to replicate those four integrations.

The independent pipeline smoke uses synthetic source text, a hash encoder, and a stub language model. It covers extraction/chunking, FAISS indexing, retrieval, the three analysis modes, and comparison output without paid calls or downloading model weights. It cannot measure semantic retrieval quality or literature-grounded answer quality.

Tests can create ignored `reports/` files. The adversarial fixtures test rejection, containment, or resistance to instruction injection; they are not operational instructions or endorsements.

See `reports/public_validation.json` for the actual release check results and explicit omissions. Historical model-specific evaluation outputs and full-course-corpus benchmarks are not bundled or rerun for this release.
