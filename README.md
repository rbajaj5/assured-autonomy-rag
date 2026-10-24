# Assured Autonomy Agentic RAG

A research toolkit developed by Ravi Bajaj with AI-assisted implementation during and after Johns Hopkins Assured Autonomy coursework. It connects a locally supplied literature corpus to retrieval, structured paper cards, code-reading workflows, proposal assessment, and explicit evidence checks.

**This public release contains the application code, regression tests, and synthetic examples. It does not contain the class's licensed readings, course discussions, private indexes, saved conversations, or API keys.** It is not an official JHU product, a hosted ChatGPT Custom GPT, or a certified safety monitor.

## Videos: The Companion Simulation Project

See the research applications alongside this literature toolkit. **These are companion simulation demonstrations, not recordings of the RAG or evidence that it controlled or validated the systems.** Click a preview to open/download its MP4.

| 3D formation and runtime assurance | Synchronized covariance and allocation |
| --- | --- |
| [![Sensor uncertainty and runtime assurance in a synthetic 3D formation](https://raw.githubusercontent.com/rbajaj5/delay-aware-runtime-assurance-lab/main/docs/media/sensor.jpg)](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/videos-2026-10-24/sensor_uncertainty_rta_comparison_narrated.mp4) | [![Synthetic basketball with moving covariance metrics and RTA HOLD](https://raw.githubusercontent.com/rbajaj5/delay-aware-runtime-assurance-lab/main/docs/media/portfolio.jpg)](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/videos-2026-10-24/portfolio_rta_synchronized.mp4) |
| **[MP4: narrated 3D comparison](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/videos-2026-10-24/sensor_uncertainty_rta_comparison_narrated.mp4)** · 0:22 | **[MP4: covariance and RTA](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/videos-2026-10-24/portfolio_rta_synchronized.mp4)** · 1:30 · silent |

**[Full video gallery](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab#videos-and-mp4-demonstrations)** includes the narrated infrastructure-planning dashboard, basketball conditioning, bridge fleets, landing coordination, and Hex-like delay examples. **[Download all eight MP4s](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/tag/videos-2026-10-24).** Original media hashes, methods, and interpretation limits accompany the gallery. Playback versus download depends on the browser.

## Two related interfaces

- `aate`: Assured Autonomy Transfer Evaluator. Compare baseline, RAG, and RAG-plus-rubric analysis; navigate modules; retrieve passages; build and audit paper cards; connect literature to code; run iterative and multi-reviewer research workflows.
- `assured_rag`: a later proposal-review and claim/evidence layer. It records deterministic decisions and citation traces. Optional external LLM reviewers are advisory and must not override the deterministic result. This is a text-review prototype, not a physical runtime safety controller.

The associated simulation project is [Delay-Aware Runtime Assurance Lab](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab). Retrieval quality and generated prose are not substitutes for that project's experimental evidence.

## Start offline without keys

Use Python 3.11+ and run commands from this repository's root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m assured_rag.evaluate tests/fixtures/proposals/safe_complete_offline_log_replay.txt --corpus-dir docs/safe_assurance_evaluator.md --no-embeddings --format markdown
```

That command uses local keyword retrieval and deterministic rules, not an LLM API. An `approve` result means only that the textual proposal passes this heuristic rubric. It is not independent evidence of containment, implementation correctness, or deployment safety.

## Test ingestion through comparison

```bash
python -m pip install -r requirements-test.txt
python test_pipeline.py
python -m unittest discover -s tests -p "test_*.py" -v
```

The pipeline test uses an explicitly synthetic three-document corpus, a deterministic hash encoder, and a stub LLM. It checks plumbing, not semantic performance. Four retained corpus-integration tests are skipped unless `ASSURED_RAG_PRIVATE_CORPUS_TESTS=1`; they require privately retained PDFs and repaired/distilled cards. Their assertions remain intact. See [validation scope](docs/VALIDATION.md).

## Use your own permitted corpus

```bash
python -m pip install -r requirements.txt
python -m aate --help
python -m aate ingest --corpus papers --index-dir .aate_index
python -m aate agent "What evidence distinguishes runtime assurance from formal verification?" --papers-dir papers --index-dir .aate_index --provider openai --model YOUR_AVAILABLE_MODEL
```

`papers/README.md` describes the directory layout. Add only documents you are authorized to process. The vector path may download embedding/reranking models. External model calls can transmit your query and retrieved passages to the selected provider and incur charges. Inspect prompts and corpus permissions first. Set provider keys in environment variables; never commit them. Legacy model names are retained in source defaults as historical configuration, not guarantees of current availability. Explicitly choose your provider/model.

The optional training, paper acquisition, and multi-provider sweep functions are available in source. They are not automatically invoked or validated by the offline demonstration. The full requirements file includes optional training packages; use the smaller core/test files for the offline path.

## Architecture and research methods

1. `ingest.py`, `indexing.py`, `hybrid_retriever.py`: document extraction, provenance-bearing chunks, vector and keyword retrieval, optional reranking.
2. `agentic_rag.py`, `syllabus.py`, `prompt_router.py`: module navigation and scoped retrieval.
3. `card_curator.py`, `assurance_case_cards.py`, `code_reading_upgrade.py`: structured literature and code-reading artifacts.
4. `compare.py`, `evaluate_agent.py`, `assured_rag/`: controlled comparisons, schema/claim checks, review disagreement, coverage and ablation diagnostics.
5. `tests/`: safe, incomplete, adverse and prompt-injection fixtures. Adverse examples are rejection tests, not requests to carry out their content.

The `aate` facade loads the original flat modules; keep that layout when cloning. `AATE_PROJECT_ROOT` can override project paths. Run the CLI from the checkout root for relative input/output defaults.

## Evidence and limitations

This release does not claim that RAG universally improves answer quality, that an LLM judge is ground truth, or that text-level approval proves safety. Retrieval can miss relevant passages; cards can misstate source scope; provider/model changes can alter results; generated advice needs review. No paid benchmark or historical corpus replication was performed to prepare this release.

The export is additive: the original local project and its datasets remain unchanged. [Source provenance](docs/source_export.json) records the selected original file hashes. Public-copy changes are limited to path portability, documentation, examples, and explicit handling of non-redistributable test prerequisites.

## Data and security

See [data boundaries](docs/DATA_BOUNDARIES.md) and [security notes](SECURITY.md). Never load an untrusted serialized index: the historical index implementation uses Python pickle for local metadata. No serialized indexes are distributed here.

Public visibility does not automatically grant a software license. This initial source release does not add a new license; obtain permission before redistribution beyond applicable rights. Third-party libraries and documents retain their own licenses. No warranty or operational safety assurance is provided.
