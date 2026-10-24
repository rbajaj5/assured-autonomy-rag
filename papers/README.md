# Bring Your Own Corpus

The class reading corpus is not redistributed. Place authorized PDFs, text, or Markdown documents in module folders such as `M03_SafetySecurityPrivacy`, `M05_DataSetShift`, `M12_RuntimeAssurance`, or `M14_MultiAgentCooperative`.

Run `python -m aate ingest --corpus papers --index-dir .aate_index` after installing the vector dependencies. Corpus content and generated indexes are ignored by Git. Do not upload downloaded papers simply because they were used in class.

For an offline smoke test without real papers, run `python test_pipeline.py`. Its synthetic mini-corpus lives under `tests/mini_corpus/` and is not historical evidence.
