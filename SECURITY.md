# Security Notes

This is a local research prototype. Do not expose its filesystem or provider credentials as an unauthenticated public service. Model output and retrieved documents are untrusted data, not instructions with authority over the application.

Provider calls may transmit query text, retrieved evidence, and report content. Inspect these before enabling API calls. Keep keys in environment variables or a secret manager, never in source, examples, notebooks, logs, issues, or commits.

The original FAISS metadata loader uses pickle. Only load indexes built locally from trusted inputs; untrusted pickle files can execute arbitrary code. A safe interchange format is future work. This repository intentionally distributes no pickle files or prebuilt indexes.

Deterministic text rules and advisory LLM reviewers are not safety certifications. Keep simulations separate from live actuation, private production systems, or consequential deployment decisions.

Report vulnerabilities without credentials, personal data, or working exploit payloads in public issues.
