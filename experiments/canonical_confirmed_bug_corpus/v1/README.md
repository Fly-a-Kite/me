# Canonical Confirmed-Bug Corpus v1

This directory is the P3.3 versioned ground-truth corpus for recall,
localization, regression, and reducer evaluation.

- `manifest.json` assigns one stable root ID per independently confirmed root,
  merges aliases/signatures, freezes affected/fixed targets, and records open
  fixed-source validation gaps.
- `cases/` stores one minimal semantic DSL case per confirmed root plus the
  submitted-but-not-yet-confirmed DataFusion #23534 candidate.
- `native_reproducers.py` imports no project code and emits a uniform
  `bug_present` Boolean for every root.

Validate structure and checksums:

```bash
.venv/bin/python scripts/validate_canonical_bug_corpus.py
```

Replay all registered current-version observations, including the pending root:

```bash
.venv/bin/python scripts/validate_canonical_bug_corpus.py \
  --execute-current \
  --include-pending
```

The corpus is structurally frozen, but P3.3 is not yet exit-ready: the merged
fixed source revisions for DataFusion negative-zero, Arrow sliced Boolean
aggregation, and DuckDB join-filter pushdown still require a local source build
or the first released package containing each fix.
