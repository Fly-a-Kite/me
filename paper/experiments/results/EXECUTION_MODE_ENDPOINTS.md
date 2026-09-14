# Execution-mode endpoints — status

Date: 2026-09-14.

| Endpoint | Status | Evidence |
| --- | --- | --- |
| Arrow layout: contiguous / sliced / chunked / dictionary | already existed | `pyarrow_backend.supported_physical_layouts` |
| Arrow layout: **large_string**, **run_end** | **added** | `tests/test_pyarrow_layouts.py` (6 layouts agree on the corpus case) |
| Polars eager / lazy / streaming | already existed as targets | `targets.py` (`polars_lazy`, `polars_streaming`) |
| **pandas copy-on-write on/off** | **dropped — no-op** | pandas 3.0.3 always enables CoW; `mode.copy_on_write` is deprecated and has no impact (verified in the frozen env) |

Decision: do not build a pandas CoW toggle. It would add code with zero behavioural
effect on the frozen target. Execution-mode coverage now comes from Polars modes plus the
six Arrow layouts.

Cross-version coverage is separate and already implemented:
`datadiff cross-version-scan --env-pair LEFT->RIGHT`.
