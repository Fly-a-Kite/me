# Phase 2 public interface freeze

Version: `osc-public-api-freeze-v1`

Authority manifest: `src/datadiff_osc/public_api_freeze.json`

Manifest SHA-256: `669816be0f913def018518ea85d2cc03b517453cc8e5ba764908dc3c3f42920f`

## Frozen boundaries

- Canonical JSON is UTF-8, sorted-key, compact JSON. Floats and bytes use tagged lossless encodings; mappings require string keys; unsupported values fail closed.
- Stable IDs use SHA-256 over `canonical_schema_version`, namespace and payload. Python hash and list position are not identities.
- `VerdictKind` has one source in `datadiff_osc.schemas`; execution status/failure remains a separate structured outcome.
- Unsupported becomes `INAPPLICABLE` only with endpoint/contract-bound capability evidence. Timeout, crash, adapter error and missing result remain structured and map to `INCONCLUSIVE`; none can become pass or unsupported.
- Semantic atoms carry provenance. Seed lineage is typed and stage-keyed. Task identity, dependencies, resources, stable result ordering, evidence envelopes and ledger events are versioned public records.
- `observed` ledger admission requires activation, applicability and observation certificate digests; generated/selected metadata cannot create activation credit.
- Non-screening evidence requires exact comparison. Fresh confirmation and native reproduction are uncached.
- Fresh and regression backend-pair obligations are separately typed collections. The frozen public denominators are fresh `502` and regression `228`; a combined `730` public denominator is forbidden.
- Root-owned `api.py` and the JSON manifest name the unique source for every public symbol. Private lookalike dataclasses are forbidden.

## Files and hashes

```text
97b995ff22eef099ceaab5e22d02b92078549ccb08f3023c63ec02ea32c0eba3  src/datadiff_osc/_canonical.py
db350fe33cd09e096fb8c5618190d25a2c496f8b9dfe0781ed273a6ec603f33d  src/datadiff_osc/schemas.py
54ddc035fa9b948e7c4d41da555cbf035c89916bed51143f16d1cb74e1b47d5c  src/datadiff_osc/api.py
669816be0f913def018518ea85d2cc03b517453cc8e5ba764908dc3c3f42920f  src/datadiff_osc/public_api_freeze.json
5ebd0dca86745a6991f102086e85db03791986dd5c894af5ea35574d32d67726  src/datadiff_osc/__init__.py
18307cd6fa68b292a6d9ddd74f14610fdd757d7a81f4cd9d0394994aeaf8478d  src/datadiff_osc/contract_engine/model.py
867fa8babefcf30bdd25faf192854232a8fdafc390a56eb294c5723d65cccfd0  src/datadiff_osc/semantic_targets/model.py
ae424cd05066996fb293aa0c1aac594635d399ca256071c65afa4c7188c1d37c  src/datadiff_osc/semantic_targets/compiler.py
5bb39a735119f3db7c3afe87997eb13187f03edb6de0d7fbcc834c4e5a3caa12  tests/osc/test_public_api_freeze.py
```

## Validation

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/osc/test_public_api_freeze.py` -> `7 passed`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/osc` -> `73 passed in 0.42s`.
- Public import and canonical envelope byte-for-byte round trip -> pass.
- Existing tracked diff-stat hash remained `341553eede7eb403553383d713f9a803207214388e7b1c0378d78b74445b45d9`; staged paths remain zero.

`twenty_four_hour_run_authorized=false`.
