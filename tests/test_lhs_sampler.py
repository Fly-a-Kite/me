import random

import datadiff.synthesis.lhs_sampler as lhs_sampler_module
from datadiff.synthesis.lhs_sampler import SchemaSpec, lhs_schemas, schema_spec_for_seed


def test_lhs_schemas_cover_each_axis_bins():
    specs = lhs_schemas(64, rnd=random.Random(123))

    assert len(specs) == 64
    assert len({spec.column_count for spec in specs}) >= 6
    assert len({spec.row_count for spec in specs}) >= 48
    assert min(spec.null_density for spec in specs) >= 0.0
    assert max(spec.null_density for spec in specs) <= 0.4
    assert max(spec.null_density for spec in specs) - min(spec.null_density for spec in specs) > 0.30
    assert all(len(spec.type_mix) == spec.column_count for spec in specs)


def test_schema_spec_for_seed_is_deterministic_within_lhs_window():
    first = schema_spec_for_seed(17, sample_count=32)
    second = schema_spec_for_seed(17, sample_count=32)

    assert first == second


def test_schema_spec_for_seed_reuses_lhs_block(monkeypatch):
    lhs_sampler_module._lhs_schema_block.cache_clear()
    calls = []

    def fake_lhs_schemas(n_samples, *, rnd):
        calls.append(n_samples)
        return [
            SchemaSpec(
                column_count=1,
                row_count=index,
                null_density=0.0,
                type_mix=("numeric",),
            )
            for index in range(n_samples)
        ]

    monkeypatch.setattr(lhs_sampler_module, "lhs_schemas", fake_lhs_schemas)

    assert schema_spec_for_seed(0, sample_count=4).row_count == 0
    assert schema_spec_for_seed(3, sample_count=4).row_count == 3
    assert schema_spec_for_seed(4, sample_count=4).row_count == 0

    assert calls == [4, 4]
    lhs_sampler_module._lhs_schema_block.cache_clear()
