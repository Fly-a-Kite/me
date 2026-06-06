import random

from datadiff.synthesis.lhs_sampler import lhs_schemas, schema_spec_for_seed


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
