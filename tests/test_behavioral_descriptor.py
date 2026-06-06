from datadiff.behavioral_descriptor import BehavioralDescriptor, compute_behavioral_descriptor
from datadiff.dsl import Case, ColumnSpec, Program, TableData


def _case(seed: int) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("flag", "bool"),
                ],
                [{"id": 1, "g": "a", "flag": True}],
            )
        ],
        Program(
            f"prog-{seed}",
            seed,
            [
                {"op": "filter", "column": "id", "cmp": ">=", "value": 0},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "id", "func": "count", "as": "n"}]},
            ],
        ),
        metadata={"mixed_generator_profile": "discovery_fresh"},
    )


def test_behavioral_descriptor_computes_stable_axes_from_case_and_targets():
    descriptor = compute_behavioral_descriptor(
        _case(1),
        target_keys=["semantic_family:aggregation_cardinality", "capability:op:groupby"],
    )

    axes = dict(descriptor.axis_tuples())
    assert axes["bd_profile"] == "discovery_fresh"
    assert axes["bd_target_class"] == "target:semantic_family_aggregation_cardinality"
    assert axes["bd_type_mix"] == "type:num1_str1_bool1"
    assert axes["bd_op_skeleton"].startswith("op:")
    assert axes["bd_column_count"] in {"cols:2", "cols:3"}
    assert axes["bd_backend_disagreement"] == "pair:none"


def test_behavioral_descriptor_round_trips_dict_payload():
    descriptor = compute_behavioral_descriptor(_case(2), profile_key="common", target_keys=[])
    restored = BehavioralDescriptor.from_dict(descriptor.to_dict())

    assert restored == descriptor
    assert restored.to_dict()["axis_tuples"] == [list(item) for item in descriptor.axis_tuples()]


def test_behavioral_descriptor_uses_pairwise_backend_disagreement_axis():
    case = _case(3)
    case.metadata["disagreement_descriptor"] = {
        "pair_disagrees": [
            {"left": "duckdb", "right": "pandas", "disagrees": True},
            {"left": "pandas", "right": "sqlite", "disagrees": False},
        ],
        "mismatch_class": "value",
    }

    descriptor = compute_behavioral_descriptor(case)
    axes = dict(descriptor.axis_tuples())

    assert axes["bd_backend_disagreement"] == "pair:duckdb_pandas"
    assert descriptor.to_dict()["backend_disagreement_axis"] == "pair:duckdb_pandas"
