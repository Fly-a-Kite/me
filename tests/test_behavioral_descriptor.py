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


def test_semantic_plan_descriptor_adds_interaction_plan_layout_and_cold_axes():
    case = _case(4)
    case.metadata["interaction_descriptor"] = {
        "tokens": [
            {
                "category": "semantic_physical",
                "components": ["backend=datafusion", "operator=sort", "op=limit"],
            },
            {
                "category": "plan_state",
                "components": [
                    "backend=datafusion",
                    "mode=query_engine",
                    "kind=physical",
                    "operators=limit",
                ],
            },
            {
                "category": "plan_edge",
                "components": [
                    "backend=datafusion",
                    "edge=optimized_logical->physical",
                    "changed=true",
                    "added=none",
                    "removed=sort",
                ],
            },
            {
                "category": "order_required_plan_sort_loss",
                "components": ["backend=datafusion", "logical_sort_count=1"],
            },
        ]
    }
    case.metadata["input_layouts"] = {
        "t0": {
            "representation": "chunked",
            "chunk_count": 3,
            "dictionary_columns": ["g"],
            "attributes": {"slice_offset": 1},
        }
    }

    legacy = compute_behavioral_descriptor(case, mode="legacy_qd")
    semantic_plan = compute_behavioral_descriptor(case, mode="semantic_plan_qd")
    axes = dict(semantic_plan.axis_tuples())

    assert "bd_interaction" not in dict(legacy.axis_tuples())
    assert axes["bd_interaction"].startswith("interaction:")
    assert axes["bd_plan"].startswith("plan:")
    assert axes["bd_layout"].startswith("layout:")
    assert axes["bd_cold_stratum"].startswith("cold:")
    assert BehavioralDescriptor.from_dict(semantic_plan.to_dict()) == semantic_plan


def test_plan_axis_distinguishes_logical_to_physical_edge_changes():
    first = _case(5)
    second = _case(6)
    first.metadata["interaction_descriptor"] = {
        "tokens": [
            {
                "category": "plan_edge",
                "components": [
                    "backend=datafusion",
                    "edge=optimized_logical->physical",
                    "changed=true",
                    "added=none",
                    "removed=sort",
                ],
            }
        ]
    }
    second.metadata["interaction_descriptor"] = {
        "tokens": [
            {
                "category": "plan_edge",
                "components": [
                    "backend=datafusion",
                    "edge=optimized_logical->physical",
                    "changed=false",
                    "added=none",
                    "removed=none",
                ],
            }
        ]
    }

    first_axis = compute_behavioral_descriptor(first, mode="semantic_plan_qd").plan_axis
    second_axis = compute_behavioral_descriptor(second, mode="semantic_plan_qd").plan_axis

    assert first_axis != second_axis
