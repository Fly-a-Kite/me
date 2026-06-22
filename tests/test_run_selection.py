from types import SimpleNamespace

from datadiff import runner as runner_module
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.operation_combo import describe_operation_combo
from datadiff.run_selection import select_iteration_case


def _case(seed: int, op: dict | None = None) -> Case:
    return Case(
        case_id=f"case-{seed}",
        seed=seed,
        tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}, {"x": None}])],
        program=Program(
            f"prog-{seed}",
            seed,
            [op or {"op": "filter", "column": "x", "cmp": ">", "value": 0}],
        ),
    )


def test_select_iteration_case_uses_first_candidate_without_guidance_and_selects_version_pair():
    config = ExperimentConfig(
        target_version="latest",
        fixed_version="fixed",
        enable_metamorphic_oracle=False,
    )
    first = _case(1)
    second = _case(2)

    selected = select_iteration_case(
        candidates=[first, second],
        candidate_meta={},
        config=config,
        config_payload=config.to_dict(),
        feedback=None,
        guidance=None,
        include_online_weight_snapshot=False,
        version_pair_pool=("latest->preview", "latest->fixed"),
        version_pair_pool_metadata={"selected": ["latest->preview", "latest->fixed"]},
        version_pair_context_features=("backend:pandas",),
        target_capabilities=("op:filter",),
        backends=["pandas", "duckdb"],
    )

    assert selected.case is first
    assert selected.guidance_row == {
        "strategy": config.guidance_strategy,
        "score": 0.0,
        "features": [],
        "matched_targets": [],
        "candidate_count": 1,
    }
    assert selected.selected_meta["source"] == "generated"
    assert selected.selected_meta["operation_combo"]["template"]
    assert selected.selected_meta["selected_version_pair"] == "latest->preview"
    assert selected.selected_meta["version_pair_selection"]["strategy"] == "fixed"
    assert selected.selected_meta["version_pair_pool_metadata"] == {
        "selected": ["latest->preview", "latest->fixed"]
    }
    assert selected.effective_config.fixed_version == "preview"
    assert selected.effective_config_payload["fixed_version"] == "preview"
    assert selected.version_pair_id == "latest->preview"
    assert selected.preflight_row["valid"] is True
    assert selected.case_seed == 1
    assert selected.scheduler_elapsed_ms >= 0.0


def test_select_iteration_case_records_guidance_metadata_and_quality_context():
    class FakeDecision:
        def __init__(self, case):
            self.case = case
            operation_combo = dict(describe_operation_combo(case.program.operations))
            operation_combo["template"] = "guided_combo"
            self.analysis = SimpleNamespace(operation_combo=operation_combo)

        def to_dict(self):
            return {
                "score": 3.0,
                "features": ["exploration_objective:null_boundary"],
                "matched_targets": ["join"],
                "candidate_count": 2,
            }

    class FakeGuidance:
        def __init__(self, selected_case):
            self.selected_case = selected_case
            self.calls = []

        def select_case(self, candidates, *, include_online_weight_snapshot=False):
            self.calls.append(
                {
                    "candidates": list(candidates),
                    "include_online_weight_snapshot": include_online_weight_snapshot,
                }
            )
            return FakeDecision(self.selected_case)

    class FakeFeedback:
        def __init__(self):
            self.quality_context_calls = []

        def candidate_quality_context(self, case, *, target_keys=None):
            self.quality_context_calls.append((case, tuple(target_keys or ())))
            return {"archive_candidate": True, "case_id": case.case_id}

    first = _case(10)
    selected_case = _case(11, {"op": "join", "right": "t1", "on": [["x", "x"]], "how": "inner"})
    meta = {
        id(selected_case): {
            "source": "generated",
            "generated_seed": selected_case.seed,
            "seed_lineage": {"depth": 0, "root_seed": selected_case.seed},
            "mutation": {"operator": "generated"},
            "feedback_decision": {"target_keys": ["target:join"]},
            "quality_archive_context": {},
            "operation_combo": {},
            "preflight": {"valid": True, "repaired": False, "fallback_used": False},
            "replay_filter": {"enabled": False},
            "family_saturation_filter": {"enabled": True},
        }
    }
    config = ExperimentConfig(
        guidance_strategy="guided",
        enable_metamorphic_oracle=False,
        target_version="latest",
        fixed_version="fixed",
    )
    guidance = FakeGuidance(selected_case)
    feedback = FakeFeedback()

    selected = select_iteration_case(
        candidates=[first, selected_case],
        candidate_meta=meta,
        config=config,
        config_payload=config.to_dict(),
        feedback=feedback,
        guidance=guidance,
        include_online_weight_snapshot=True,
        version_pair_pool=("latest->fixed",),
        version_pair_pool_metadata={"selected": ["latest->fixed"]},
        version_pair_context_features=("backend:pandas",),
        target_capabilities=("op:join",),
        backends=["pandas", "duckdb"],
    )

    assert selected.case is selected_case
    assert guidance.calls == [
        {
            "candidates": [first, selected_case],
            "include_online_weight_snapshot": True,
        }
    ]
    assert selected.guidance_row["strategy"] == "guided"
    assert selected.guidance_row["score"] == 3.0
    assert selected.selected_meta is meta[id(selected_case)]
    assert selected.selected_meta["operation_combo"]["template"] == "guided_combo"
    assert selected.selected_meta["operation_combo"]["frequency_bucket"]
    assert selected.selected_meta["quality_archive_context"] == {
        "archive_candidate": True,
        "case_id": selected_case.case_id,
    }
    assert selected_case.metadata["quality_archive_context"]["archive_candidate"] is True
    assert feedback.quality_context_calls == [
        (selected_case, ("target:join", "capability:op:join"))
    ]
    assert selected.selected_meta["case_learning_context"]
    assert selected.selected_meta["semantic_objective_selection"]["scope"] == "semantic_objective"
    assert selected.selected_meta["metamorphic_relation_selection"]["strategy"] == "none"
    assert selected.selected_meta["metamorphic_relation_order"] == []


def test_runner_reexports_iteration_selection_helper_for_compatibility():
    assert runner_module.select_iteration_case is select_iteration_case
