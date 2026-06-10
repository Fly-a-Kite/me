from types import SimpleNamespace

from datadiff import runner as runner_module
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.operation_combo import describe_operation_combo
from datadiff.run_row import apply_iteration_row_updates


def _case(seed: int = 1) -> Case:
    return Case(
        case_id=f"case-{seed}",
        seed=seed,
        tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}, {"x": None}])],
        program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
    )


def _row(case: Case, *, behavior: str = "behavior-1", discovery: str = "discovery-1") -> dict:
    descriptor = {
        "feature_tokens": ["mismatch:value", "root:select"],
        "mismatch_class": "value",
        "primary_root_cause": "select",
        "pair_disagrees": [{"left": "duckdb", "right": "pandas", "disagrees": True}],
    }
    fingerprint = {
        "hash": "abc",
        "feature_tokens": ["fp_op:select", "fp_column_count:1"],
    }
    row_case = case.to_dict()
    return {
        "case": row_case,
        "normalized": {
            "duckdb": {"status": "ok", "columns": ["x"], "rows": [{"x": 1}]},
            "pandas": {"status": "ok", "columns": ["x"], "rows": [{"x": 2}]},
        },
        "findings": [],
        "behavior_signature": behavior,
        "discovery_signature": discovery,
        "disagreement_descriptor": descriptor,
        "case_fingerprint": fingerprint,
        "status": "ok",
    }


def _selected_meta(case: Case) -> dict:
    return {
        "source": "generated",
        "seed_lineage": {"root_seed": case.seed, "depth": 0},
        "mutation": {"operator": "generated"},
        "feedback_selection": {"target_keys": ["target:select"]},
        "feedback_decision": {},
        "quality_archive_context": {"archive_candidate": True},
        "generator_profile_selection": {"strategy": "fixed", "profile": "common"},
        "semantic_objective_selection": {"strategy": "fixed", "scope": "semantic_objective", "action": ""},
        "selected_semantic_objective": "",
        "metamorphic_relation_selection": {"strategy": "none", "scope": "metamorphic_relation", "action": ""},
        "selected_metamorphic_relation": "",
        "version_pair_selection": {"strategy": "fixed", "scope": "version_pair", "action": "latest->fixed"},
        "selected_version_pair": "latest->fixed",
        "case_learning_context": ["op:select"],
        "operation_combo": describe_operation_combo(case.program.operations),
        "preflight": {"valid": True, "repaired": False, "fallback_used": False},
        "replay_filter": {"enabled": False},
        "family_saturation_filter": {"enabled": False},
    }


def test_apply_iteration_row_updates_records_metadata_backend_pairs_and_quality_oracles():
    case = _case(1)
    row = _row(case)
    selected_meta = _selected_meta(case)

    update = apply_iteration_row_updates(
        row=row,
        case=case,
        selected_meta=selected_meta,
        preflight_row=selected_meta["preflight"],
        feedback=None,
        config=ExperimentConfig(
            backend_pair_learning_weight=0.0,
            backend_pair_priority_limit=1,
            guidance_strategy="random",
        ),
        guidance_row={"strategy": "random", "matched_targets": [], "features": []},
        guidance_targets=[],
        backend_pair_pool=("duckdb|pandas",),
        version_pair_id="latest->fixed",
        target_capabilities=("op:select",),
        seen=set(),
        signal_seen=set(),
        candidate_seed_start=10,
        candidate_pool=3,
        case_index=7,
        elapsed_s=1.25,
    )

    assert update.behavior_signature == "behavior-1"
    assert update.discovery_signature == "discovery-1"
    assert row["is_new_behavior"] is True
    assert row["signal_new_behavior"] is True
    assert row["candidate_source"] == "generated"
    assert row["selected_generator_profile"] == "common"
    assert row["selected_version_pair"] == "latest->fixed"
    assert row["backend_pair_priority"] == ["duckdb|pandas"]
    assert "op:select" in row["backend_pair_context"]
    assert "fp_op:select" in row["backend_pair_context"]
    assert row["operation_combo"]["template"]
    assert row["candidate_seed_start"] == 10
    assert row["candidate_pool_size"] == 3
    assert row["case_index"] == 7
    assert row["elapsed_s"] == 1.25
    assert {oracle["name"] for oracle in update.quality_oracles} == {
        "mutation",
        "feedback",
        "guidance",
    }
    assert row["quality_oracles"] == update.quality_oracles
    assert row["case"]["metadata"]["disagreement_descriptor"]["mismatch_class"] == "value"
    assert row["case"]["metadata"]["case_fingerprint"]["hash"] == "abc"
    assert row["semantic_contract_lattice"]["schema_version"] == "semantic-contract-lattice-v1"
    assert row["case"]["metadata"]["semantic_contract_lattice"] == row["semantic_contract_lattice"]
    assert case.metadata["semantic_contract_lattice"] == row["semantic_contract_lattice"]
    assert case.metadata["disagreement_descriptor"]["mismatch_class"] == "value"
    assert case.metadata["case_fingerprint"]["hash"] == "abc"


def test_apply_iteration_row_updates_uses_signal_seen_for_coarse_behavior_deduplication():
    case = _case(2)
    first = _row(case, behavior="behavior-a", discovery="discovery-a")
    second = _row(case, behavior="behavior-b", discovery="discovery-b")
    selected_meta = _selected_meta(case)
    seen: set[str] = set()
    signal_seen: set[str] = set()
    kwargs = {
        "case": case,
        "selected_meta": selected_meta,
        "preflight_row": selected_meta["preflight"],
        "feedback": None,
        "config": ExperimentConfig(),
        "guidance_row": {"strategy": "random", "matched_targets": [], "features": []},
        "guidance_targets": [],
        "backend_pair_pool": ("duckdb|pandas",),
        "version_pair_id": "latest->fixed",
        "target_capabilities": (),
        "seen": seen,
        "signal_seen": signal_seen,
        "candidate_seed_start": 1,
        "candidate_pool": 1,
        "case_index": 0,
        "elapsed_s": 0.1,
    }

    first_update = apply_iteration_row_updates(row=first, **kwargs)
    second_update = apply_iteration_row_updates(row=second, **kwargs)

    assert first["is_new_behavior"] is True
    assert second["is_new_behavior"] is True
    assert first_update.signal_signature == second_update.signal_signature
    assert first["signal_new_behavior"] is True
    assert second["signal_new_behavior"] is False


def test_apply_iteration_row_updates_delegates_backend_pair_bandit_ranking():
    class FakeLearning:
        def __init__(self):
            self.bandits = {
                "backend_pair": SimpleNamespace(
                    arms={
                        "duckdb|pandas": SimpleNamespace(pulls=3),
                        "duckdb|sqlite": SimpleNamespace(pulls=3),
                    }
                )
            }

        def rank_top(self, scope, action_ids, *, limit, **kwargs):
            assert scope == "backend_pair"
            assert action_ids == ("duckdb|pandas", "duckdb|sqlite")
            assert limit == 2
            return [
                {"action_id": "duckdb|sqlite", "score": 9.0},
                {"action_id": "duckdb|pandas", "score": 1.0},
            ]

    case = _case(3)
    row = _row(case)
    selected_meta = _selected_meta(case)
    feedback = SimpleNamespace(adaptive_learning=FakeLearning())

    apply_iteration_row_updates(
        row=row,
        case=case,
        selected_meta=selected_meta,
        preflight_row=selected_meta["preflight"],
        feedback=feedback,
        config=ExperimentConfig(backend_pair_learning_weight=1.0, backend_pair_priority_limit=2),
        guidance_row={"strategy": "random", "matched_targets": [], "features": []},
        guidance_targets=[],
        backend_pair_pool=("duckdb|pandas", "duckdb|sqlite"),
        version_pair_id="latest->fixed",
        target_capabilities=(),
        seen=set(),
        signal_seen=set(),
        candidate_seed_start=1,
        candidate_pool=1,
        case_index=0,
        elapsed_s=0.0,
    )

    assert row["backend_pair_selection"]["strategy"] == "contextual_bandit"
    assert row["backend_pair_priority"] == ["duckdb|sqlite", "duckdb|pandas"]


def test_runner_reexports_row_update_helper_for_compatibility():
    assert runner_module.apply_iteration_row_updates is apply_iteration_row_updates
