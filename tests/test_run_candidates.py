from types import SimpleNamespace

from datadiff import runner as runner_module
from datadiff import run_candidates as candidates_module
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.goal_first import (
    SEMANTIC_WITNESS_EPOCH_SPAN,
    semantic_witness_data_pattern_lane,
    semantic_witness_epoch_key,
    semantic_witness_palette_construction_seed,
)
from datadiff.run_candidates import (
    _generate_case_with_optional_schema,
    _known_replay_source_filter_reason,
    _replay_bug_filter_reason,
    generate_candidate_batch,
)


def _case(seed: int, *, profile: str = "common") -> Case:
    return Case(
        case_id=f"case-{seed}-{profile}",
        seed=seed,
        tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
        metadata={"generator_profile": profile},
    )


def test_generate_candidate_batch_builds_pool_and_metadata_without_filters():
    calls: list[dict] = []

    def fake_generate_case(seed, *, type_aware=True, profile="common", schema_spec=None):
        calls.append(
            {
                "seed": seed,
                "type_aware": type_aware,
                "profile": profile,
                "schema_spec": schema_spec,
            }
        )
        return _case(seed, profile=profile)

    batch = generate_candidate_batch(
        seed_start=10,
        candidate_pool=2,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=True,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=("context",),
        target_capabilities=("op:select",),
        schema_spec_for_seed=lambda seed: {"seed": seed},
        generate_case_fn=fake_generate_case,
        replay_filter_fn=lambda case, config: "",
    )

    assert [case.seed for case in batch.candidates] == [10, 11]
    assert batch.next_seed == 12
    assert batch.counter_deltas() == {
        "replay_filtered_candidates": 0,
        "replay_fallback_candidates": 0,
        "saturated_family_filtered_candidates": 0,
        "saturated_family_fallback_candidates": 0,
    }
    assert [call["schema_spec"] for call in calls] == [{"seed": 10}, {"seed": 11}]
    first_meta = batch.candidate_meta[id(batch.candidates[0])]
    assert first_meta["source"] == "generated"
    assert first_meta["generated_seed"] == 10
    assert first_meta["seed_lineage"]["depth"] == 0
    assert first_meta["generator_profile_selection"]["profile"] == "common"
    assert first_meta["replay_filter"]["enabled"] is False
    assert first_meta["family_saturation_filter"]["enabled"] is False


def test_goal_first_candidate_metadata_carries_semantic_activation_evidence():
    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(
            method_arm="p5_goal_first",
            enable_replay_bug=True,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        replay_filter_fn=lambda case, config: "",
    )

    candidate = batch.candidates[0]
    meta = batch.candidate_meta[id(candidate)]
    assert meta["goal_first_generation"]["constructible"] is True
    assert meta["semantic_activation"]["goal_id"] == "nullable_membership_join"
    assert meta["semantic_activation"]["evaluation_status"] == "activated"
    assert candidate.metadata["semantic_activation"] == meta["semantic_activation"]


def test_semantic_witness_arm_enforces_activation_in_candidate_pipeline():
    batch = generate_candidate_batch(
        seed_start=0,
        candidate_pool=6,
        config=ExperimentConfig(
            method_arm="p8_semantic_witness_v1",
            enable_replay_bug=True,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        replay_filter_fn=lambda case, config: "",
    )

    assert len(batch.candidates) == 6
    for candidate in batch.candidates:
        meta = batch.candidate_meta[id(candidate)]
        assert candidate.metadata["generation_mode"] == "goal_first_witness"
        assert meta["semantic_activation"]["evaluation_status"] == "activated"
        assert meta["goal_first_generation"]["activation_repair"]["satisfied"] is True


def test_semantic_witness_v2_arm_preserves_activation_and_rotates_variants_by_epoch():
    variants = set()
    variant_patterns = set()
    for epoch_index in range(9):
        epoch_seed_start = epoch_index * SEMANTIC_WITNESS_EPOCH_SPAN
        epoch_key = semantic_witness_epoch_key(epoch_seed_start)
        batch = generate_candidate_batch(
            seed_start=epoch_seed_start,
            candidate_pool=36,
            config=ExperimentConfig(
                method_arm="p8_semantic_witness_v2",
                enable_replay_bug=True,
            ),
            feedback=None,
            guidance=None,
            generator_profile_pool=("common",),
            generator_profile_pool_metadata={"selected": ["common"]},
            generator_profile_context_features=(),
            target_capabilities=(),
            schema_spec_for_seed=lambda seed: None,
            replay_filter_fn=lambda case, config: "",
        )

        assert len(batch.candidates) == 36
        epoch_variants = set()
        cache_cells = {}
        for candidate in batch.candidates:
            meta = batch.candidate_meta[id(candidate)]
            assert candidate.metadata["generation_mode"] == "goal_first_witness_v2"
            assert meta["semantic_activation"]["evaluation_status"] == "activated"
            variant = candidate.metadata["goal_builder_variant"]
            assert variant["diversity_preserving"] is True
            expected_construction_seed = semantic_witness_palette_construction_seed(
                semantic_witness_data_pattern_lane(epoch_seed_start),
                candidate.metadata["goal_id"],
                variant["selection_index"],
            )
            assert variant["construction_seed"] == expected_construction_seed
            epoch_trace = candidate.metadata["goal_first_generation"]["diversity_epoch"]
            assert epoch_trace == {
                "enabled": True,
                "epoch_key": epoch_key,
                "seed_span": SEMANTIC_WITNESS_EPOCH_SPAN,
                "epoch_lane": epoch_key % 3,
                "data_pattern_lane": semantic_witness_data_pattern_lane(
                    epoch_seed_start
                ),
                "palette_cell_index": epoch_trace["palette_cell_index"],
                "palette_size": 12,
                "registered_palette_size": 18,
                "active_variant_indexes": [epoch_key % 3, (epoch_key + 1) % 3],
            }
            assert 0 <= epoch_trace["palette_cell_index"] < 18
            epoch_variants.add(variant["variant_id"])
            variants.add(variant["variant_id"])
            pattern_index = str(
                variant["data_pattern"]["pattern_index"]
            )
            variant_patterns.add(
                (
                    candidate.metadata["goal_id"],
                    variant["variant_id"],
                    pattern_index,
                )
            )
            cell_key = (
                candidate.metadata["goal_id"],
                variant["variant_id"],
                pattern_index,
            )
            signature = (
                [table.to_dict() for table in candidate.tables],
                [operation.to_dict() for operation in candidate.program.operations],
            )
            previous = cache_cells.setdefault(cell_key, signature)
            assert signature == previous
        assert len(epoch_variants) == 12
        assert len(cache_cells) == 12
        by_cell = {}
        for candidate in batch.candidates:
            cell = (
                candidate.metadata["goal_id"],
                candidate.metadata["goal_builder_variant"]["variant_id"],
                str(
                    candidate.metadata["goal_builder_variant"]["data_pattern"][
                        "pattern_index"
                    ]
                ),
            )
            signature = (
                [table.to_dict() for table in candidate.tables],
                [operation.to_dict() for operation in candidate.program.operations],
            )
            previous = by_cell.setdefault(cell, signature)
            assert signature == previous
    assert len(variants) == 18
    assert len(variant_patterns) == 54


def test_semantic_witness_v3_keeps_bounded_cache_cells_and_routes_root_guided_variants():
    variants = set()
    variant_patterns = set()
    root_guided = set()
    for epoch_index in range(9):
        epoch_seed_start = epoch_index * SEMANTIC_WITNESS_EPOCH_SPAN
        batch = generate_candidate_batch(
            seed_start=epoch_seed_start,
            candidate_pool=12,
            config=ExperimentConfig(
                method_arm="p8_semantic_witness_v3",
                enable_replay_bug=True,
            ),
            feedback=None,
            guidance=None,
            generator_profile_pool=("common",),
            generator_profile_pool_metadata={"selected": ["common"]},
            generator_profile_context_features=(),
            target_capabilities=(),
            schema_spec_for_seed=lambda seed: None,
            replay_filter_fn=lambda case, config: "",
        )

        assert len(batch.candidates) == 12
        cache_cells = {}
        for candidate in batch.candidates:
            meta = batch.candidate_meta[id(candidate)]
            assert candidate.metadata["generation_mode"] == "goal_first_witness_v3"
            assert meta["semantic_activation"]["evaluation_status"] == "activated"
            variant = candidate.metadata["goal_builder_variant"]
            assert variant["variant_family"] == "v3"
            expected_construction_seed = semantic_witness_palette_construction_seed(
                semantic_witness_data_pattern_lane(epoch_seed_start),
                candidate.metadata["goal_id"],
                variant["selection_index"],
                variant_family="v3",
            )
            assert variant["construction_seed"] == expected_construction_seed
            pattern_index = str(variant["data_pattern"]["pattern_index"])
            cell = (
                candidate.metadata["goal_id"],
                variant["variant_id"],
                pattern_index,
            )
            signature = (
                [table.to_dict() for table in candidate.tables],
                [operation.to_dict() for operation in candidate.program.operations],
            )
            previous = cache_cells.setdefault(cell, signature)
            assert signature == previous
            variants.add(variant["variant_id"])
            variant_patterns.add(cell)
            if variant.get("root_guidance"):
                root_guided.add(variant["variant_id"])
        assert len(cache_cells) == 12

    assert len(variants) == 18
    assert len(variant_patterns) == 54
    assert root_guided == {"join_cut_membership", "ordered_cut_membership"}


def test_generate_candidate_batch_reserves_distinct_profiles_within_one_pool():
    batch = generate_candidate_batch(
        seed_start=30,
        candidate_pool=3,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=True,
            generator_profile_learning_weight=0.0,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("lane_a", "lane_b", "lane_c"),
        generator_profile_pool_metadata={"selected": ["lane_a", "lane_b", "lane_c"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(
            seed,
            profile=kwargs.get("profile", "common"),
        ),
        replay_filter_fn=lambda case, config: "",
    )

    selected_profiles = {
        batch.candidate_meta[id(case)]["generator_profile_selection"]["profile"]
        for case in batch.candidates
    }
    assert selected_profiles == {"lane_a", "lane_b", "lane_c"}


def test_replay_rejection_advances_to_another_profile_without_common_fallback():
    batch = generate_candidate_batch(
        seed_start=40,
        candidate_pool=1,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=False,
            generator_profile_learning_weight=0.0,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("known_replay", "fresh_lane"),
        generator_profile_pool_metadata={"selected": ["known_replay", "fresh_lane"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(
            seed,
            profile=kwargs.get("profile", "common"),
        ),
        replay_filter_fn=lambda case, config: (
            "known_replay" if case.metadata.get("generator_profile") == "known_replay" else ""
        ),
    )

    meta = batch.candidate_meta[id(batch.candidates[0])]
    assert meta["generator_profile_selection"]["profile"] == "fresh_lane"
    assert batch.replay_filtered_candidates == 1
    assert batch.replay_fallback_candidates == 0


def test_generate_candidate_batch_uses_feedback_metadata_and_quality_context():
    class FakeFeedback:
        def __init__(self):
            self.last_candidate_source = "feedback_mutation"
            self.last_candidate_metadata = {}
            self.quality_target_keys = None

        def select_case(self, seed, generated):
            self.last_candidate_metadata = {
                "seed_lineage": {
                    "root_seed": generated.seed,
                    "parent_seed": 1,
                    "parent_case_id": "case-parent",
                    "mutation_seed": seed,
                    "depth": 1,
                },
                "mutation": {"operator": "value", "detail": "x", "changed": True},
                "feedback_decision": {"target_keys": ["semantic_family:cast_semantics"]},
            }
            return generated

        def candidate_quality_context(self, case, *, target_keys=None):
            self.quality_target_keys = list(target_keys or [])
            return {"archive_known": True, "target_keys": list(target_keys or [])}

    feedback = FakeFeedback()

    batch = generate_candidate_batch(
        seed_start=20,
        candidate_pool=1,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=True,
        ),
        feedback=feedback,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=("op:select",),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=lambda case, config: "",
    )

    meta = batch.candidate_meta[id(batch.candidates[0])]
    assert meta["source"] == "feedback_mutation"
    assert meta["seed_lineage"]["parent_case_id"] == "case-parent"
    assert meta["mutation"]["operator"] == "value"
    assert meta["quality_archive_context"]["archive_known"] is True
    assert "semantic_family:cast_semantics" in feedback.quality_target_keys
    assert "capability:op:select" in feedback.quality_target_keys
    assert batch.candidates[0].metadata["quality_archive_context"] == meta["quality_archive_context"]


def test_generate_candidate_batch_can_emit_semantic_metamorphic_source(monkeypatch):
    parent = _case(24)
    variant = _case(25)
    monkeypatch.setattr(
        candidates_module,
        "all_metamorphic_variants",
        lambda case: [SimpleNamespace(name="row_permutation:reverse", relation="row_permutation", case=variant)],
    )

    batch = generate_candidate_batch(
        seed_start=24,
        candidate_pool=1,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=True,
            enable_semantic_metamorphic_source=True,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: parent,
        replay_filter_fn=lambda case, config: "",
    )

    selected = batch.candidates[0]
    meta = batch.candidate_meta[id(selected)]
    assert selected.case_id == variant.case_id
    assert meta["source"] == "semantic_metamorphic_mutation"
    assert meta["mutation"]["relation"] == "row_permutation"
    assert meta["seed_lineage"]["parent_case_id"] == parent.case_id


def test_generate_candidate_batch_can_label_known_regression_replay(monkeypatch):
    monkeypatch.setattr(candidates_module, "case_discovery_origin", lambda case: "issue_replay")

    batch = generate_candidate_batch(
        seed_start=25,
        candidate_pool=1,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=True,
            enable_known_regression_source=True,
        ),
        feedback=None,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed),
        replay_filter_fn=lambda case, config: "",
    )

    meta = batch.candidate_meta[id(batch.candidates[0])]
    assert meta["source"] == "known_regression_replay"
    assert meta["mutation"]["detail"] == "known_issue_replay"


def test_generate_candidate_batch_consumes_feedback_seed_energy_batch():
    class BatchFeedback:
        def __init__(self):
            self.last_candidate_source = "generated"
            self.last_candidate_metadata = {}
            self.pending = []
            self.batch_calls = []

        def select_case_batch(self, seed, generated, *, max_batch=None, enqueue_remaining=False):
            self.batch_calls.append(
                {
                    "seed": seed,
                    "max_batch": max_batch,
                    "enqueue_remaining": enqueue_remaining,
                }
            )
            if self.pending:
                selected, metadata = self.pending.pop(0)
                self.last_candidate_source = metadata["source"]
                self.last_candidate_metadata = metadata
                return [selected]
            batch = []
            for offset in range(min(3, int(max_batch or 1))):
                candidate = _case(seed + offset + 100)
                metadata = {
                    "source": "feedback_mutation",
                    "seed_lineage": {
                        "root_seed": generated.seed,
                        "parent_index": 0,
                        "parent_case_id": "case-parent",
                        "mutation_seed": seed + offset,
                        "depth": 1,
                    },
                    "mutation": {
                        "operator": "value",
                        "detail": f"value:{offset}",
                        "changed": True,
                    },
                    "feedback_decision": {
                        "parent_index": 0,
                        "selected_operator": "value",
                    },
                }
                batch.append((candidate, metadata))
            self.pending.extend(batch[1:] if enqueue_remaining else [])
            selected, metadata = batch[0]
            self.last_candidate_source = metadata["source"]
            self.last_candidate_metadata = metadata
            return [candidate for candidate, _ in batch]

        def pop_pending_candidate(self):
            if not self.pending:
                return None
            selected, metadata = self.pending.pop(0)
            self.last_candidate_source = metadata["source"]
            self.last_candidate_metadata = metadata
            return SimpleNamespace(case=selected, source=metadata["source"], metadata=metadata)

        def candidate_quality_context(self, case, *, target_keys=None):
            return {}

    feedback = BatchFeedback()
    generate_calls = []

    def fake_generate_case(seed, **kwargs):
        generate_calls.append(seed)
        return _case(seed, profile=kwargs.get("profile", "common"))

    batch = generate_candidate_batch(
        seed_start=30,
        candidate_pool=3,
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_replay_bug=True,
        ),
        feedback=feedback,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=fake_generate_case,
        replay_filter_fn=lambda case, config: "",
    )

    assert generate_calls == [30]
    assert [case.case_id for case in batch.candidates] == [
        "case-130-common",
        "case-131-common",
        "case-132-common",
    ]
    assert batch.next_seed == 33
    assert [batch.candidate_meta[id(case)]["source"] for case in batch.candidates] == [
        "feedback_mutation",
        "feedback_mutation",
        "feedback_mutation",
    ]
    assert batch.candidate_meta[id(batch.candidates[0])]["seed_lineage"]["mutation_seed"] == 30
    assert batch.candidate_meta[id(batch.candidates[1])]["seed_lineage"]["mutation_seed"] == 31
    assert batch.candidate_meta[id(batch.candidates[2])]["seed_lineage"]["mutation_seed"] == 32
    assert feedback.batch_calls[0] == {
        "seed": 30,
        "max_batch": 3,
        "enqueue_remaining": True,
    }


def test_generate_candidate_batch_falls_back_after_replay_filter_saturation():
    def replay_filter(case, config):
        return "issue_replay_probe" if case.seed < 22 else ""

    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=False),
        feedback=None,
        guidance=None,
        generator_profile_pool=("known_replay_profile",),
        generator_profile_pool_metadata={"selected": ["known_replay_profile"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=replay_filter,
    )

    candidate = batch.candidates[0]
    meta = batch.candidate_meta[id(candidate)]
    assert candidate.seed == 22
    assert batch.next_seed == 23
    assert batch.replay_filtered_candidates == 21
    assert batch.replay_fallback_candidates == 1
    assert meta["source"] == "generated_fresh_fallback"
    assert meta["generated_seed"] == 22
    assert meta["replay_filter"]["filtered_before_candidate"] == 21
    assert meta["replay_filter"]["fallback_used"] is True
    assert meta["replay_filter"]["last_skip_reason"] == "issue_replay_probe"


def test_replay_filter_runs_before_feedback_decision_for_generated_candidates():
    class FeedbackShouldNotBeCalled:
        def select_case(self, seed, generated):
            raise AssertionError("replay-filtered generated candidates must not enter feedback selection")

    def replay_filter(case, config):
        return "issue_replay_probe" if case.seed < 22 else ""

    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=False),
        feedback=FeedbackShouldNotBeCalled(),
        guidance=None,
        generator_profile_pool=("known_replay_profile",),
        generator_profile_pool_metadata={"selected": ["known_replay_profile"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=replay_filter,
    )

    meta = batch.candidate_meta[id(batch.candidates[0])]
    assert meta["source"] == "generated_fresh_fallback"
    assert meta["replay_filter"]["filtered_before_candidate"] == 21
    assert meta["replay_filter"]["fallback_used"] is True


def test_generate_candidate_batch_falls_back_after_family_saturation_filter():
    class SaturatedGuidance:
        def predicted_saturated_family_roots_for_candidate(self, case, *, include_known_families=True):
            return ["window_boundary"]

    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=True),
        feedback=None,
        guidance=SaturatedGuidance(),
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=lambda case, config: "",
    )

    candidate = batch.candidates[0]
    meta = batch.candidate_meta[id(candidate)]
    assert candidate.seed == 22
    assert batch.next_seed == 23
    assert batch.saturated_family_filtered_candidates == 21
    assert batch.saturated_family_fallback_candidates == 1
    assert meta["source"] == "generated_saturation_fallback"
    assert meta["family_saturation_filter"]["filtered_before_candidate"] == 21
    assert meta["family_saturation_filter"]["fallback_used"] is True
    assert meta["family_saturation_filter"]["last_skip_reason"] == "window_boundary"


def test_generate_case_with_optional_schema_falls_back_for_legacy_generators():
    calls: list[tuple[int, dict]] = []

    def legacy_generate_case(seed, *, type_aware=True, profile="common"):
        calls.append((seed, {"type_aware": type_aware, "profile": profile}))
        return _case(seed, profile=profile)

    case = _generate_case_with_optional_schema(
        30,
        type_aware=False,
        profile="common",
        schema_spec={"columns": []},
        generate_case_fn=legacy_generate_case,
    )

    assert case.seed == 30
    assert calls == [(30, {"type_aware": False, "profile": "common"})]


def test_runner_reexports_run_candidates_helpers_for_compatibility():
    assert runner_module._generate_case_with_optional_schema is _generate_case_with_optional_schema
    assert runner_module._known_replay_source_filter_reason is _known_replay_source_filter_reason
    assert runner_module._replay_bug_filter_reason is _replay_bug_filter_reason
