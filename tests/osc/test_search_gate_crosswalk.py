from __future__ import annotations

from dataclasses import fields

from datadiff.datagen import generate_case
from datadiff_osc._canonical import canonical_envelope, decode_canonical_envelope
from datadiff_osc.search.context_receipts import (
    SEARCH_CONTEXT_SUBJECT_SPECS,
    FocusHitReceipt,
    FormalLanePlanReceipt,
    FormalRunBinding,
    IntrinsicFocusProof,
    MutationAttemptReceipt,
    ObservationContextReceipt,
    ScheduledTargetAttemptReceipt,
    SearchContextSubjectSpec,
    intrinsic_search_universe,
    reconstruct_registered_mutation_outcome,
)
from datadiff_osc.search.context_replay import (
    ContextReplayStatus,
    replay_context_admission,
)
from datadiff_osc.search.lane_registry import formal_focus_rule, formal_lane_registry
from datadiff_osc.semantic_targets.model import TargetAssignment


EXPECTED_CONTEXT_CROSSWALK = (
    ("formal_lanes", "FormalLaneRegistry", "lane_id", False),
    ("formal_lane_plans", "FormalLanePlanReceipt", "lane_id", False),
    (
        "focus_signals",
        "FormalLaneRegistry",
        "lane_signal_obligation_id",
        False,
    ),
    ("unique_focus_signals", "FormalLaneRegistry", "signal_id", False),
    (
        "fresh_cells_declared",
        "IntrinsicSearchUniverse",
        "target_cell_id",
        False,
    ),
    (
        "contrast_edges_declared",
        "IntrinsicSearchUniverse",
        "contrast_edge_id",
        False,
    ),
    (
        "backend_pair_obligations_declared",
        "IntrinsicSearchUniverse",
        "backend_pair_obligation_id",
        False,
    ),
    (
        "scheduled_target_attempts",
        "ScheduledTargetAttemptReceipt",
        "attempt_id",
        True,
    ),
    ("mutation_attempts", "MutationAttemptReceipt", "attempt_id", True),
    (
        "observation_contexts",
        "ObservationContextReceipt",
        "context_id",
        True,
    ),
    ("focus_hits", "FocusHitReceipt", "focus_context_id", True),
)


def _receipt_values() -> dict[str, object]:
    registry = formal_lane_registry()
    universe = intrinsic_search_universe()
    protocol = "phase6-search-crosswalk"
    lane_id = registry.lane_ids[0]
    run = FormalRunBinding.build(
        protocol_digest=protocol,
        lane_id=lane_id,
        seed_block_id="seed-block-a",
        master_seed=23,
        indexed_case_ids=(("case-000", 0),),
    )
    plan = FormalLanePlanReceipt.build(
        protocol_digest=protocol,
        lane_id=lane_id,
        planned_case_ids=("case-000",),
        seed_block_ids=("seed-block-a",),
        run_bindings=(run,),
    )
    _run, case_binding = plan.case_run_binding("case-000", 0)
    assignment = TargetAssignment.build(
        selected_cell_ids=(universe.fresh_cell_ids[0],),
        seed_lineage=case_binding.seed_lineage,
    )
    scheduled = ScheduledTargetAttemptReceipt.build(
        plan=plan,
        case_id="case-000",
        case_index=0,
        assignment=assignment,
    )
    obligation = registry.obligations_for(plan.lane_id)[0]
    source_case = generate_case(
        case_binding.seed_lineage.subseed,
        profile=formal_focus_rule(obligation.signal_id).source_generator,
    )
    operator_id = "value"
    outcome = reconstruct_registered_mutation_outcome(
        source_case=source_case,
        assignment=assignment,
        operator_id=operator_id,
    )
    mutation = MutationAttemptReceipt.build(
        plan=plan,
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        source_case=source_case,
        operator_id=operator_id,
        outcome=outcome,
    )
    observation = ObservationContextReceipt.build(
        plan=plan,
        case_id="case-000",
        assignment=assignment,
    )
    proof = IntrinsicFocusProof.build(
        plan=plan,
        case_id="case-000",
        obligation_id=obligation.obligation_id,
    )
    focus = FocusHitReceipt.build(
        plan=plan,
        observation_context=observation,
        obligation_id=obligation.obligation_id,
        intrinsic_proofs=(proof,),
    )
    return {
        "registry": registry,
        "universe": universe,
        "plan": plan,
        "scheduled": scheduled,
        "mutation": mutation,
        "observation": observation,
        "focus": focus,
    }


def _replay(
    type_name: str,
    value: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
):
    envelope = decode_canonical_envelope(
        canonical_envelope(type_name, value.schema_version, value)
    )
    return replay_context_admission(
        receipt_type=type_name,
        schema_version=value.schema_version,
        payload=envelope["payload"],
        subject_kind=subject_kind,
        subject_ids=subject_ids,
    )


def test_crosswalk_is_exact_context_routing_and_contains_no_gate_fields():
    assert tuple(
        (
            item.subject_kind,
            item.producer_type,
            item.identity_kind,
            item.runtime_context_required,
        )
        for item in SEARCH_CONTEXT_SUBJECT_SPECS
    ) == EXPECTED_CONTEXT_CROSSWALK
    assert len({item.subject_kind for item in SEARCH_CONTEXT_SUBJECT_SPECS}) == len(
        SEARCH_CONTEXT_SUBJECT_SPECS
    )
    assert tuple(field.name for field in fields(SearchContextSubjectSpec)) == (
        "subject_kind",
        "producer_type",
        "identity_kind",
        "runtime_context_required",
        "schema_version",
    )


def test_intrinsic_crosswalk_subjects_validate_without_becoming_authority():
    values = _receipt_values()
    registry = values["registry"]
    universe = values["universe"]
    plan = values["plan"]
    claims = (
        ("FormalLaneRegistry", registry, "formal_lanes", tuple(sorted(registry.lane_ids))),
        (
            "FormalLaneRegistry",
            registry,
            "focus_signals",
            tuple(sorted(item.obligation_id for item in registry.focus_obligations)),
        ),
        (
            "FormalLaneRegistry",
            registry,
            "unique_focus_signals",
            registry.unique_focus_signal_ids,
        ),
        (
            "FormalLanePlanReceipt",
            plan,
            "formal_lane_plans",
            (plan.lane_id,),
        ),
        (
            "IntrinsicSearchUniverse",
            universe,
            "fresh_cells_declared",
            universe.fresh_cell_ids,
        ),
        (
            "IntrinsicSearchUniverse",
            universe,
            "contrast_edges_declared",
            universe.contrast_edge_ids,
        ),
        (
            "IntrinsicSearchUniverse",
            universe,
            "backend_pair_obligations_declared",
            universe.backend_pair_obligation_ids,
        ),
    )

    assert (
        len(universe.fresh_family_ids),
        len(universe.fresh_cell_ids),
        len(universe.contrast_edge_ids),
        len(universe.backend_pair_obligation_ids),
    ) == (16, 232, 384, 502)
    for type_name, value, subject_kind, subject_ids in claims:
        result = _replay(type_name, value, subject_kind, subject_ids)
        assert result.status is ContextReplayStatus.VALIDATED_INTRINSIC
        assert result.derived_subject_ids == tuple(sorted(subject_ids))
        assert not result.authority_eligible
        assert not result.admitted


def test_runtime_crosswalk_subjects_remain_context_required():
    values = _receipt_values()
    claims = (
        (
            "ScheduledTargetAttemptReceipt",
            values["scheduled"],
            "scheduled_target_attempts",
            (values["scheduled"].attempt_id,),
        ),
        (
            "MutationAttemptReceipt",
            values["mutation"],
            "mutation_attempts",
            (values["mutation"].attempt_id,),
        ),
        (
            "ObservationContextReceipt",
            values["observation"],
            "observation_contexts",
            (values["observation"].context_id,),
        ),
        (
            "FocusHitReceipt",
            values["focus"],
            "focus_hits",
            (values["focus"].focus_context_id,),
        ),
    )

    for type_name, value, subject_kind, subject_ids in claims:
        result = _replay(type_name, value, subject_kind, subject_ids)
        assert result.status is ContextReplayStatus.CONTEXT_REQUIRED
        assert result.errors == (f"runtime_context_required:{subject_kind}",)
        assert not result.authority_eligible
        assert not result.admitted


def test_subject_forgery_and_cross_producer_claims_fail_closed():
    values = _receipt_values()
    registry = values["registry"]

    forged = _replay("FormalLaneRegistry", registry, "formal_lanes", ("forged",))
    assert forged.status is ContextReplayStatus.REJECTED
    assert not forged.admitted

    cross_producer = _replay(
        "FormalLaneRegistry",
        registry,
        "fresh_cells_declared",
        ("forged",),
    )
    assert cross_producer.status is ContextReplayStatus.REJECTED
    assert cross_producer.errors == ("subject_producer_type_mismatch",)
    assert not cross_producer.authority_eligible
