from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import FunctionType, MappingProxyType, ModuleType

import pytest

from datadiff.datagen import generate_case
from datadiff.family_witness_registry import latest_family_witness_registrations
from datadiff.mutator import ALL_MUTATION_OPERATOR_PROFILES

from datadiff_osc._canonical import (
    canonical_envelope,
    canonical_roundtrip,
    decode_canonical_envelope,
)
from datadiff_osc.generation.extraction import AtomExtractor
from datadiff_osc.generation.mutation import MutationOutcome, TargetPreservingMutator
from datadiff_osc.schemas import SeedLineage, SeedStage
import datadiff_osc.search.context_receipts as context_receipts_module
from datadiff_osc.search.context_receipts import (
    CanonicalCaseBinding,
    FocusHitReceipt,
    FormalLanePlanReceipt,
    FormalRunBinding,
    IntrinsicFocusProof,
    MutationAttemptReceipt,
    ObservationContextReceipt,
    ScheduledTargetAttemptReceipt,
    intrinsic_search_universe,
    mutation_operator_application_policy_id,
    reconstruct_registered_mutation_outcome,
    registered_mutation_operator_digest,
)
from datadiff_osc.search.context_receipts import (
    _callable_implementation_digest,
    _policy_callable_implementation_digest,
)
from datadiff_osc.search.context_replay import (
    ContextReplayStatus,
    reconstruct_context_receipt,
    replay_context_admission,
)
from datadiff_osc.search.epochs import derive_stage_lineage
from datadiff_osc.search.lane_registry import formal_focus_rule, formal_lane_registry
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


def _payload(type_name: str, value: object) -> object:
    encoded = canonical_envelope(type_name, value.schema_version, value)
    return decode_canonical_envelope(encoded)["payload"]


def _exact_outcome(assignment: TargetAssignment, source_case, operator_id: str):
    return reconstruct_registered_mutation_outcome(
        source_case=source_case,
        assignment=assignment,
        operator_id=operator_id,
    )


def _clone_python_function(
    value,
    *,
    globals_update: dict[str, object] | None = None,
    defaults: tuple[object, ...] | None = None,
    kwdefaults: dict[str, object] | None = None,
    closure=None,
):
    globals_copy = dict(value.__globals__)
    globals_copy.update(globals_update or {})
    cloned = FunctionType(
        value.__code__,
        globals_copy,
        value.__name__,
        value.__defaults__ if defaults is None else defaults,
        value.__closure__ if closure is None else closure,
    )
    cloned.__module__ = value.__module__
    cloned.__qualname__ = value.__qualname__
    cloned.__kwdefaults__ = (
        dict(value.__kwdefaults__ or {})
        if kwdefaults is None
        else dict(kwdefaults)
    )
    cloned.__annotations__ = dict(value.__annotations__)
    cloned.__dict__.update(value.__dict__)
    return cloned


def _alternate_mutate_value(table, rnd):
    del rnd
    if table.rows:
        table.rows.clear()
    return "value:alternate-global"


def _one_attempt_mutation_outcome(
    accepted_case,
    certificate,
    status,
    reason,
    attempts,
    mutation_seed,
):
    del attempts
    return MutationOutcome(
        accepted_case,
        certificate,
        status,
        reason,
        1,
        mutation_seed,
    )


class _IntermediateOwnerHolder:
    STATE = MappingProxyType({"marker": "class-baseline"})


_INTERMEDIATE_OWNER_MODULE = ModuleType("osc_r4_intermediate_owner_module")
_INTERMEDIATE_OWNER_MODULE.__file__ = __file__
_INTERMEDIATE_OWNER_MODULE.STATE = MappingProxyType({"marker": "module-baseline"})


def _class_intermediate_owner_get():
    return _IntermediateOwnerHolder.STATE.get("marker")


def _module_intermediate_owner_get():
    return _INTERMEDIATE_OWNER_MODULE.STATE.get("marker")


_POLICY_R4_RECURSIVE_MARKER = "recursive-baseline"


def _policy_r4_recursive_leaf(
    value,
    marker="default-baseline",
    *,
    mode="keyword-baseline",
):
    return f"{value}:{_POLICY_R4_RECURSIVE_MARKER}:{marker}:{mode}"


def _policy_r4_recursive_delegate(value):
    return _policy_r4_recursive_leaf(value)


def _policy_r4_recursive_root(value):
    return _policy_r4_recursive_delegate(value)


def _policy_r4_dynamic_root(value):
    return getattr(value, "marker")


def _r5_rebound_seed_lineage(*args, **kwargs):
    rebound_kwargs = dict(kwargs)
    rebound_kwargs["counter"] = int(rebound_kwargs["counter"]) + 1
    return SeedLineage(*args, **rebound_kwargs)


_R6_CONTEXTVAR_SEED_OFFSET = ContextVar(
    "osc_r6_seed_lineage_offset",
    default=0,
)


def _r6_contextvar_seed_lineage(*args, **kwargs):
    rebound_kwargs = dict(kwargs)
    rebound_kwargs["counter"] = (
        int(rebound_kwargs["counter"])
        + _R6_CONTEXTVAR_SEED_OFFSET.get()
    )
    return SeedLineage(*args, **rebound_kwargs)


def _r5_dynamic_seed_lineage(*args, **kwargs):
    return getattr(SeedLineage, "__call__")(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class _R5CyclicSeedLineageContext:
    reference: object


_R4_REPLAY_CALLS = 0
_R4_BASELINE_VALUE_APPLY = ALL_MUTATION_OPERATOR_PROFILES["value"].apply


def _r4_replay_counting_apply(tables, operations, rnd):
    global _R4_REPLAY_CALLS
    _R4_REPLAY_CALLS += 1
    return _R4_BASELINE_VALUE_APPLY(tables, operations, rnd)


def _contextual_profile_apply_factory(delegate, marker):
    def contextual_apply(
        tables,
        operations,
        rnd,
        suffix="baseline-default",
        *,
        mode="baseline-keyword-default",
    ):
        detail = delegate(tables, operations, rnd)
        return f"{detail}:{marker}:{suffix}:{mode}"

    return contextual_apply


class _ClassBoundMutationBehavior:
    @staticmethod
    def apply(tables, operations, rnd):
        del operations, rnd
        if tables and tables[0].rows and tables[0].rows[0]:
            first_column = next(iter(tables[0].rows[0]))
            tables[0].rows[0][first_column] = None
        return "value:baseline-class-global"


def _class_bound_mutation_apply(tables, operations, rnd):
    return _ClassBoundMutationBehavior.apply(tables, operations, rnd)


def _alternate_registered_mutation_apply(tables, operations, rnd):
    del operations
    if not tables:
        return "value:none"
    return _alternate_mutate_value(tables[0], rnd)


@dataclass(frozen=True, slots=True)
class _UnsupportedCallableMutationBehavior:
    marker: str

    def __call__(self, tables, operations, rnd):
        del tables, operations, rnd
        return self.marker


_UNSUPPORTED_CALLABLE_DEPENDENCY = _UnsupportedCallableMutationBehavior(
    "unsupported-callable-dependency"
)


def _callable_object_dependency_apply(tables, operations, rnd):
    return _UNSUPPORTED_CALLABLE_DEPENDENCY(tables, operations, rnd)


def _dynamic_globals_apply(tables, operations, rnd):
    return globals()["_alternate_registered_mutation_apply"](
        tables, operations, rnd
    )


def _dynamic_locals_apply(tables, operations, rnd):
    del tables, operations, rnd
    return str(locals())


def _dynamic_vars_apply(tables, operations, rnd):
    del tables, operations, rnd
    return str(vars())


def _dynamic_getattr_apply(tables, operations, rnd):
    del operations
    return getattr(_ClassBoundMutationBehavior, "apply")(tables, [], rnd)


def _dynamic_eval_apply(tables, operations, rnd):
    del tables, operations, rnd
    return eval("'dynamic-eval'")


def _dynamic_exec_apply(tables, operations, rnd):
    del tables, operations, rnd
    namespace: dict[str, str] = {}
    exec("result = 'dynamic-exec'", namespace)
    return namespace["result"]


def _dynamic_dunder_import_apply(tables, operations, rnd):
    del tables, operations, rnd
    return __import__("builtins").str("dynamic-import")


def _inline_import_apply(tables, operations, rnd):
    del tables, operations, rnd
    import math

    return str(math.pi)


def _replacement_profile(baseline_profile, replacement_kind):
    if replacement_kind == "callable":
        return replace(
            baseline_profile,
            apply=_clone_python_function(
                baseline_profile.apply,
                globals_update={"_mutate_value": _alternate_mutate_value},
            ),
        )
    if replacement_kind == "profile":
        return replace(
            baseline_profile,
            semantic_family_affinity=("replacement-family",),
        )
    if replacement_kind == "registry":
        return baseline_profile
    raise AssertionError(f"unexpected replacement kind: {replacement_kind}")


def _receipt_bundle() -> dict[str, object]:
    registry = formal_lane_registry()
    universe = intrinsic_search_universe()
    protocol = "phase6-search-protocol"
    lane_id = registry.lane_ids[0]
    run_a = FormalRunBinding.build(
        protocol_digest=protocol,
        lane_id=lane_id,
        seed_block_id="seed-block-a",
        master_seed=17,
        indexed_case_ids=(("case-000", 0),),
    )
    run_b = FormalRunBinding.build(
        protocol_digest=protocol,
        lane_id=lane_id,
        seed_block_id="seed-block-b",
        master_seed=29,
        indexed_case_ids=(("case-001", 1),),
    )
    plan = FormalLanePlanReceipt.build(
        protocol_digest=protocol,
        lane_id=lane_id,
        planned_case_ids=("case-001", "case-000"),
        seed_block_ids=("seed-block-b", "seed-block-a"),
        run_bindings=(run_b, run_a),
    )
    case_run, case_binding = plan.case_run_binding("case-000", 0)
    assignment = TargetAssignment.build(
        selected_cell_ids=(universe.fresh_cell_ids[0],),
        seed_lineage=case_binding.seed_lineage,
        parameters={"generator_profile": "comparison-only-metadata"},
    )
    obligation = registry.obligations_for(plan.lane_id)[0]
    rule = formal_focus_rule(obligation.signal_id)
    source_case = generate_case(
        case_binding.seed_lineage.subseed,
        profile=rule.source_generator,
    )
    scheduled = ScheduledTargetAttemptReceipt.build(
        plan=plan,
        case_id="case-000",
        case_index=0,
        assignment=assignment,
    )
    operator_id = "value"
    outcome = _exact_outcome(assignment, source_case, operator_id)
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
    complete_observation = ObservationContextReceipt.build(
        plan=plan,
        case_id="case-000",
        assignment=assignment,
        activation_certificate_ref="activation-opaque-ref",
        runtime_task_refs=("task-opaque-ref",),
        runtime_outcome_refs=("outcome-opaque-ref",),
        observation_certificate_ref="observation-opaque-ref",
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
        "run_a": run_a,
        "run_b": run_b,
        "case_binding": case_binding,
        "case_run": case_run,
        "plan": plan,
        "assignment": assignment,
        "source_case": source_case,
        "scheduled": scheduled,
        "mutation": mutation,
        "observation": observation,
        "complete_observation": complete_observation,
        "proof": proof,
        "focus": focus,
    }


def _target_valid_mutation_context(bundle: dict[str, object]):
    compiled = compile_target_universe(legacy_v4_target_templates())
    cell = next(
        item
        for item in compiled.fresh_cells
        if item.test_family_id == "polars_lazy_filter_groupby_window"
        and item.coordinate_map["data_pattern"] == "balanced"
    )
    registration = next(
        item
        for item in latest_family_witness_registrations()
        if item.family_id == cell.test_family_id
    )
    source_case = registration.generate_case(cell.construction_index)
    source_case.seed = bundle["case_binding"].seed_lineage.subseed
    assignment = TargetAssignment.build(
        selected_cell_ids=(cell.target_cell_id,),
        seed_lineage=bundle["case_binding"].seed_lineage,
    )
    return source_case, assignment


def test_plan_builders_reject_every_duplicate_before_canonical_ordering():
    bundle = _receipt_bundle()
    run = bundle["run_a"]
    plan = bundle["plan"]

    with pytest.raises(ValueError, match="planned_case_ids contains duplicates"):
        FormalLanePlanReceipt.build(
            protocol_digest=plan.protocol_digest,
            lane_id=plan.lane_id,
            planned_case_ids=("case-000", "case-000"),
            seed_block_ids=("seed-block-a",),
            run_bindings=(run,),
        )
    with pytest.raises(ValueError, match="seed_block_ids contains duplicates"):
        FormalLanePlanReceipt.build(
            protocol_digest=plan.protocol_digest,
            lane_id=plan.lane_id,
            planned_case_ids=("case-000",),
            seed_block_ids=("seed-block-a", "seed-block-a"),
            run_bindings=(run,),
        )
    with pytest.raises(ValueError, match="run_bindings contains duplicates"):
        FormalLanePlanReceipt.build(
            protocol_digest=plan.protocol_digest,
            lane_id=plan.lane_id,
            planned_case_ids=("case-000",),
            seed_block_ids=("seed-block-a",),
            run_bindings=(run, run),
        )

    rebuilt = FormalLanePlanReceipt.build(
        protocol_digest=plan.protocol_digest,
        lane_id=plan.lane_id,
        planned_case_ids=reversed(plan.planned_case_ids),
        seed_block_ids=reversed(plan.seed_block_ids),
        run_bindings=reversed(plan.run_bindings),
    )
    assert rebuilt == plan


def test_runtime_and_focus_builders_reject_duplicates_instead_of_collapsing():
    bundle = _receipt_bundle()
    plan = bundle["plan"]
    assignment = bundle["assignment"]
    proof = bundle["proof"]

    with pytest.raises(ValueError, match="runtime_task_refs contains duplicates"):
        ObservationContextReceipt.build(
            plan=plan,
            case_id="case-000",
            assignment=assignment,
            runtime_task_refs=("task-a", "task-a"),
        )
    with pytest.raises(ValueError, match="runtime_outcome_refs contains duplicates"):
        ObservationContextReceipt.build(
            plan=plan,
            case_id="case-000",
            assignment=assignment,
            runtime_outcome_refs=("outcome-a", "outcome-a"),
        )
    with pytest.raises(ValueError, match="intrinsic_proofs contains duplicates"):
        FocusHitReceipt.build(
            plan=plan,
            observation_context=bundle["observation"],
            obligation_id=bundle["focus"].obligation.obligation_id,
            intrinsic_proofs=(proof, proof),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("protocol_digest", "wrong-protocol"),
        ("lane_id", "wrong-lane"),
        ("master_seed", 999),
        ("stage_name", SeedStage.DATA),
        ("case_index", 7),
        ("counter", 7),
        ("parent_digest", "wrong-parent"),
    ),
)
def test_formal_run_rejects_wrong_seed_lineage_relationships(field, replacement):
    run = _receipt_bundle()["run_a"]
    case = run.case_bindings[0]
    forged_lineage = replace(case.seed_lineage, **{field: replacement})
    forged_case = replace(case, seed_lineage=forged_lineage)
    with pytest.raises(ValueError, match="lineage does not recompute"):
        replace(run, case_bindings=(forged_case,))


def test_wrong_run_block_case_mapping_and_cross_run_assignment_reject():
    bundle = _receipt_bundle()
    run = bundle["run_a"]
    plan = bundle["plan"]

    with pytest.raises(ValueError, match="identity mismatch"):
        replace(run, seed_block_id="seed-block-forged")
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(run, run_id="run-forged")
    with pytest.raises(ValueError, match="exact formal case mapping"):
        ScheduledTargetAttemptReceipt.build(
            plan=plan,
            case_id="case-001",
            case_index=1,
            assignment=bundle["assignment"],
        )
    with pytest.raises(ValueError, match="index-bound"):
        ScheduledTargetAttemptReceipt.build(
            plan=plan,
            case_id="case-000",
            case_index=1,
            assignment=bundle["assignment"],
        )


def test_mutation_binds_source_lineage_operator_outcome_and_certificate():
    bundle = _receipt_bundle()
    mutation = bundle["mutation"]

    assert mutation.mutation_lineage.stage_name is SeedStage.MUTATION
    assert mutation.mutation_seed == mutation.mutation_lineage.subseed
    assert mutation.outcome.mutation_seed == mutation.mutation_seed
    assert mutation.activation_certificate is mutation.outcome.certificate
    assert mutation.activation_certificate.assignment_digest == (
        mutation.assignment.digest
    )
    assert mutation.operator_application_policy_id == (
        mutation_operator_application_policy_id()
    )
    assert mutation.operator_digest == registered_mutation_operator_digest(
        mutation.operator_id
    )
    assert mutation.outcome == _exact_outcome(
        mutation.assignment,
        mutation.source_case.case,
        mutation.operator_id,
    )

    with pytest.raises(ValueError, match="seed does not recompute"):
        replace(mutation, mutation_seed=mutation.mutation_seed + 1)
    with pytest.raises(ValueError, match="lineage does not recompute"):
        replace(
            mutation,
            mutation_lineage=replace(
                mutation.mutation_lineage,
                stage_name=SeedStage.DATA,
            ),
        )
    with pytest.raises(ValueError, match="lineage does not recompute"):
        replace(
            mutation,
            mutation_lineage=replace(
                mutation.mutation_lineage,
                parent_digest="swapped-parent",
            ),
        )
    with pytest.raises(ValueError, match="lineage does not recompute"):
        replace(
            mutation,
            mutation_lineage=replace(mutation.mutation_lineage, counter=1),
        )
    with pytest.raises(ValueError, match="operator identity"):
        replace(mutation, operator_digest="caller-authored-operator-digest")
    with pytest.raises(ValueError, match="application policy"):
        replace(
            mutation,
            operator_application_policy_id="caller-selected-no-repair-policy",
        )
    with pytest.raises(ValueError, match="certificate is not outcome-bound"):
        replace(
            mutation,
            activation_certificate=replace(
                mutation.activation_certificate,
                assignment_digest="swapped-assignment",
            ),
        )
    with pytest.raises(ValueError, match="outcome seed"):
        replace(
            mutation,
            outcome=replace(
                mutation.outcome,
                mutation_seed=mutation.mutation_seed + 1,
            ),
        )


def test_selected_registry_operator_is_causally_executed_not_co_hashed():
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    value_outcome = _exact_outcome(assignment, source_case, "value")
    nullify_outcome = _exact_outcome(
        assignment,
        source_case,
        "nullify_value",
    )

    assert value_outcome != nullify_outcome
    value_receipt = MutationAttemptReceipt.build(
        plan=bundle["plan"],
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        source_case=source_case,
        operator_id="value",
        outcome=value_outcome,
    )
    assert value_receipt.operator_digest == registered_mutation_operator_digest(
        "value"
    )
    assert registered_mutation_operator_digest(
        "value"
    ) != registered_mutation_operator_digest("nullify_value")

    with pytest.raises(ValueError, match="does not reproduce"):
        MutationAttemptReceipt.build(
            plan=bundle["plan"],
            case_id="case-000",
            case_index=0,
            assignment=assignment,
            source_case=source_case,
            operator_id="nullify_value",
            outcome=value_outcome,
        )
    with pytest.raises(ValueError, match="does not reproduce"):
        replace(
            value_receipt,
            operator_id="nullify_value",
            operator_digest=registered_mutation_operator_digest(
                "nullify_value"
            ),
        )
    forged_source = deepcopy(source_case)
    forged_source.tables[0].rows.clear()
    with pytest.raises(ValueError, match="does not reproduce"):
        replace(
            value_receipt,
            source_case=CanonicalCaseBinding.build(
                binding_case_id=value_receipt.case_id,
                case=forged_source,
            ),
        )


def test_callable_identity_binds_globals_defaults_kwdefaults_and_closure():
    original_apply = ALL_MUTATION_OPERATOR_PROFILES["value"].apply
    exact_clone = _clone_python_function(original_apply)
    assert _callable_implementation_digest(
        original_apply, "exact value operator clone"
    ) == _callable_implementation_digest(
        exact_clone, "exact value operator clone"
    )
    rebound_global_apply = _clone_python_function(
        original_apply,
        globals_update={"_mutate_value": _alternate_mutate_value},
    )
    assert _callable_implementation_digest(
        original_apply, "cold original value operator"
    ) != _callable_implementation_digest(
        rebound_global_apply, "cold original value operator"
    )

    def defaulted_apply(tables, operations, rnd, marker="original", *, mode="cold"):
        del tables, operations, rnd
        return f"{marker}:{mode}"

    changed_defaults = _clone_python_function(
        defaulted_apply,
        defaults=("replacement",),
    )
    changed_kwdefaults = _clone_python_function(
        defaulted_apply,
        kwdefaults={"mode": "warm"},
    )
    baseline_default_digest = _callable_implementation_digest(
        defaulted_apply, "default-bound operator"
    )
    assert baseline_default_digest != _callable_implementation_digest(
        changed_defaults, "default-bound operator"
    )
    assert baseline_default_digest != _callable_implementation_digest(
        changed_kwdefaults, "default-bound operator"
    )

    def closure_factory(marker: str):
        def closure_apply(tables, operations, rnd):
            del tables, operations, rnd
            return marker

        return closure_apply

    assert _callable_implementation_digest(
        closure_factory("original"), "closure-bound operator"
    ) != _callable_implementation_digest(
        closure_factory("replacement"), "closure-bound operator"
    )


def test_live_global_rebinding_changes_warm_registered_identity(monkeypatch):
    baseline_profile = ALL_MUTATION_OPERATOR_PROFILES["value"]
    baseline_digest = registered_mutation_operator_digest("value")

    with monkeypatch.context() as scoped:
        scoped.setitem(
            baseline_profile.apply.__globals__,
            "_mutate_value",
            _alternate_mutate_value,
        )
        assert registered_mutation_operator_digest("value") != baseline_digest


def test_policy_callable_identity_binds_live_globals_and_rejects_prior_id_outcome(
    monkeypatch,
):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    baseline_outcome = _exact_outcome(assignment, source_case, "value")
    baseline_receipt = MutationAttemptReceipt.build(
        plan=bundle["plan"],
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        source_case=source_case,
        operator_id="value",
        outcome=baseline_outcome,
    )
    original_mutate = TargetPreservingMutator.mutate
    replacement_mutate = _clone_python_function(
        original_mutate,
        globals_update={"MutationOutcome": _one_attempt_mutation_outcome},
    )
    baseline_policy = mutation_operator_application_policy_id()
    assert mutation_operator_application_policy_id() == baseline_policy

    assert _policy_callable_implementation_digest(
        original_mutate,
        "original target-preserving mutator",
    ) != _policy_callable_implementation_digest(
        replacement_mutate,
        "original target-preserving mutator",
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(TargetPreservingMutator, "mutate", replacement_mutate)
        replacement_policy = mutation_operator_application_policy_id()
        assert replacement_policy != baseline_policy
        assert mutation_operator_application_policy_id() == replacement_policy
        replacement_outcome = _exact_outcome(assignment, source_case, "value")
        assert replacement_outcome.status == baseline_outcome.status == "rejected"
        assert replacement_outcome.attempts == 1
        assert baseline_outcome.attempts == 2
        with pytest.raises(ValueError, match="application policy mismatch"):
            replace(
                baseline_receipt,
                outcome=replacement_outcome,
                activation_certificate=replacement_outcome.certificate,
            )


def test_cross_module_derive_stage_lineage_rebind_changes_policy_cold_and_warm(
    monkeypatch,
):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    baseline_outcome = _exact_outcome(assignment, source_case, "value")
    baseline_receipt = MutationAttemptReceipt.build(
        plan=bundle["plan"],
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        source_case=source_case,
        operator_id="value",
        outcome=baseline_outcome,
    )
    baseline_policy = mutation_operator_application_policy_id()
    assert mutation_operator_application_policy_id() == baseline_policy

    baseline_lineage = derive_stage_lineage(
        assignment.seed_lineage,
        SeedStage.MUTATION,
    )
    rebound_derive_stage_lineage = _clone_python_function(
        derive_stage_lineage,
        globals_update={"SeedLineage": _r5_rebound_seed_lineage},
    )
    assert rebound_derive_stage_lineage.__code__ is derive_stage_lineage.__code__
    assert rebound_derive_stage_lineage.__module__ == derive_stage_lineage.__module__
    assert rebound_derive_stage_lineage.__qualname__ == derive_stage_lineage.__qualname__
    rebound_lineage = rebound_derive_stage_lineage(
        assignment.seed_lineage,
        SeedStage.MUTATION,
    )
    assert rebound_lineage.counter == baseline_lineage.counter + 1
    assert rebound_lineage.subseed != baseline_lineage.subseed
    assert baseline_outcome.mutation_seed == baseline_lineage.subseed

    original_mutate = TargetPreservingMutator.mutate
    replacement_mutate = _clone_python_function(
        original_mutate,
        globals_update={"derive_stage_lineage": rebound_derive_stage_lineage},
    )
    assert _policy_callable_implementation_digest(
        original_mutate,
        "TargetPreservingMutator.mutate",
        bind_cross_module_globals=True,
    ) != _policy_callable_implementation_digest(
        replacement_mutate,
        "TargetPreservingMutator.mutate",
        bind_cross_module_globals=True,
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(TargetPreservingMutator, "mutate", replacement_mutate)
        rebound_policy = mutation_operator_application_policy_id()
        assert rebound_policy != baseline_policy
        assert mutation_operator_application_policy_id() == rebound_policy
        with pytest.raises(ValueError, match="application policy mismatch"):
            replace(baseline_receipt)
        rebound_outcome = _exact_outcome(assignment, source_case, "value")
        assert rebound_outcome.mutation_seed == rebound_lineage.subseed
        assert rebound_outcome.mutation_seed != baseline_outcome.mutation_seed
        with pytest.raises(ValueError, match="application policy mismatch"):
            replace(
                baseline_receipt,
                outcome=rebound_outcome,
                activation_certificate=rebound_outcome.certificate,
            )


def test_cross_module_contextvar_active_value_rebinds_policy_before_replay(
    monkeypatch,
):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    original_mutate = TargetPreservingMutator.mutate
    rebound_derive_stage_lineage = _clone_python_function(
        derive_stage_lineage,
        globals_update={"SeedLineage": _r6_contextvar_seed_lineage},
    )
    assert rebound_derive_stage_lineage.__code__ is derive_stage_lineage.__code__
    assert rebound_derive_stage_lineage.__module__ == derive_stage_lineage.__module__
    assert rebound_derive_stage_lineage.__qualname__ == derive_stage_lineage.__qualname__
    replacement_mutate = _clone_python_function(
        original_mutate,
        globals_update={"derive_stage_lineage": rebound_derive_stage_lineage},
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(TargetPreservingMutator, "mutate", replacement_mutate)
        default_policy = mutation_operator_application_policy_id()
        assert mutation_operator_application_policy_id() == default_policy
        default_lineage = rebound_derive_stage_lineage(
            assignment.seed_lineage,
            SeedStage.MUTATION,
        )
        default_outcome = _exact_outcome(assignment, source_case, "value")
        assert default_outcome.mutation_seed == default_lineage.subseed
        default_receipt = MutationAttemptReceipt.build(
            plan=bundle["plan"],
            case_id="case-000",
            case_index=0,
            assignment=assignment,
            source_case=source_case,
            operator_id="value",
            outcome=default_outcome,
        )
        assert default_receipt.operator_application_policy_id == default_policy

        token = _R6_CONTEXTVAR_SEED_OFFSET.set(1)
        try:
            active_policy = mutation_operator_application_policy_id()
            assert active_policy != default_policy
            assert mutation_operator_application_policy_id() == active_policy
            active_lineage = rebound_derive_stage_lineage(
                assignment.seed_lineage,
                SeedStage.MUTATION,
            )
            assert active_lineage.counter == default_lineage.counter + 1
            assert active_lineage.subseed != default_lineage.subseed
            with pytest.raises(ValueError, match="application policy mismatch"):
                replace(default_receipt)
        finally:
            _R6_CONTEXTVAR_SEED_OFFSET.reset(token)


def test_cross_module_direct_callable_abnormal_context_fails_closed(monkeypatch):
    original_mutate = TargetPreservingMutator.mutate
    cyclic_context = object.__new__(_R5CyclicSeedLineageContext)
    object.__setattr__(cyclic_context, "reference", cyclic_context)
    scenarios = (
        ("mutable", {"counter": "replacement"}, "mutable live context"),
        ("cyclic", cyclic_context, "cyclicly unresolved"),
        ("uninspectable", ModuleType("osc_r5_uninspectable"), "uninspectable"),
        ("unsupported", object(), "live dependency type object is unsupported"),
        ("dynamic", _r5_dynamic_seed_lineage, "dynamic dependency resolver"),
    )

    for label, seed_lineage_context, expected_error in scenarios:
        rebound_derive_stage_lineage = _clone_python_function(
            derive_stage_lineage,
            globals_update={"SeedLineage": seed_lineage_context},
        )
        replacement_mutate = _clone_python_function(
            original_mutate,
            globals_update={"derive_stage_lineage": rebound_derive_stage_lineage},
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(TargetPreservingMutator, "mutate", replacement_mutate)
            with pytest.raises(ValueError, match=expected_error):
                mutation_operator_application_policy_id()

    unresolved_derive_stage_lineage = _clone_python_function(derive_stage_lineage)
    del unresolved_derive_stage_lineage.__globals__["SeedLineage"]
    unresolved_mutate = _clone_python_function(
        original_mutate,
        globals_update={"derive_stage_lineage": unresolved_derive_stage_lineage},
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(TargetPreservingMutator, "mutate", unresolved_mutate)
        with pytest.raises(ValueError, match="global dependency 'SeedLineage' is unresolved"):
            mutation_operator_application_policy_id()


def test_policy_callable_identity_binds_recursive_defaults_kwdefaults_and_closure():
    baseline = _policy_callable_implementation_digest(
        _policy_r4_recursive_root,
        "policy recursive root",
    )
    assert _policy_callable_implementation_digest(
        _policy_r4_recursive_root,
        "policy recursive root",
    ) == baseline
    assert _policy_r4_recursive_root("value") == (
        "value:recursive-baseline:default-baseline:keyword-baseline"
    )

    replacement_leaf = _clone_python_function(
        _policy_r4_recursive_leaf,
        globals_update={"_POLICY_R4_RECURSIVE_MARKER": "recursive-replacement"},
        defaults=("default-replacement",),
        kwdefaults={"mode": "keyword-replacement"},
    )
    replacement_delegate = _clone_python_function(
        _policy_r4_recursive_delegate,
        globals_update={"_policy_r4_recursive_leaf": replacement_leaf},
    )
    replacement_root = _clone_python_function(
        _policy_r4_recursive_root,
        globals_update={"_policy_r4_recursive_delegate": replacement_delegate},
    )
    assert replacement_root("value") == (
        "value:recursive-replacement:default-replacement:keyword-replacement"
    )
    replacement = _policy_callable_implementation_digest(
        replacement_root,
        "policy recursive root",
    )
    assert replacement != baseline
    assert _policy_callable_implementation_digest(
        replacement_root,
        "policy recursive root",
    ) == replacement

    def closure_factory(marker):
        def closure_bound_policy(value):
            return f"{value}:{marker}"

        return closure_bound_policy

    assert _policy_callable_implementation_digest(
        closure_factory("closure-baseline"),
        "policy closure callable",
    ) != _policy_callable_implementation_digest(
        closure_factory("closure-replacement"),
        "policy closure callable",
    )
    with pytest.raises(ValueError, match="dynamic dependency resolver"):
        _policy_callable_implementation_digest(
            _policy_r4_dynamic_root,
            "dynamic policy callable",
        )


def test_callable_identity_binds_all_module_and_class_path_owners(monkeypatch):
    class_digest = _callable_implementation_digest(
        _class_intermediate_owner_get,
        "class intermediate owner getter",
    )
    module_digest = _callable_implementation_digest(
        _module_intermediate_owner_get,
        "module intermediate owner getter",
    )
    assert _class_intermediate_owner_get() == "class-baseline"
    assert _module_intermediate_owner_get() == "module-baseline"

    with monkeypatch.context() as scoped:
        scoped.setattr(
            _IntermediateOwnerHolder,
            "STATE",
            MappingProxyType({"marker": "class-replacement"}),
        )
        assert _class_intermediate_owner_get() == "class-replacement"
        assert _callable_implementation_digest(
            _class_intermediate_owner_get,
            "class intermediate owner getter",
        ) != class_digest

    with monkeypatch.context() as scoped:
        scoped.setattr(
            _INTERMEDIATE_OWNER_MODULE,
            "STATE",
            MappingProxyType({"marker": "module-replacement"}),
        )
        assert _module_intermediate_owner_get() == "module-replacement"
        assert _callable_implementation_digest(
            _module_intermediate_owner_get,
            "module intermediate owner getter",
        ) != module_digest


def test_callable_identity_rejects_mutable_cyclic_unresolved_and_uninspectable_path_owners(
    monkeypatch,
):
    with monkeypatch.context() as scoped:
        scoped.setattr(_IntermediateOwnerHolder, "STATE", {"marker": "mutable"})
        with pytest.raises(ValueError, match="mutable live context"):
            _callable_implementation_digest(
                _class_intermediate_owner_get,
                "mutable class intermediate owner",
            )

    cyclic_state: dict[str, object] = {}
    cyclic_state["self"] = cyclic_state
    with monkeypatch.context() as scoped:
        scoped.setattr(_INTERMEDIATE_OWNER_MODULE, "STATE", cyclic_state)
        with pytest.raises(ValueError, match="mutable live context|cyclicly unresolved"):
            _callable_implementation_digest(
                _module_intermediate_owner_get,
                "cyclic module intermediate owner",
            )

    with monkeypatch.context() as scoped:
        scoped.delattr(_IntermediateOwnerHolder, "STATE")
        with pytest.raises(ValueError, match="attribute dependency is unresolved"):
            _callable_implementation_digest(
                _class_intermediate_owner_get,
                "unresolved class intermediate owner",
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            _INTERMEDIATE_OWNER_MODULE,
            "__file__",
            "/definitely/not/an/inspectable/r4/module.py",
        )
        with pytest.raises(ValueError, match="source module cannot be read exactly"):
            _callable_implementation_digest(
                _module_intermediate_owner_get,
                "uninspectable module owner",
            )


def test_selected_and_recursive_callable_objects_fail_closed(monkeypatch):
    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    baseline_profile = baseline_registry["value"]
    replacements = (
        _UnsupportedCallableMutationBehavior("selected-callable-object"),
        _callable_object_dependency_apply,
    )
    for replacement_apply in replacements:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                context_receipts_module,
                "ALL_MUTATION_OPERATOR_PROFILES",
                MappingProxyType(
                    {
                        **dict(baseline_registry),
                        "value": replace(
                            baseline_profile,
                            apply=replacement_apply,
                        ),
                    }
                ),
            )
            with pytest.raises(ValueError, match="not replayable|callable object"):
                registered_mutation_operator_digest("value")


@pytest.mark.parametrize(
    "dynamic_apply",
    (
        _dynamic_globals_apply,
        _dynamic_locals_apply,
        _dynamic_vars_apply,
        _dynamic_getattr_apply,
        _dynamic_eval_apply,
        _dynamic_exec_apply,
        _dynamic_dunder_import_apply,
        _inline_import_apply,
    ),
)
def test_dynamic_dependency_resolution_and_imports_fail_closed(
    monkeypatch,
    dynamic_apply,
):
    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    baseline_profile = baseline_registry["value"]
    with monkeypatch.context() as scoped:
        scoped.setattr(
            context_receipts_module,
            "ALL_MUTATION_OPERATOR_PROFILES",
            MappingProxyType(
                {
                    **dict(baseline_registry),
                    "value": replace(baseline_profile, apply=dynamic_apply),
                }
            ),
        )
        with pytest.raises(ValueError, match="dynamic dependency|dynamic import"):
            registered_mutation_operator_digest("value")


def test_registered_operator_identity_recomputes_live_profile_after_warm_call(
    monkeypatch,
):
    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    baseline_profile = baseline_registry["value"]
    baseline_digest = registered_mutation_operator_digest("value")
    rebound_apply = _clone_python_function(
        baseline_profile.apply,
        globals_update={"_mutate_value": _alternate_mutate_value},
    )

    substitutions = (
        replace(baseline_profile, apply=rebound_apply),
        replace(
            baseline_profile,
            semantic_family_affinity=("replacement-family",),
        ),
        replace(
            baseline_profile,
            semantic_signal_affinity=("replacement-signal",),
        ),
        replace(
            baseline_profile,
            exploration_objective_affinity=("replacement-objective",),
        ),
        replace(
            baseline_profile,
            divergence_affinity=("replacement-divergence",),
        ),
        replace(
            baseline_profile,
            structural_risk_tags=("replacement-risk",),
        ),
        replace(
            baseline_profile,
            coverage_axes=("replacement-axis",),
        ),
        replace(
            baseline_profile,
            expandability_bias=baseline_profile.expandability_bias + 0.01,
        ),
        replace(
            baseline_profile,
            validity_floor=baseline_profile.validity_floor - 0.01,
        ),
    )
    for replacement_profile in substitutions:
        replacement_registry = MappingProxyType(
            {
                **dict(baseline_registry),
                "value": replacement_profile,
            }
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                "datadiff_osc.search.context_receipts.ALL_MUTATION_OPERATOR_PROFILES",
                replacement_registry,
            )
            assert registered_mutation_operator_digest("value") != baseline_digest


def test_warm_registered_identity_binds_defaults_kwdefaults_and_closure(
    monkeypatch,
):
    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    baseline_profile = baseline_registry["value"]
    baseline_apply = _contextual_profile_apply_factory(
        baseline_profile.apply,
        "baseline-closure",
    )
    baseline_context_profile = replace(baseline_profile, apply=baseline_apply)
    baseline_context_registry = MappingProxyType(
        {
            **dict(baseline_registry),
            "value": baseline_context_profile,
        }
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            context_receipts_module,
            "ALL_MUTATION_OPERATOR_PROFILES",
            baseline_context_registry,
        )
        warm_digest = registered_mutation_operator_digest("value")
        assert registered_mutation_operator_digest("value") == warm_digest

    replacements = (
        _clone_python_function(
            baseline_apply,
            defaults=("replacement-default",),
        ),
        _clone_python_function(
            baseline_apply,
            kwdefaults={"mode": "replacement-keyword-default"},
        ),
        _contextual_profile_apply_factory(
            baseline_profile.apply,
            "replacement-closure",
        ),
    )
    for replacement_apply in replacements:
        replacement_registry = MappingProxyType(
            {
                **dict(baseline_registry),
                "value": replace(
                    baseline_context_profile,
                    apply=replacement_apply,
                ),
            }
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                context_receipts_module,
                "ALL_MUTATION_OPERATOR_PROFILES",
                replacement_registry,
            )
            assert registered_mutation_operator_digest("value") != warm_digest


def test_referenced_callable_class_substitution_changes_live_identity(
    monkeypatch,
):
    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    baseline_profile = replace(
        baseline_registry["value"],
        apply=_class_bound_mutation_apply,
    )
    replacement_class = type(
        _ClassBoundMutationBehavior.__name__,
        (object,),
        {
            "__module__": _ClassBoundMutationBehavior.__module__,
            "__qualname__": _ClassBoundMutationBehavior.__qualname__,
            "apply": staticmethod(_alternate_registered_mutation_apply),
        },
    )
    replacement_apply = _clone_python_function(
        _class_bound_mutation_apply,
        globals_update={"_ClassBoundMutationBehavior": replacement_class},
    )
    replacement_profile = replace(baseline_profile, apply=replacement_apply)

    exact_clone = _clone_python_function(_class_bound_mutation_apply)
    assert _callable_implementation_digest(
        _class_bound_mutation_apply, "exact class-bound operator clone"
    ) == _callable_implementation_digest(
        exact_clone, "exact class-bound operator clone"
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            context_receipts_module,
            "ALL_MUTATION_OPERATOR_PROFILES",
            MappingProxyType(
                {
                    **dict(baseline_registry),
                    "value": baseline_profile,
                }
            ),
        )
        baseline_digest = registered_mutation_operator_digest("value")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            context_receipts_module,
            "ALL_MUTATION_OPERATOR_PROFILES",
            MappingProxyType(
                {
                    **dict(baseline_registry),
                    "value": replacement_profile,
                }
            ),
        )
        replacement_digest = registered_mutation_operator_digest("value")

    assert replacement_digest != baseline_digest


def test_replacement_outcome_rejects_under_prior_warm_identity_before_replay(
    monkeypatch,
):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    baseline_outcome = _exact_outcome(assignment, source_case, "value")
    baseline_receipt = MutationAttemptReceipt.build(
        plan=bundle["plan"],
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        source_case=source_case,
        operator_id="value",
        outcome=baseline_outcome,
    )
    baseline_digest = registered_mutation_operator_digest("value")
    assert baseline_receipt.operator_digest == baseline_digest

    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    replacement_profile = _replacement_profile(
        baseline_registry["value"],
        "callable",
    )
    replacement_registry = MappingProxyType(
        {
            **dict(baseline_registry),
            "value": replacement_profile,
        }
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            context_receipts_module,
            "ALL_MUTATION_OPERATOR_PROFILES",
            replacement_registry,
        )
        replacement_outcome = _exact_outcome(assignment, source_case, "value")
        assert replacement_outcome != baseline_outcome
        with pytest.raises(ValueError, match="mutation operator identity mismatch"):
            replace(
                baseline_receipt,
                outcome=replacement_outcome,
                activation_certificate=replacement_outcome.certificate,
            )


@pytest.mark.parametrize("replacement_kind", ("registry", "callable", "profile"))
def test_replay_rejects_live_substitution_after_identity_capture(
    monkeypatch,
    replacement_kind,
):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    baseline_profile = ALL_MUTATION_OPERATOR_PROFILES["value"]
    backing_registry = dict(ALL_MUTATION_OPERATOR_PROFILES)
    captured_registry = MappingProxyType(backing_registry)
    replacement_profile = _replacement_profile(
        baseline_profile,
        replacement_kind,
    )
    original_mutate = TargetPreservingMutator.mutate

    def swap_before_selected_operator(
        self,
        case,
        target_assignment,
        operator,
        *,
        allow_repair=True,
    ):
        if replacement_kind == "registry":
            monkeypatch.setattr(
                context_receipts_module,
                "ALL_MUTATION_OPERATOR_PROFILES",
                MappingProxyType(dict(backing_registry)),
            )
        else:
            backing_registry["value"] = replacement_profile
        return original_mutate(
            self,
            case,
            target_assignment,
            operator,
            allow_repair=allow_repair,
        )

    monkeypatch.setattr(
        context_receipts_module,
        "ALL_MUTATION_OPERATOR_PROFILES",
        captured_registry,
    )
    monkeypatch.setattr(
        TargetPreservingMutator,
        "mutate",
        swap_before_selected_operator,
    )
    with pytest.raises(
        ValueError,
        match="selected mutation operator context is not replayable",
    ):
        reconstruct_registered_mutation_outcome(
            source_case=source_case,
            assignment=assignment,
            operator_id="value",
        )


def test_replay_revalidates_profile_identity_between_two_fresh_replays(monkeypatch):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    baseline_profile = ALL_MUTATION_OPERATOR_PROFILES["value"]
    counting_profile = replace(
        baseline_profile,
        apply=_r4_replay_counting_apply,
    )
    counting_registry = MappingProxyType(
        {
            **dict(ALL_MUTATION_OPERATOR_PROFILES),
            "value": counting_profile,
        }
    )
    monkeypatch.setattr(
        context_receipts_module,
        "ALL_MUTATION_OPERATOR_PROFILES",
        counting_registry,
    )
    monkeypatch.setattr(__import__(__name__), "_R4_REPLAY_CALLS", 0)
    with pytest.raises(
        ValueError,
        match="selected mutation operator context is not replayable",
    ):
        reconstruct_registered_mutation_outcome(
            source_case=source_case,
            assignment=assignment,
            operator_id="value",
        )
    # The first execution changed a bound global.  Identity is checked before
    # the second execution, so the second replay never starts.
    assert _R4_REPLAY_CALLS == 1


def test_replay_revalidates_profile_identity_after_two_fresh_replays(monkeypatch):
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    baseline_registry = ALL_MUTATION_OPERATOR_PROFILES
    replacement_registry = MappingProxyType(
        {
            **dict(baseline_registry),
            "value": replace(
                baseline_registry["value"],
                semantic_family_affinity=("r4-post-replay-replacement",),
            ),
        }
    )
    original_assert = context_receipts_module._assert_registered_mutation_replay_context
    assertion_count = 0

    def swap_before_final_identity_check(context):
        nonlocal assertion_count
        assertion_count += 1
        if assertion_count == 4:
            monkeypatch.setattr(
                context_receipts_module,
                "ALL_MUTATION_OPERATOR_PROFILES",
                replacement_registry,
            )
        return original_assert(context)

    monkeypatch.setattr(
        context_receipts_module,
        "_assert_registered_mutation_replay_context",
        swap_before_final_identity_check,
    )
    with pytest.raises(
        ValueError,
        match="selected mutation operator context is not replayable",
    ):
        reconstruct_registered_mutation_outcome(
            source_case=source_case,
            assignment=assignment,
            operator_id="value",
        )
    # capture, pre-first, post-first, and post-second checks are all reached.
    assert assertion_count == 4


def test_exact_operator_outcome_states_round_trip_where_replayable():
    bundle = _receipt_bundle()
    source_case, assignment = _target_valid_mutation_context(bundle)
    witnessed: dict[str, tuple[str, object]] = {}

    for operator_id in ALL_MUTATION_OPERATOR_PROFILES:
        try:
            outcome = _exact_outcome(assignment, source_case, operator_id)
        except ValueError:
            continue
        witnessed.setdefault(outcome.status, (operator_id, outcome))
        if len(witnessed) == 3:
            break

    assert {"accepted", "rejected"} <= set(witnessed)
    assert set(witnessed) <= {"accepted", "repaired", "rejected"}
    for operator_id, outcome in witnessed.values():
        receipt = MutationAttemptReceipt.build(
            plan=bundle["plan"],
            case_id="case-000",
            case_index=0,
            assignment=assignment,
            source_case=source_case,
            operator_id=operator_id,
            outcome=outcome,
        )
        encoded = canonical_envelope(
            "MutationAttemptReceipt",
            receipt.schema_version,
            receipt,
        )
        decoded = decode_canonical_envelope(encoded)
        assert reconstruct_context_receipt(
            receipt_type=decoded["type"],
            schema_version=decoded["schema_version"],
            payload=decoded["payload"],
        ) == receipt

    with pytest.raises(ValueError, match="not registered"):
        reconstruct_registered_mutation_outcome(
            source_case=source_case,
            assignment=assignment,
            operator_id="unregistered-caller-operator",
        )


def test_cross_attempt_mutation_source_outcome_and_assignment_swaps_reject():
    bundle = _receipt_bundle()
    plan = bundle["plan"]
    _run, case_one = plan.case_run_binding("case-001", 1)
    assignment_one = TargetAssignment.build(
        selected_cell_ids=(bundle["universe"].fresh_cell_ids[0],),
        seed_lineage=case_one.seed_lineage,
    )
    obligation = bundle["focus"].obligation
    source_one = generate_case(
        case_one.seed_lineage.subseed,
        profile=formal_focus_rule(obligation.signal_id).source_generator,
    )
    mutation_one = MutationAttemptReceipt.build(
        plan=plan,
        case_id="case-001",
        case_index=1,
        assignment=assignment_one,
        source_case=source_one,
        operator_id="value",
        outcome=_exact_outcome(assignment_one, source_one, "value"),
    )
    mutation_zero = bundle["mutation"]

    with pytest.raises(ValueError, match="another formal case"):
        replace(mutation_zero, source_case=mutation_one.source_case)
    with pytest.raises(ValueError, match="outcome seed"):
        replace(mutation_zero, outcome=mutation_one.outcome)
    with pytest.raises(ValueError, match="exact formal case mapping"):
        replace(mutation_zero, assignment=assignment_one)


def test_intrinsic_focus_proof_recomputes_case_extraction_feature_and_rule():
    bundle = _receipt_bundle()
    proof = bundle["proof"]
    focus = bundle["focus"]

    assert proof.signal_id == focus.signal_id
    assert proof.extraction == AtomExtractor().extract(proof.source_case.case)

    relabeled_case = proof.source_case.case
    relabeled_case.metadata["generator_profile"] = "caller-relabeled-profile"
    relabeled_binding = CanonicalCaseBinding.build(
        binding_case_id=proof.case_id,
        case=relabeled_case,
    )
    with pytest.raises(ValueError, match="does not recompute"):
        replace(proof, source_case=relabeled_binding)
    with pytest.raises(ValueError, match="source is caller-substituted"):
        replace(
            proof,
            rule=replace(proof.rule, source_generator="caller-source"),
        )
    with pytest.raises(ValueError, match="feature is caller-substituted"):
        replace(
            proof,
            rule=replace(proof.rule, required_feature="caller:feature"),
        )
    proof_source_plan = bundle["plan"]
    other_obligation = next(
        item
        for item in proof_source_plan.registry.obligations_for(
            proof_source_plan.lane_id
        )
        if item.signal_id != proof.signal_id
    )
    other_proof = IntrinsicFocusProof.build(
        plan=proof_source_plan,
        case_id=proof.case_id,
        obligation_id=other_obligation.obligation_id,
    )
    with pytest.raises(ValueError, match="atom extraction does not recompute"):
        replace(
            proof,
            extraction=other_proof.extraction,
        )


def test_signal_mismatching_focus_proof_and_arbitrary_digest_payload_reject():
    bundle = _receipt_bundle()
    plan = bundle["plan"]
    focus = bundle["focus"]
    other_obligation = next(
        item
        for item in plan.registry.obligations_for(plan.lane_id)
        if item.signal_id != focus.signal_id
    )
    other_proof = IntrinsicFocusProof.build(
        plan=plan,
        case_id=focus.case_id,
        obligation_id=other_obligation.obligation_id,
    )
    with pytest.raises(ValueError, match="case, lineage or signal is swapped"):
        replace(focus, intrinsic_proofs=(other_proof,))

    payload = deepcopy(_payload("FocusHitReceipt", focus))
    payload["intrinsic_proofs"] = [
        {"source_kind": "ccs_ir", "evidence_digest": "arbitrary-digest"}
    ]
    result = replay_context_admission(
        receipt_type="FocusHitReceipt",
        schema_version=focus.schema_version,
        payload=payload,
        subject_kind="focus_hits",
        subject_ids=(focus.focus_context_id,),
    )
    assert result.status is ContextReplayStatus.REJECTED
    assert not result.admitted


def test_private_receipts_reconstruct_and_round_trip_canonically():
    bundle = _receipt_bundle()
    owned = (
        ("FormalLaneRegistry", bundle["registry"]),
        ("IntrinsicSearchUniverse", bundle["universe"]),
        ("FormalCaseRunBinding", bundle["case_binding"]),
        ("FormalRunBinding", bundle["run_a"]),
        ("FormalLanePlanReceipt", bundle["plan"]),
        ("CanonicalCaseBinding", bundle["mutation"].source_case),
        ("ScheduledTargetAttemptReceipt", bundle["scheduled"]),
        ("MutationAttemptReceipt", bundle["mutation"]),
        ("ObservationContextReceipt", bundle["observation"]),
        ("IntrinsicFocusProof", bundle["proof"]),
        ("FocusHitReceipt", bundle["focus"]),
    )

    for type_name, value in owned:
        encoded = canonical_envelope(type_name, value.schema_version, value)
        decoded = decode_canonical_envelope(encoded)
        reconstructed = reconstruct_context_receipt(
            receipt_type=decoded["type"],
            schema_version=decoded["schema_version"],
            payload=decoded["payload"],
        )
        assert reconstructed == value
        assert canonical_roundtrip(encoded) == encoded
        assert canonical_envelope(
            type_name, value.schema_version, reconstructed
        ) == encoded


def test_runtime_refs_are_comparison_only_and_all_runtime_subjects_fail_closed():
    bundle = _receipt_bundle()
    runtime_subjects = (
        (
            "ScheduledTargetAttemptReceipt",
            bundle["scheduled"],
            "scheduled_target_attempts",
            (bundle["scheduled"].attempt_id,),
        ),
        (
            "MutationAttemptReceipt",
            bundle["mutation"],
            "mutation_attempts",
            (bundle["mutation"].attempt_id,),
        ),
        (
            "ObservationContextReceipt",
            bundle["observation"],
            "observation_contexts",
            (bundle["observation"].context_id,),
        ),
        (
            "ObservationContextReceipt",
            bundle["complete_observation"],
            "observation_contexts",
            (bundle["complete_observation"].context_id,),
        ),
        (
            "FocusHitReceipt",
            bundle["focus"],
            "focus_hits",
            (bundle["focus"].focus_context_id,),
        ),
    )

    assert not bundle["observation"].runtime_context_complete
    assert bundle["complete_observation"].runtime_context_complete
    for type_name, value, subject_kind, subject_ids in runtime_subjects:
        result = replay_context_admission(
            receipt_type=type_name,
            schema_version=value.schema_version,
            payload=_payload(type_name, value),
            subject_kind=subject_kind,
            subject_ids=subject_ids,
        )
        assert result.status is ContextReplayStatus.CONTEXT_REQUIRED
        assert not result.authority_eligible
        assert not result.admitted


def test_caller_authority_fields_and_duplicate_subject_claims_are_rejected():
    focus = _receipt_bundle()["focus"]
    payload = deepcopy(_payload("FocusHitReceipt", focus))
    payload["authority_eligible"] = True
    result = replay_context_admission(
        receipt_type="FocusHitReceipt",
        schema_version=focus.schema_version,
        payload=payload,
        subject_kind="focus_hits",
        subject_ids=(focus.focus_context_id,),
    )
    assert result.status is ContextReplayStatus.REJECTED
    assert not result.authority_eligible

    result = replay_context_admission(
        receipt_type="FocusHitReceipt",
        schema_version=focus.schema_version,
        payload=_payload("FocusHitReceipt", focus),
        subject_kind="focus_hits",
        subject_ids=(focus.focus_context_id, focus.focus_context_id),
    )
    assert result.status is ContextReplayStatus.REJECTED
    assert result.errors == ("subject_ids_invalid",)
