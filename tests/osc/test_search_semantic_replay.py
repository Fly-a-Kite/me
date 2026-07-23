from __future__ import annotations

from copy import deepcopy
import inspect
import json

import pytest

from datadiff_osc._canonical import canonical_json
from datadiff_osc.generation.construction import BackwardConstructor
from datadiff_osc.generation.extraction import AtomExtractor, merge_extractions
from datadiff_osc.generation.mutation import MutationOutcome
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.search.semantic_replay import (
    FormalLanePlan,
    _FormalFocusPlan,
    _reconstruct_search_payload,
    replay_search_admission,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


@pytest.fixture(scope="module")
def replay_values():
    templates = legacy_v4_target_templates()
    universe = compile_target_universe(templates)
    cell = universe.fresh_cells[0]
    assignment = TargetAssignment(
        (cell.target_cell_id,),
        SeedLineage(
            "search-replay-protocol",
            17,
            "search-replay-lane",
            cell.construction_index,
            SeedStage.CONSTRUCTOR,
        ),
    )
    construction = BackwardConstructor(universe, templates).construct(assignment)
    assert construction.successful
    assert construction.case is not None
    assert construction.certificate is not None
    extraction = AtomExtractor().extract(construction.case)
    edge = universe.fresh_edges[0]
    contrast = merge_extractions(
        (
            (edge.base_cell_id, extraction),
            (edge.sibling_cell_id, extraction),
        )
    )
    mutation = MutationOutcome(
        construction.case,
        construction.certificate,
        "accepted",
        "target_preserved",
        1,
        991,
    )
    case_ids = tuple(f"case-{index:03d}" for index in range(10))
    lane = FormalLanePlan(
        protocol_digest="protocol-digest",
        lane_registry_digest="lane-registry-digest",
        lane_id="lane-a",
        planned_case_ids=case_ids,
        focus_plans=(
            _FormalFocusPlan("signal-a", case_ids[:5]),
            _FormalFocusPlan("signal-b", case_ids[5:]),
        ),
    )
    return {
        "universe": universe,
        "cell": cell,
        "assignment": assignment,
        "extraction": extraction,
        "contrast": contrast,
        "certificate": construction.certificate,
        "construction": construction,
        "mutation": mutation,
        "lane": lane,
    }


def _replay(
    envelope_type: str,
    schema_version: str,
    value: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return replay_search_admission(
        envelope_type=envelope_type,
        schema_version=schema_version,
        payload=_payload(value),
        subject_kind=subject_kind,
        subject_ids=subject_ids,
    )


def _payload(value: object) -> object:
    """Match the JSON primitives Root passes after canonical decoding."""

    return json.loads(canonical_json(value))


def test_replay_search_admission_signature_is_frozen():
    signature = inspect.signature(replay_search_admission)
    assert tuple(signature.parameters) == (
        "envelope_type",
        "schema_version",
        "payload",
        "subject_kind",
        "subject_ids",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(
    ("envelope_type", "schema_version", "key"),
    (
        ("ConstructionOutcome", "osc-root-construction-outcome-v1", "construction"),
        ("MutationOutcome", "osc-root-mutation-outcome-v1", "mutation"),
        ("TargetAssignment", "osc-root-target-assignment-v1", "assignment"),
        ("ExtractionResult", "osc-atom-extraction-v1", "extraction"),
        ("ContrastExtraction", "osc-contrast-extraction-v1", "contrast"),
        ("ActivationCertificate", "osc-activation-certificate-v1", "certificate"),
        ("FormalLanePlan", "osc-root-formal-lane-plan-v1", "lane"),
    ),
)
def test_all_search_types_strictly_reconstruct_and_roundtrip(
    replay_values, envelope_type, schema_version, key
):
    value = replay_values[key]
    assert (
        _reconstruct_search_payload(
            envelope_type=envelope_type,
            schema_version=schema_version,
            payload=_payload(value),
        )
        == value
    )


def test_target_assignment_derives_only_compiler_owned_subjects(replay_values):
    assignment = replay_values["assignment"]
    cell = replay_values["cell"]
    assert (
        _replay(
            "TargetAssignment",
            "osc-root-target-assignment-v1",
            assignment,
            "fresh_cells_declared",
            (cell.target_cell_id,),
        )
        == ()
    )
    assert (
        _replay(
            "TargetAssignment",
            "osc-root-target-assignment-v1",
            assignment,
            "fresh_families_declared",
            (cell.test_family_id,),
        )
        == ()
    )


def test_target_assignment_restores_tagged_and_nested_immutable_parameters(
    replay_values,
):
    base = replay_values["assignment"]
    assignment = TargetAssignment.build(
        selected_cell_ids=base.selected_cell_ids,
        seed_lineage=base.seed_lineage,
        parameters={
            "binary": b"\x00\xff",
            "negative_zero": -0.0,
            "nested": (1, ("key", "value")),
        },
    )
    reconstructed = _reconstruct_search_payload(
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        payload=_payload(assignment),
    )
    assert reconstructed == assignment


def test_extraction_and_contrast_derive_intrinsic_dimensions(replay_values):
    extraction = replay_values["extraction"]
    operations = tuple(
        sorted({atom.value for atom in extraction.atoms if atom.namespace == "op"})
    )
    assert operations
    assert (
        _replay(
            "ExtractionResult",
            "osc-atom-extraction-v1",
            extraction,
            "operations_declared",
            operations,
        )
        == ()
    )
    contrast = replay_values["contrast"]
    contrast_operations = tuple(
        sorted({atom.value for atom in contrast.atoms if atom.namespace == "op"})
    )
    assert (
        _replay(
            "ContrastExtraction",
            "osc-contrast-extraction-v1",
            contrast,
            "operations_declared",
            contrast_operations,
        )
        == ()
    )


def test_case_bound_outcomes_can_derive_local_cell_activation(replay_values):
    cell_id = replay_values["cell"].target_cell_id
    assert (
        _replay(
            "ConstructionOutcome",
            "osc-root-construction-outcome-v1",
            replay_values["construction"],
            "cells_activation_denominator",
            (cell_id,),
        )
        == ()
    )
    assert (
        _replay(
            "MutationOutcome",
            "osc-root-mutation-outcome-v1",
            replay_values["mutation"],
            "cells_activation_denominator",
            (cell_id,),
        )
        == ()
    )


def test_caller_subject_substitution_is_rejected(replay_values):
    errors = _replay(
        "TargetAssignment",
        "osc-root-target-assignment-v1",
        replay_values["assignment"],
        "fresh_cells_declared",
        (replay_values["universe"].fresh_cells[1].target_cell_id,),
    )
    assert len(errors) == 1
    assert errors[0].startswith("search_replay_subject_mismatch:fresh_cells_declared:")


@pytest.mark.parametrize(
    "subject_kind",
    (
        "scheduled_target_attempts",
        "mutation_attempts",
        "edge_observation_denominator",
        "edges_activation_denominator",
        "fresh_cells_reachability_denominator",
        "fresh_family_reachability_denominator",
        "mutation_family_denominator",
        "scheduled_activation_family_denominator",
    ),
)
def test_attempt_observation_and_reachability_subjects_fail_closed(
    replay_values, subject_kind
):
    errors = _replay(
        "ConstructionOutcome",
        "osc-root-construction-outcome-v1",
        replay_values["construction"],
        subject_kind,
        ("untrusted-subject",),
    )
    assert errors == (f"search_replay_context_required:{subject_kind}",)


@pytest.mark.parametrize("subject_kind", ("formal_lanes", "focus_signals"))
def test_lane_and_focus_subjects_fail_closed_pending_frozen_registry(
    replay_values, subject_kind
):
    errors = _replay(
        "FormalLanePlan",
        "osc-root-formal-lane-plan-v1",
        replay_values["lane"],
        subject_kind,
        ("caller-authored",),
    )
    assert errors == (f"search_replay_context_required:{subject_kind}",)


@pytest.mark.parametrize(
    "subject_kind",
    ("confirmed_roots_total", "candidate_records", "bug_claims", "gate_pass", "24h"),
)
def test_search_replay_never_creates_verdict_candidate_bug_gate_or_24h_authority(
    replay_values, subject_kind
):
    errors = _replay(
        "TargetAssignment",
        "osc-root-target-assignment-v1",
        replay_values["assignment"],
        subject_kind,
        ("forbidden",),
    )
    assert errors == (f"search_replay_authority_forbidden:{subject_kind}",)


def test_unknown_type_version_and_extra_fields_fail_closed(replay_values):
    payload = _payload(replay_values["assignment"])
    errors = replay_search_admission(
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v2",
        payload=payload,
        subject_kind="fresh_cells_declared",
        subject_ids=(replay_values["cell"].target_cell_id,),
    )
    assert "unsupported Search type/version" in errors[0]

    forged = deepcopy(payload)
    forged["extra"] = "shape-only"
    errors = replay_search_admission(
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        payload=forged,
        subject_kind="fresh_cells_declared",
        subject_ids=(replay_values["cell"].target_cell_id,),
    )
    assert "field mismatch" in errors[0]


def test_assignment_unknown_target_and_misdirected_edge_fail_closed(replay_values):
    payload = _payload(replay_values["assignment"])
    unknown = deepcopy(payload)
    unknown["selected_cell_ids"] = ["unknown-cell"]
    errors = replay_search_admission(
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        payload=unknown,
        subject_kind="fresh_cells_declared",
        subject_ids=("unknown-cell",),
    )
    assert "unknown target cells" in errors[0]

    edge = replay_values["universe"].fresh_edges[0]
    wrong_direction = deepcopy(payload)
    wrong_direction["selected_cell_ids"] = [edge.sibling_cell_id, edge.base_cell_id]
    wrong_direction["selected_edge_ids"] = [edge.contrast_edge_id]
    errors = replay_search_admission(
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        payload=wrong_direction,
        subject_kind="contrast_edges_declared",
        subject_ids=(edge.contrast_edge_id,),
    )
    assert "base then sibling" in errors[0]


def test_extraction_provenance_and_contrast_merge_forgery_fail_closed(replay_values):
    extraction = _payload(replay_values["extraction"])
    forged_source = deepcopy(extraction)
    forged_source["atoms"][0]["provenance"][0]["source_digest"] = "forged-source"
    errors = replay_search_admission(
        envelope_type="ExtractionResult",
        schema_version="osc-atom-extraction-v1",
        payload=forged_source,
        subject_kind="operations_declared",
        subject_ids=("filter",),
    )
    assert "not bound to extraction source" in errors[0]

    forged_metadata = deepcopy(extraction)
    forged_metadata["atoms"][0]["provenance"][0]["source_kind"] = "case_metadata"
    errors = replay_search_admission(
        envelope_type="ExtractionResult",
        schema_version="osc-atom-extraction-v1",
        payload=forged_metadata,
        subject_kind="operations_declared",
        subject_ids=("filter",),
    )
    assert "metadata cannot provide semantic evidence" in errors[0]

    contrast = _payload(replay_values["contrast"])
    forged_merge = deepcopy(contrast)
    forged_merge["source_digest"] = "forged-contrast-source"
    errors = replay_search_admission(
        envelope_type="ContrastExtraction",
        schema_version="osc-contrast-extraction-v1",
        payload=forged_merge,
        subject_kind="operations_declared",
        subject_ids=("filter",),
    )
    assert "do not recompute exactly" in errors[0]


def test_activation_fingerprint_and_case_certificate_substitution_fail_closed(
    replay_values,
):
    certificate = _payload(replay_values["certificate"])
    forged_fingerprint = deepcopy(certificate)
    forged_fingerprint["target_fingerprint"]["universe_digest"] = "forged-universe"
    errors = replay_search_admission(
        envelope_type="ActivationCertificate",
        schema_version="osc-activation-certificate-v1",
        payload=forged_fingerprint,
        subject_kind="fresh_cells_declared",
        subject_ids=(replay_values["cell"].target_cell_id,),
    )
    assert "does not match the frozen compiler universe" in errors[0]

    construction = _payload(replay_values["construction"])
    forged_case = deepcopy(construction)
    forged_case["case"]["tables"][0]["rows"][0][
        next(iter(forged_case["case"]["tables"][0]["rows"][0]))
    ] = "forged-value"
    errors = replay_search_admission(
        envelope_type="ConstructionOutcome",
        schema_version="osc-root-construction-outcome-v1",
        payload=forged_case,
        subject_kind="cells_activation_denominator",
        subject_ids=(replay_values["cell"].target_cell_id,),
    )
    assert any(
        detail in errors[0]
        for detail in (
            "independent Case extraction",
            "invalid Case",
            "does not round-trip exactly",
        )
    )


def test_mutation_state_and_formal_focus_case_forgery_fail_closed(replay_values):
    mutation = _payload(replay_values["mutation"])
    mutation["status"] = "repaired"
    errors = replay_search_admission(
        envelope_type="MutationOutcome",
        schema_version="osc-root-mutation-outcome-v1",
        payload=mutation,
        subject_kind="cells_activation_denominator",
        subject_ids=(replay_values["cell"].target_cell_id,),
    )
    assert "repaired mutation state is inconsistent" in errors[0]

    lane = _payload(replay_values["lane"])
    lane["focus_plans"][0]["planned_case_ids"][4] = "case-004x"
    errors = replay_search_admission(
        envelope_type="FormalLanePlan",
        schema_version="osc-root-formal-lane-plan-v1",
        payload=lane,
        subject_kind="formal_lanes",
        subject_ids=("lane-a",),
    )
    assert "formal focus cases must belong" in errors[0]


def test_subject_ids_must_be_nonempty_unique_sorted_strings(replay_values):
    value = replay_values["assignment"]
    common = {
        "envelope_type": "TargetAssignment",
        "schema_version": "osc-root-target-assignment-v1",
        "payload": _payload(value),
        "subject_kind": "fresh_cells_declared",
    }
    assert replay_search_admission(**common, subject_ids=()) == (
        "search_replay_subject_ids_invalid",
    )
    assert replay_search_admission(**common, subject_ids=("b", "a")) == (
        "search_replay_subject_ids_invalid",
    )
    assert replay_search_admission(**common, subject_ids=("a", "a")) == (
        "search_replay_subject_ids_invalid",
    )
