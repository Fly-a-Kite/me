from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path

import pytest

import datadiff_osc
from datadiff_osc._canonical import (
    canonical_envelope,
    canonical_json,
    canonical_roundtrip,
    stable_digest,
)
from datadiff_osc.api import PUBLIC_API_SCHEMA_VERSION, PUBLIC_SYMBOL_SOURCES
from datadiff_osc.schemas import (
    AtomProvenance,
    ContractFingerprint,
    CoverageLevel,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    LedgerEvent,
    RuntimeCacheKey,
    SeedLineage,
    SeedStage,
    SemanticAtom,
    StagedComparisonResult,
    StructuredExecutionOutcome,
    VerdictKind,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import ProvenanceClass


FREEZE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "datadiff_osc"
    / "public_api_freeze.json"
)


def test_public_symbols_have_one_frozen_source_and_field_order():
    manifest = json.loads(FREEZE_PATH.read_text())
    assert manifest["schema_version"] == PUBLIC_API_SCHEMA_VERSION
    assert manifest["twenty_four_hour_run_authorized"] is False
    authority = manifest["contract_authority"]
    assert "authority RelationRegistry" in authority["contract_fingerprint_registry_digest"]
    assert "separately binds" in authority["derivation_rule_registry_digest"]
    assert "trusted implementation" in authority["evaluator_identity"]
    assert "fails closed" in authority["registry_injection"]
    assert set(manifest["symbols"]) == set(PUBLIC_SYMBOL_SOURCES)
    for name, source in PUBLIC_SYMBOL_SOURCES.items():
        value = getattr(datadiff_osc, name)
        frozen = manifest["symbols"][name]
        assert frozen["source"] == source
        assert value.__module__ == source
        if isinstance(value, type) and is_dataclass(value):
            assert value.__dataclass_params__.frozen
            assert frozen["fields"] == [field.name for field in fields(value)]
        elif isinstance(value, type) and issubclass(value, Enum):
            assert frozen["values"] == [member.value for member in value]


def test_canonical_encoding_tags_bytes_rejects_key_coercion_and_round_trips():
    provenance = AtomProvenance(
        source_kind="ccs_ir",
        source_digest="program-1",
        source_path="steps[0]",
        evidence_digest="evidence-1",
    )
    atom = SemanticAtom("op:join", "op", "join", (provenance,))
    payload = canonical_envelope(
        "datadiff_osc.schemas.SemanticAtom",
        atom.schema_version,
        {"atom": atom, "binary": b"\x00\xff"},
    )
    assert canonical_roundtrip(payload) == payload
    assert '"$bytes":"AP8="' in payload
    assert stable_digest("smoke", atom) == stable_digest("smoke", atom)
    with pytest.raises(TypeError, match="string keys"):
        canonical_json({1: "integer", "1": "string"})


def test_seed_lineage_is_stage_keyed_and_reproducible():
    base = dict(
        protocol_digest="protocol-v1",
        master_seed=31000001,
        lane_id="lane-a",
        case_index=7,
    )
    target_a = SeedLineage(stage_name=SeedStage.TARGET, **base)
    target_b = SeedLineage(stage_name=SeedStage.TARGET, **base)
    data = SeedLineage(stage_name=SeedStage.DATA, **base)
    assert target_a.digest == target_b.digest
    assert target_a.subseed == target_b.subseed
    assert target_a.digest != data.digest
    assert target_a.subseed != data.subseed


def test_execution_outcomes_preserve_failure_taxonomy():
    ok = StructuredExecutionOutcome(
        endpoint_id="e-ok",
        status=ExecutionStatus.OK,
        failure_kind=FailureKind.NONE,
    )
    unsupported = StructuredExecutionOutcome(
        endpoint_id="e-unsupported",
        status=ExecutionStatus.UNSUPPORTED,
        failure_kind=FailureKind.UNSUPPORTED_CAPABILITY,
        unsupported_evidence_digest="unsupported-evidence-1",
    )
    assert ok.digest != unsupported.digest
    with pytest.raises(ValueError, match="requires structured capability evidence"):
        StructuredExecutionOutcome(
            endpoint_id="e",
            status=ExecutionStatus.UNSUPPORTED,
            failure_kind=FailureKind.UNSUPPORTED_CAPABILITY,
        )
    with pytest.raises(ValueError, match="requires failure kind timeout"):
        StructuredExecutionOutcome(
            endpoint_id="e",
            status=ExecutionStatus.TIMEOUT,
            failure_kind=FailureKind.NONE,
        )


def test_observed_ledger_event_requires_all_certificate_bindings():
    with pytest.raises(ValueError, match="activation_certificate_digest"):
        LedgerEvent(
            task_id="task-1",
            level=CoverageLevel.OBSERVED,
            event_order_key=("0001",),
            cell_ids=("cell-1",),
        )
    event = LedgerEvent(
        task_id="task-1",
        level=CoverageLevel.OBSERVED,
        event_order_key=("0001",),
        cell_ids=("cell-1",),
        activation_certificate_digest="activation-1",
        applicability_certificate_digest="applicability-1",
        observation_certificate_digest="observation-1",
        execution_outcome_digests=("outcome-1",),
    )
    assert event.event_id.startswith("osc-ledger-event-id-")


def test_non_screening_comparison_is_exact_and_confirmation_cache_is_forbidden():
    contract = ContractFingerprint("contract-1", "registry-1")
    with pytest.raises(ValueError, match="must use exact authority"):
        StagedComparisonResult(
            request_digest="request-1",
            plan_digest="plan-1",
            observation_certificate_digest="observation-1",
            evidence_tier=EvidenceTier.FINDING,
            comparison_stage="S2_COMPONENT_FINGERPRINT",
            exact_escalated=False,
            endpoint_order=("left", "right"),
            component_fingerprint_digests=("fingerprint-1",),
            verdict_kind=VerdictKind.SATISFIED,
        )
    key = RuntimeCacheKey(
        endpoint_digest="endpoint-1",
        contract_fingerprint=contract,
        semantic_schema_version="osc-semantic-target-v1",
        backend="backend",
        backend_version="1.0",
        adapter_revision="adapter-v1",
        execution_mode="eager",
        physical_layout="contiguous",
        evidence_tier=EvidenceTier.FRESH_CONFIRMATION,
    )
    assert key.cache_allowed is False


def test_fresh_and_regression_obligation_denominators_are_typed_and_disjoint():
    universe = compile_target_universe(legacy_v4_target_templates())
    assert len(universe.fresh_cells) == 232
    assert len(universe.regression_cells) == 144
    assert len(universe.fresh_edges) == 384
    assert len(universe.fresh_backend_pair_obligations) == 502
    assert len(universe.regression_backend_pair_obligations) == 228
    assert all(
        item.provenance_class == ProvenanceClass.FRESH_DISCOVERY
        for item in universe.fresh_backend_pair_obligations
    )
    assert all(
        item.provenance_class == ProvenanceClass.KNOWN_REGRESSION
        for item in universe.regression_backend_pair_obligations
    )
    assert {
        item.obligation_id for item in universe.fresh_backend_pair_obligations
    }.isdisjoint(
        item.obligation_id
        for item in universe.regression_backend_pair_obligations
    )
