from __future__ import annotations

from dataclasses import replace
import json

import pytest

import datadiff_osc
from datadiff_osc._canonical import canonical_json, stable_digest
from datadiff_osc.runtime._phase6_v3_run_provenance import (
    V3_RUN_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    V3RunProductionProvenance,
    build_v3_run_production_provenance,
    canonical_v3_run_production_provenance_envelope,
    reconstruct_v3_run_production_provenance_payload,
    reconstruct_v3_run_receipt_payload,
    v3_run_execution_transcript_digest,
)
from datadiff_osc.runtime._phase6_v3_run_receipt_builder import (
    v3_case_pipeline_digest,
    v3_execution_run_digest,
)
from datadiff_osc.runtime._private_receipts import (
    V3BackendTaskBinding,
    V3CaseBinding,
    V3RunReceipt,
)
from datadiff_osc.schemas import ExecutionStatus


def _digest(label: str) -> str:
    return stable_digest("phase6-v3-run-provenance-test", label)


def _case(case_index: int, *, id_prefix: str = "case") -> V3CaseBinding:
    suffix = f"{case_index:03d}"
    case_id = f"{id_prefix}-{suffix}"
    case_task_id = f"case-task-{suffix}"
    result_digest = _digest(f"result-{suffix}")
    execution_outcome_digest = _digest(f"case-outcome-{suffix}")
    evidence_envelope_digest = _digest(f"evidence-{suffix}")
    backend = V3BackendTaskBinding(
        task_id=f"backend-task-{suffix}",
        endpoint_digest=_digest(f"endpoint-{suffix}"),
        task_spec_digest=_digest(f"backend-spec-{suffix}"),
        execution_outcome_digest=_digest(f"backend-outcome-{suffix}"),
        status=ExecutionStatus.OK,
    )
    return V3CaseBinding(
        case_id=case_id,
        case_index=case_index,
        seed_lineage_digest=_digest(f"seed-{suffix}"),
        case_task_id=case_task_id,
        case_task_spec_digest=_digest(f"case-spec-{suffix}"),
        result_digest=result_digest,
        pipeline_digest=v3_case_pipeline_digest(
            case_id=case_id,
            case_task_id=case_task_id,
            result_digest=result_digest,
            execution_outcome_digest=execution_outcome_digest,
            evidence_envelope_digest=evidence_envelope_digest,
            backend_task_ids=(backend.task_id,),
        ),
        execution_outcome_digest=execution_outcome_digest,
        evidence_envelope_digest=evidence_envelope_digest,
        executed=True,
        iteration_failure=False,
        pipeline_error=False,
        backend_tasks=(backend,),
    )


def _receipt(
    *,
    source_digest: str | None = None,
    lane_id: str = "lane-00",
    id_prefix: str = "case",
) -> V3RunReceipt:
    cases = tuple(_case(index, id_prefix=id_prefix) for index in range(100))
    plan_digest = _digest("v3-gate-plan")
    run_id = "run-000"
    seed = 30700001
    return V3RunReceipt(
        source_digest=source_digest or _digest("source"),
        protocol_digest=_digest("protocol"),
        v3_gate_plan_digest=plan_digest,
        run_id=run_id,
        lane_id=lane_id,
        seed=seed,
        execution_run_digest=v3_execution_run_digest(
            v3_gate_plan_digest=plan_digest,
            lane_id=lane_id,
            seed=seed,
            run_id=run_id,
            case_ids=tuple(item.case_id for item in cases),
            case_result_digests=tuple(item.result_digest for item in cases),
        ),
        run_completed=True,
        cases=cases,
    )


def _provenance(
    *, source_digest: str | None = None
) -> V3RunProductionProvenance:
    return build_v3_run_production_provenance(
        receipt=_receipt(source_digest=source_digest)
    )


def test_private_v3_run_provenance_rebuilds_receipt_without_public_authority():
    provenance = _provenance()
    envelope = canonical_v3_run_production_provenance_envelope(provenance)
    decoded = json.loads(envelope)

    assert "V3RunProductionProvenance" not in datadiff_osc.__all__
    assert decoded["type"] == V3_RUN_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
    assert decoded["schema_version"] == (
        V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION
    )
    assert provenance.source_snapshot_digest == provenance.receipt.source_digest
    assert provenance.execution_transcript_digest == (
        v3_run_execution_transcript_digest(provenance.receipt)
    )
    assert provenance.provenance_id
    assert reconstruct_v3_run_production_provenance_payload(
        decoded["payload"]
    ) == provenance

    changed = _provenance(source_digest=_digest("other-source"))
    assert changed.receipt.digest != provenance.receipt.digest
    assert changed.provenance_id != provenance.provenance_id


def test_private_v3_run_receipt_payload_roundtrips_and_fails_closed():
    receipt = _receipt()
    payload = json.loads(canonical_json(receipt))

    assert reconstruct_v3_run_receipt_payload(payload) == receipt

    payload["cases"][0]["case_id"] = ""
    with pytest.raises((ValueError, TypeError)):
        reconstruct_v3_run_receipt_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.pop("receipt"),
        lambda payload: payload.update({"caller_digest": _digest("forged")}),
        lambda payload: payload["receipt"].update(
            {"source_digest": _digest("forged-source")}
        ),
        lambda payload: payload["receipt"].update(
            {"execution_run_digest": _digest("forged-run")}
        ),
        lambda payload: payload["receipt"]["cases"][0].update(
            {"pipeline_digest": _digest("forged-pipeline")}
        ),
        lambda payload: payload.update(
            {"execution_transcript_digest": _digest("forged-transcript")}
        ),
        lambda payload: payload.update({"producer_module_sha256": "0" * 64}),
        lambda payload: payload.update({"bridge_module_sha256": "not-a-sha"}),
        lambda payload: payload.update({"schema_version": "osc-forged-v1"}),
    ),
)
def test_private_v3_run_provenance_rejects_malformed_or_misbound_payload(mutate):
    provenance = _provenance()
    payload = json.loads(canonical_json(provenance))

    mutate(payload)

    with pytest.raises((ValueError, TypeError)):
        reconstruct_v3_run_production_provenance_payload(payload)


def test_private_v3_run_provenance_rejects_wrong_object_and_module_binding():
    with pytest.raises(TypeError, match="V3RunReceipt"):
        build_v3_run_production_provenance(
            receipt=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="V3RunProductionProvenance"):
        canonical_v3_run_production_provenance_envelope(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="implementation binding mismatch"):
        replace(_provenance(), producer_module_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        (
            lambda provenance: replace(provenance, schema_version="osc-forged-v1"),
            "schema version mismatch",
        ),
        (
            lambda provenance: replace(
                provenance, source_snapshot_digest=_digest("other-source")
            ),
            "source binding mismatch",
        ),
        (
            lambda provenance: replace(
                provenance,
                execution_transcript_digest=_digest("forged-transcript"),
            ),
            "transcript binding mismatch",
        ),
        (
            lambda provenance: replace(provenance, producer_module_sha256="0" * 64),
            "implementation binding mismatch:producer_module_sha256",
        ),
        (
            lambda provenance: replace(provenance, bridge_module_sha256="0" * 64),
            "implementation binding mismatch:bridge_module_sha256",
        ),
        (
            lambda provenance: replace(
                provenance,
                receipt=replace(
                    provenance.receipt,
                    execution_run_digest=_digest("forged-run"),
                ),
            ),
            "execution run digest binding mismatch",
        ),
    ),
)
def test_private_v3_run_provenance_rejects_every_field_rebinding(
    mutate, expected_error
):
    provenance = _provenance()

    with pytest.raises(ValueError, match=expected_error):
        mutate(provenance)


def test_private_v3_run_provenance_rejects_receipt_pipeline_forgery():
    receipt = _receipt()
    forged_case = replace(
        receipt.cases[9], pipeline_digest=_digest("forged-pipeline")
    )
    forged_receipt = replace(
        receipt,
        cases=receipt.cases[:9] + (forged_case,) + receipt.cases[10:],
    )

    with pytest.raises(ValueError, match="pipeline digest binding mismatch"):
        build_v3_run_production_provenance(receipt=forged_receipt)


def test_private_v3_run_provenance_rejects_marker_injection_with_clean_node_ids():
    # Every case identity stays clean; the marker rides only on the lane.
    lane_receipt = _receipt(lane_id="lane-synthetic-00")
    assert all("synthetic" not in item.case_id for item in lane_receipt.cases)
    with pytest.raises(ValueError, match="synthetic marker"):
        build_v3_run_production_provenance(receipt=lane_receipt)

    # And symmetrically with clean run identity but marked case identities.
    case_receipt = _receipt(id_prefix="Synthetic-case")
    assert "synthetic" not in case_receipt.lane_id
    with pytest.raises(ValueError, match="synthetic marker"):
        build_v3_run_production_provenance(receipt=case_receipt)


def test_private_v3_run_provenance_rejects_non_receipt_object():
    provenance = _provenance()

    with pytest.raises(ValueError, match="requires a V3RunReceipt"):
        V3RunProductionProvenance(
            receipt=None,  # type: ignore[arg-type]
            source_snapshot_digest=provenance.source_snapshot_digest,
            execution_transcript_digest=provenance.execution_transcript_digest,
            producer_module_sha256=provenance.producer_module_sha256,
            bridge_module_sha256=provenance.bridge_module_sha256,
        )
