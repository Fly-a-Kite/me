from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from datadiff_osc.runtime.gate_artifacts import (
    GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
    RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    MetricReducer,
    identity_universe_digest,
    metric_series_payload,
    recompute_gate_metrics,
    verify_gate_artifact_manifest,
    verify_gate_authority_plan,
)


SOURCE = "frozen-source-snapshot-1"


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_bundle(
    tmp_path,
    *,
    series,
    claims,
    universes,
    source_digest=SOURCE,
    manifest_source_digest=None,
    artifact_source_digest=None,
    artifact_schema=RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    reference_schema=RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    reference_hash=None,
    reference_path="raw.json",
    authority_artifact_hash=None,
    gate_spec_set_digest="test-gate-spec-set",
):
    artifact_path = tmp_path / "raw.json"
    artifact_bytes = _json_bytes(
        {
            "schema_version": artifact_schema,
            "artifact_id": "raw-a",
            "source_digest": artifact_source_digest or source_digest,
            "series": list(series),
        }
    )
    artifact_path.write_bytes(artifact_bytes)
    actual_artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    manifest = {
        "schema_version": GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "source_digest": manifest_source_digest or source_digest,
        "claimed_metrics": claims,
        "artifacts": [
            {
                "artifact_id": "raw-a",
                "path": reference_path,
                "sha256": reference_hash or actual_artifact_hash,
                "schema_version": reference_schema,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))

    authority_payload = {
        "schema_version": GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
        "source_digest": source_digest,
        "gate_spec_set_digest": gate_spec_set_digest,
        "artifacts": [
            {
                "artifact_id": "raw-a",
                "sha256": authority_artifact_hash or actual_artifact_hash,
                "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
                "provenance_digest": "root-artifact-index-1",
            }
        ],
        "denominator_universes": [
            {
                "metric_key": metric_key,
                "cardinality": len(tuple(record_ids)),
                "identity_digest": identity_universe_digest(record_ids),
                "provenance_digest": f"root-declaration-{metric_key}",
            }
            for metric_key, record_ids in sorted(universes.items())
        ],
    }
    authority_path = tmp_path / "authority.json"
    authority_bytes = _json_bytes(authority_payload)
    authority_path.write_bytes(authority_bytes)
    authority_sha = hashlib.sha256(authority_bytes).hexdigest()
    authority = verify_gate_authority_plan(
        authority_path,
        expected_sha256=authority_sha,
        expected_source_digest=source_digest,
    )
    evidence = verify_gate_artifact_manifest(
        manifest_path,
        expected_source_digest=source_digest,
        expected_authority_plan_sha256=authority_sha,
        authority_plan=authority,
    )
    return authority, evidence, artifact_path, authority_path


def _recompute(evidence, authority, reducers, denominators=("count",)):
    return recompute_gate_metrics(
        evidence,
        reducers,
        expected_source_digest=SOURCE,
        expected_authority_plan_sha256=authority.expected_plan_sha256,
        authority_plan=authority,
        required_denominator_keys=denominators,
    )


def test_external_authority_plan_and_raw_recomputation_pass_when_all_bindings_match(
    tmp_path,
):
    count_ids = ("a", "b")
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1), ("b", 1))
            ),
            metric_series_payload(
                "p95",
                MetricReducer.NEAREST_RANK_P95,
                (("a", 1.0), ("b", 2.0), ("c", 100.0)),
            ),
        ),
        claims={"count": 2, "p95": 100.0},
        universes={"count": count_ids},
    )
    recomputed = _recompute(
        evidence,
        authority,
        {
            "count": MetricReducer.COUNT_UNIQUE,
            "p95": MetricReducer.NEAREST_RANK_P95,
        },
    )
    assert authority.valid
    assert evidence.valid
    assert recomputed.errors == ()
    assert recomputed.metric_map() == {"count": 2, "p95": 100.0}


def test_self_asserted_manifest_and_raw_source_cannot_replace_expected_source(tmp_path):
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("fake", 1),)
            ),
        ),
        claims={"count": 1},
        universes={"count": ("fake",)},
        manifest_source_digest="attacker-self-asserted-source",
        artifact_source_digest="attacker-self-asserted-source",
    )
    recomputed = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    )
    assert evidence.valid is False
    assert "gate_manifest_expected_source_mismatch" in recomputed.errors
    assert any("expected source digest mismatch" in item for item in recomputed.errors)


def test_same_cardinality_fake_denominator_fails_exact_identity_commitment(tmp_path):
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count",
                MetricReducer.COUNT_UNIQUE,
                (("not-real-a", 1), ("not-real-b", 1)),
            ),
        ),
        claims={"count": 2},
        universes={"count": ("real-a", "real-b")},
    )
    recomputed = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    )
    assert evidence.valid is False
    assert "trusted_denominator_universe_mismatch:count" in recomputed.errors


def test_raw_artifact_must_be_precommitted_by_independent_authority_plan(tmp_path):
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1),)
            ),
        ),
        claims={"count": 1},
        universes={"count": ("a",)},
        authority_artifact_hash="0" * 64,
    )
    recomputed = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    )
    assert authority.valid
    assert evidence.valid is False
    assert any("authority_artifact_hash_mismatch" in item for item in recomputed.errors)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"reference_hash": "0" * 64}, "authority_artifact_hash_mismatch"),
        ({"reference_path": "missing.json"}, "artifact_missing"),
        ({"reference_schema": "wrong-schema"}, "authority_artifact_schema_mismatch"),
        ({"artifact_schema": "wrong-schema"}, "raw artifact schema version mismatch"),
        (
            {"artifact_source_digest": "other-source"},
            "raw artifact expected source digest mismatch",
        ),
    ],
)
def test_missing_hash_schema_and_source_forgery_fail_closed(
    tmp_path, overrides, error
):
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1),)
            ),
        ),
        claims={"count": 1},
        universes={"count": ("a",)},
        **overrides,
    )
    recomputed = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    )
    assert evidence.valid is False
    assert any(error in item for item in recomputed.errors)


def test_claimed_metric_mismatch_and_reducer_substitution_fail_closed(tmp_path):
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1), ("b", 1))
            ),
        ),
        claims={"count": 1},
        universes={"count": ("a", "b")},
    )
    mismatch = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    )
    substituted = _recompute(
        evidence, authority, {"count": MetricReducer.MAXIMUM}
    )
    assert "claimed_metric_mismatch:count" in mismatch.errors
    assert "metric_reducer_mismatch:count" in substituted.errors


def test_authority_and_artifact_are_both_reverified_after_mutation(tmp_path):
    authority, evidence, artifact_path, authority_path = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1),)
            ),
        ),
        claims={"count": 1},
        universes={"count": ("a",)},
    )
    artifact_path.write_text("{}", encoding="utf-8")
    assert evidence.valid is False
    artifact_errors = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    ).errors
    assert "verified_evidence_snapshot_mismatch" in artifact_errors

    authority_path.write_text("{}", encoding="utf-8")
    assert authority.valid is False
    authority_errors = _recompute(
        evidence, authority, {"count": MetricReducer.COUNT_UNIQUE}
    ).errors
    assert "authority_plan_snapshot_mismatch" in authority_errors


def test_manual_typed_object_rebinding_cannot_bypass_file_reverification(tmp_path):
    authority, evidence, _, _ = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1),)
            ),
        ),
        claims={"count": 1},
        universes={"count": ("a",)},
    )
    forged_authority = replace(
        authority,
        source_digest="attacker-source",
        expected_plan_sha256="0" * 64,
    )
    forged_evidence = replace(evidence, authority_plan=forged_authority)
    recomputed = recompute_gate_metrics(
        forged_evidence,
        {"count": MetricReducer.COUNT_UNIQUE},
        expected_source_digest=SOURCE,
        expected_authority_plan_sha256=authority.expected_plan_sha256,
        authority_plan=forged_authority,
        required_denominator_keys=("count",),
    )
    assert "authority_plan_snapshot_mismatch" in recomputed.errors
    assert "authority_plan_rebound_to_different_external_sha256" in recomputed.errors
    assert "verified_evidence_snapshot_mismatch" in recomputed.errors


def test_authority_plan_external_sha_and_source_are_not_self_asserted(tmp_path):
    authority, _, _, authority_path = _write_bundle(
        tmp_path,
        series=(
            metric_series_payload(
                "count", MetricReducer.COUNT_UNIQUE, (("a", 1),)
            ),
        ),
        claims={"count": 1},
        universes={"count": ("a",)},
    )
    wrong_hash = verify_gate_authority_plan(
        authority_path,
        expected_sha256="0" * 64,
        expected_source_digest=SOURCE,
    )
    wrong_source = verify_gate_authority_plan(
        authority_path,
        expected_sha256=authority.byte_sha256,
        expected_source_digest="other-frozen-source",
    )
    assert wrong_hash.valid is False
    assert "authority_plan_hash_mismatch" in wrong_hash.verification_errors
    assert wrong_source.valid is False
    assert "authority_plan_expected_source_mismatch" in wrong_source.verification_errors
