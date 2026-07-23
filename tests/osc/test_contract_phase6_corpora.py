from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from datadiff.dsl import Case, Program
from datadiff_osc._canonical import canonical_json, to_primitive
from datadiff_osc.contract_engine import _phase6_corpora as corpora
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceEnvelope,
    EvidenceState,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    ResultGroup,
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_FREEZE = REPO_ROOT / "src/datadiff_osc/public_api_freeze.json"
PUBLIC_FREEZE_SHA256 = (
    "cb505dbe665e517e53314ddae8ba9302d05ac794d326e9be1e792f8bd4e22131"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bound(
    root: Path, relative: str, payload: str | bytes
) -> corpora.SourceBinding:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    path.write_bytes(raw)
    return corpora.SourceBinding(relative, _sha256(raw))


def _typed_case(root: Path, case_id: str = "case-001") -> corpora.TypedCaseSource:
    case = Case(
        case_id=case_id,
        seed=17,
        tables=[],
        program=Program(program_id=f"program-{case_id}", seed=17, operations=[]),
    )
    source = _write_bound(
        root, f"cases/{case_id}.json", canonical_json(case.to_dict())
    )
    return corpora.derive_typed_case_source(root, source)


def _contract_fingerprint(label: str = "contract") -> ContractFingerprint:
    return ContractFingerprint(
        contract_digest=f"{label}-digest",
        registry_digest="registry-digest",
    )


def _execution_record(
    *,
    label: str,
    decision_index: int,
    payload_digest: str,
    endpoint_ids: tuple[str, ...],
    contract_fingerprint: ContractFingerprint,
    verdict: corpora.ExpectedVerdict,
    protocol_digest: str = "protocol-digest",
    seed_lineage_digest: str = "seed-lineage-digest",
) -> corpora.ExactExecutionRecord:
    identity = TaskIdentity(
        protocol_digest=protocol_digest,
        task_kind=TaskKind.FINGERPRINT_CLUSTER,
        epoch_index=0,
        decision_index=decision_index,
        seed_lineage_digest=seed_lineage_digest,
        attempt=0,
    )
    task = TaskSpec(
        identity=identity,
        dependency_task_ids=(),
        resources=ResourceTokens(
            cpu_tokens=1,
            rss_bytes=0,
            io_class="none",
            backend_internal_threads=0,
        ),
        payload_digest=payload_digest,
    )
    result_group = ResultGroup(
        result_group_id=f"result-{label}",
        task_ids=(identity.task_id,),
        endpoint_ids=endpoint_ids,
        contract_fingerprint=contract_fingerprint,
        target_fingerprint=None,
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=(label,),
    )
    evidence = EvidenceEnvelope(
        evidence_id=f"evidence-{label}",
        state=EvidenceState.FINDING,
        result_group_digest=result_group.digest,
        task_id=identity.task_id,
        seed_lineage_digest=seed_lineage_digest,
        contract_fingerprint=contract_fingerprint,
        target_fingerprint=None,
        derivation_certificate_digest=f"derivation-{label}",
        applicability_certificate_digest=f"applicability-{label}",
        activation_certificate_digest=f"activation-{label}",
        observation_certificate_digest=f"observation-{label}",
        execution_outcomes=tuple(
            StructuredExecutionOutcome(
                endpoint_id=endpoint_id,
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
            )
            for endpoint_id in endpoint_ids
        ),
        verdict_kind=VerdictKind(verdict.value),
    )
    return corpora.ExactExecutionRecord(task, result_group, evidence)


def _root_manifest() -> corpora.CorpusManifestCandidate:
    inventory = corpora.derive_confirmed_root_inventory(REPO_ROOT)
    entries = tuple(
        corpora.CorpusCandidateEntry(
            corpus_kind=corpora.CorpusKind.CONFIRMED_ROOTS,
            entry_id=item.root_id,
            source=item.case_source,
            evidence=corpora.ConfirmedRootExpectation(
                root_id=item.root_id,
                expected_verdict=corpora.ExpectedVerdict.VIOLATED,
                expected_signature_digest=item.expected_signature_digest,
            ),
        )
        for item in inventory.confirmed
    )
    return corpora.CorpusManifestCandidate(
        manifest_id="phase6-confirmed-root-candidate-v2",
        corpus_kind=corpora.CorpusKind.CONFIRMED_ROOTS,
        submitter_id="contract-owner",
        entries=entries,
    )


def _false_positive_manifest(root: Path) -> corpora.CorpusManifestCandidate:
    case = _typed_case(root, "false-positive-case")
    facts = {
        "finding_id": "false-positive-001",
        "component_id": "ordered-value-component",
        "case_digest": case.case_digest,
        "observed_verdict": "VIOLATED",
        "adjudicated_verdict": "SATISFIED",
        "component_rule": "ordered equality applies only to the selected component",
        "component_diff": "the observed difference is outside that component",
        "globally_ignored_axes": [],
        "globally_ignored_families": [],
    }
    adjudication_source = _write_bound(
        root, "adjudication/facts.json", canonical_json(facts)
    )
    evidence = corpora.FalsePositiveAdjudication(
        finding_id="false-positive-001",
        component_id="ordered-value-component",
        case=case,
        adjudication_source=adjudication_source,
        observed_verdict=corpora.ExpectedVerdict.VIOLATED,
        adjudicated_verdict=corpora.ExpectedVerdict.SATISFIED,
        component_rule=facts["component_rule"],
        component_diff=facts["component_diff"],
    )
    entry_source = _write_bound(
        root, "evidence/false-positive.json", canonical_json(evidence)
    )
    entry = corpora.CorpusCandidateEntry(
        corpus_kind=corpora.CorpusKind.FALSE_POSITIVES,
        entry_id=evidence.finding_id,
        source=entry_source,
        evidence=evidence,
    )
    return corpora.CorpusManifestCandidate(
        manifest_id="false-positive-candidate-v2",
        corpus_kind=corpora.CorpusKind.FALSE_POSITIVES,
        submitter_id="submitter",
        entries=(entry,),
    )


def _compatibility_manifest(root: Path) -> corpora.CorpusManifestCandidate:
    case = _typed_case(root, "compatibility-case")
    endpoints = ("endpoint-a", "endpoint-b")
    fingerprint = _contract_fingerprint("compatibility")
    v1 = _execution_record(
        label="compat-v1",
        decision_index=1,
        payload_digest=corpora.compatibility_task_payload_digest(
            case_digest=case.case_digest,
            endpoint_ids=endpoints,
            contract_fingerprint=fingerprint,
            implementation_version="v1",
        ),
        endpoint_ids=endpoints,
        contract_fingerprint=fingerprint,
        verdict=corpora.ExpectedVerdict.SATISFIED,
    )
    v2 = _execution_record(
        label="compat-v2",
        decision_index=2,
        payload_digest=corpora.compatibility_task_payload_digest(
            case_digest=case.case_digest,
            endpoint_ids=endpoints,
            contract_fingerprint=fingerprint,
            implementation_version="v2",
        ),
        endpoint_ids=endpoints,
        contract_fingerprint=fingerprint,
        verdict=corpora.ExpectedVerdict.SATISFIED,
    )
    evidence = corpora.CompatibilityVerdictPair(
        pair_id="compatibility-pair-001",
        case=case,
        endpoint_ids=endpoints,
        contract_fingerprint=fingerprint,
        v1=v1,
        v2=v2,
    )
    source = _write_bound(root, "evidence/compatibility.json", canonical_json(evidence))
    entry = corpora.CorpusCandidateEntry(
        corpus_kind=corpora.CorpusKind.V1_V2_COMPATIBILITY,
        entry_id=evidence.pair_id,
        source=source,
        evidence=evidence,
    )
    return corpora.CorpusManifestCandidate(
        manifest_id="compatibility-candidate-v2",
        corpus_kind=corpora.CorpusKind.V1_V2_COMPATIBILITY,
        submitter_id="submitter",
        entries=(entry,),
    )


def _copy_mutation_source(root: Path) -> corpora.SourceBinding:
    return _write_bound(
        root,
        corpora.MUTATION_SOURCE_PATH,
        (REPO_ROOT / corpora.MUTATION_SOURCE_PATH).read_bytes(),
    )


def _mutant_manifest(root: Path) -> corpora.CorpusManifestCandidate:
    _copy_mutation_source(root)
    inventory = corpora.derive_mutant_inventory(root)
    case = _typed_case(root, "mutant-case")
    endpoints = ("endpoint-a", "endpoint-b")
    fingerprint = _contract_fingerprint("mutant")
    entries: list[corpora.CorpusCandidateEntry] = []
    for index, mutant_id in enumerate(inventory.all_ids):
        baseline = _execution_record(
            label=f"baseline-{index}",
            decision_index=index * 2,
            payload_digest=corpora.mutant_task_payload_digest(
                case_digest=case.case_digest,
                endpoint_ids=endpoints,
                contract_fingerprint=fingerprint,
                role="baseline",
            ),
            endpoint_ids=endpoints,
            contract_fingerprint=fingerprint,
            verdict=corpora.ExpectedVerdict.SATISFIED,
        )
        mutant = _execution_record(
            label=f"mutant-{index}",
            decision_index=index * 2 + 1,
            payload_digest=corpora.mutant_task_payload_digest(
                case_digest=case.case_digest,
                endpoint_ids=endpoints,
                contract_fingerprint=fingerprint,
                role="mutant",
                mutant_id=mutant_id,
                mutant_version="v1",
                mutant_source_digest=inventory.source.digest,
            ),
            endpoint_ids=endpoints,
            contract_fingerprint=fingerprint,
            verdict=corpora.ExpectedVerdict.VIOLATED,
        )
        evidence = corpora.MutantKillWitness(
            mutant_id=mutant_id,
            mutant_version="v1",
            mutant_source=inventory.source,
            case=case,
            endpoint_ids=endpoints,
            contract_fingerprint=fingerprint,
            kill_semantics=corpora.expected_kill_semantics(mutant_id),
            baseline=baseline,
            mutant=mutant,
        )
        source = _write_bound(
            root,
            f"evidence/mutant-{index:02d}.json",
            canonical_json(evidence),
        )
        entries.append(
            corpora.CorpusCandidateEntry(
                corpus_kind=corpora.CorpusKind.MUTANT_KILL_WITNESSES,
                entry_id=mutant_id,
                source=source,
                evidence=evidence,
            )
        )
    return corpora.CorpusManifestCandidate(
        manifest_id="mutant-witness-candidate-v2",
        corpus_kind=corpora.CorpusKind.MUTANT_KILL_WITNESSES,
        submitter_id="contract-owner",
        entries=tuple(sorted(entries, key=lambda item: item.entry_id)),
    )


def test_exact_nine_confirmed_roots_and_one_pending_root_is_excluded():
    inventory = corpora.derive_confirmed_root_inventory(REPO_ROOT)

    assert len(inventory.confirmed_ids) == 9
    assert inventory.confirmed_ids == (
        "datafusion-distinct-null-topk-001",
        "datafusion-grouped-null-topk-001",
        "datafusion-limit-offset-pushdown-001",
        "datafusion-negative-zero-comparison-001",
        "datafusion-ordered-limit-idempotence-001",
        "duckdb-join-filter-pushdown-limit-001",
        "polars-grouped-max-sort-metadata-001",
        "polars-reflected-arithmetic-operand-order-001",
        "pyarrow-sliced-bool-hash-aggregate-001",
    )
    assert inventory.excluded_pending_ids == (
        "datafusion-order-by-offset-subquery-groupby-001",
    )


def test_confirmed_root_candidate_binds_violation_and_expected_signature():
    result = corpora.validate_candidate_manifest(_root_manifest(), REPO_ROOT)

    assert result.state is corpora.CorpusValidationState.CANDIDATE_MATERIAL
    assert result.candidate_material_valid
    assert not result.independently_approved
    assert not result.root_authority_eligible
    assert not result.root_denominator_eligible(9)


def test_all_nine_confirmed_verdict_and_signature_substitutions_reject():
    manifest = _root_manifest()

    for entry in manifest.entries:
        with pytest.raises(ValueError, match="frozen as VIOLATED"):
            replace(
                entry.evidence,
                expected_verdict=corpora.ExpectedVerdict.SATISFIED,
            )

    for entry in manifest.entries:
        forged_evidence = replace(
            entry.evidence,
            expected_signature_digest="f" * 64,
        )
        forged_entry = replace(entry, evidence=forged_evidence)
        forged_manifest = replace(
            manifest,
            entries=tuple(
                forged_entry if item.entry_id == entry.entry_id else item
                for item in manifest.entries
            ),
        )
        result = corpora.validate_candidate_manifest(forged_manifest, REPO_ROOT)
        assert result.state is corpora.CorpusValidationState.REJECTED
        assert (
            f"confirmed_root_expectation_mismatch:{entry.entry_id}" in result.errors
        )


def test_pending_unknown_missing_and_changed_root_sources_fail_closed(monkeypatch):
    manifest = _root_manifest()
    payload = json.loads(
        (REPO_ROOT / corpora.CONFIRMED_ROOT_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    pending = payload["pending_roots"][0]
    pending_entry = replace(
        manifest.entries[0],
        entry_id=pending["root_id"],
        source=corpora.SourceBinding(
            pending["dsl_case"]["path"], pending["dsl_case"]["sha256"]
        ),
        evidence=corpora.ConfirmedRootExpectation(
            root_id=pending["root_id"],
            expected_verdict=corpora.ExpectedVerdict.VIOLATED,
            expected_signature_digest="0" * 64,
        ),
    )
    forged = replace(
        manifest,
        entries=tuple(
            sorted((pending_entry, *manifest.entries[1:]), key=lambda item: item.entry_id)
        ),
    )
    result = corpora.validate_candidate_manifest(forged, REPO_ROOT)
    assert any(item.startswith("pending_root_forbidden:") for item in result.errors)
    assert any(item.startswith("confirmed_roots_missing:") for item in result.errors)

    monkeypatch.setattr(corpora, "CONFIRMED_ROOT_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(corpora.CorpusSourceError, match="SHA-256 mismatch"):
        corpora.derive_confirmed_root_inventory(REPO_ROOT)


def test_false_positive_requires_exact_component_adjudication(tmp_path):
    manifest = _false_positive_manifest(tmp_path)
    result = corpora.validate_candidate_manifest(manifest, tmp_path)
    assert result.state is corpora.CorpusValidationState.CANDIDATE_MATERIAL

    evidence = manifest.entries[0].evidence
    with pytest.raises(ValueError, match="globally ignore axes"):
        replace(evidence, globally_ignored_axes=("order",))
    with pytest.raises(ValueError, match="globally ignore families"):
        replace(evidence, globally_ignored_families=("all",))

    substituted = replace(evidence, component_diff="caller summary")
    source = _write_bound(
        tmp_path, "evidence/substituted-fp.json", canonical_json(substituted)
    )
    forged = replace(
        manifest,
        entries=(replace(manifest.entries[0], source=source, evidence=substituted),),
    )
    forged_result = corpora.validate_candidate_manifest(forged, tmp_path)
    assert any(
        "adjudication facts do not match" in item for item in forged_result.errors
    )


def test_generic_or_source_only_false_positive_shell_rejects(tmp_path):
    source = _write_bound(tmp_path, "material/source.json", "{}")
    payload = {
        "corpus_kind": "false_positives",
        "entry_id": "source-only",
        "source": to_primitive(source),
        "evidence": {
            "expected_verdict": "SATISFIED",
            "schema_version": "generic-shell-v1",
        },
        "schema_version": corpora.CORPUS_ENTRY_SCHEMA_VERSION,
    }
    with pytest.raises(corpora.CorpusDecodeError):
        corpora.corpus_entry_from_mapping(payload)


def test_exact_compatibility_pair_round_trips_and_validates(tmp_path):
    manifest = _compatibility_manifest(tmp_path)
    encoded = canonical_json(manifest)
    assert corpora.decode_candidate_manifest(encoded) == manifest
    result = corpora.validate_candidate_manifest(manifest, tmp_path)
    assert result.state is corpora.CorpusValidationState.CANDIDATE_MATERIAL
    pair = manifest.entries[0].evidence
    assert pair.v1.verdict is corpora.ExpectedVerdict.SATISFIED
    assert pair.v2.verdict is corpora.ExpectedVerdict.SATISFIED


def test_compatibility_cross_context_and_incomplete_pairs_reject(tmp_path):
    manifest = _compatibility_manifest(tmp_path)
    pair = manifest.entries[0].evidence

    wrong_seed_identity = replace(
        pair.v2.task_spec.identity, seed_lineage_digest="other-seed-lineage"
    )
    wrong_seed_task = replace(pair.v2.task_spec, identity=wrong_seed_identity)
    with pytest.raises(ValueError, match="result group is not exactly task-bound"):
        replace(pair.v2, task_spec=wrong_seed_task)

    cross_context_v2 = _execution_record(
        label="compat-cross-context-v2",
        decision_index=3,
        payload_digest=corpora.compatibility_task_payload_digest(
            case_digest=pair.case.case_digest,
            endpoint_ids=pair.endpoint_ids,
            contract_fingerprint=pair.contract_fingerprint,
            implementation_version="v2",
        ),
        endpoint_ids=pair.endpoint_ids,
        contract_fingerprint=pair.contract_fingerprint,
        verdict=corpora.ExpectedVerdict.SATISFIED,
        seed_lineage_digest="other-seed-lineage",
    )
    with pytest.raises(ValueError, match="seed lineage mismatch"):
        replace(pair, v2=cross_context_v2)

    with pytest.raises(ValueError, match="task payload mismatch"):
        replace(
            pair,
            case=replace(pair.case, case_digest="other-case-digest"),
        )

    payload = to_primitive(manifest)
    del payload["entries"][0]["evidence"]["v2"]
    with pytest.raises(corpora.CorpusDecodeError):
        corpora.decode_candidate_manifest(canonical_json(payload))


def test_source_only_compatibility_shell_rejects(tmp_path):
    manifest = _compatibility_manifest(tmp_path)
    payload = to_primitive(manifest)
    payload["entries"][0]["evidence"] = {
        "expected_verdict": "SATISFIED",
        "schema_version": "source-only-v1",
    }
    with pytest.raises(corpora.CorpusDecodeError):
        corpora.decode_candidate_manifest(canonical_json(payload))


def test_exact_thirty_two_typed_mutant_witnesses_round_trip(tmp_path):
    manifest = _mutant_manifest(tmp_path)
    inventory = corpora.derive_mutant_inventory(tmp_path)
    assert (len(inventory.comparator_ids), len(inventory.axis_ids), len(inventory.hyperedge_ids)) == (12, 13, 7)
    assert len(manifest.entries) == 32
    assert corpora.decode_candidate_manifest(canonical_json(manifest)) == manifest
    result = corpora.validate_candidate_manifest(manifest, tmp_path)
    assert result.state is corpora.CorpusValidationState.CANDIDATE_MATERIAL
    assert len(result.entry_ids) == 32
    assert not result.root_authority_eligible


def test_all_source_only_mutants_and_cross_mutant_swaps_reject(tmp_path):
    _copy_mutation_source(tmp_path)
    inventory = corpora.derive_mutant_inventory(tmp_path)
    for mutant_id in inventory.all_ids:
        payload = {
            "mutant_id": mutant_id,
            "kill_semantics": corpora.expected_kill_semantics(mutant_id).value,
            "schema_version": "source-only-mutant-v1",
        }
        with pytest.raises(corpora.CorpusDecodeError):
            corpora._entry_evidence_from_mapping(
                corpora.CorpusKind.MUTANT_KILL_WITNESSES, payload
            )

    manifest = _mutant_manifest(tmp_path)
    first = manifest.entries[0].evidence
    second = manifest.entries[1].evidence
    with pytest.raises(ValueError, match="task payload mismatch"):
        replace(first, mutant=second.mutant)

    with pytest.raises(ValueError, match="unique"):
        replace(manifest, entries=(manifest.entries[0], manifest.entries[0]))

    duplicate_mutant_task = replace(
        first.mutant.task_spec,
        payload_digest=corpora.mutant_task_payload_digest(
            case_digest=second.case.case_digest,
            endpoint_ids=second.endpoint_ids,
            contract_fingerprint=second.contract_fingerprint,
            role="mutant",
            mutant_id=second.mutant_id,
            mutant_version=second.mutant_version,
            mutant_source_digest=second.mutant_source.digest,
        ),
    )
    duplicate_mutant_result = replace(
        first.mutant.result_group,
        result_group_id="result-duplicate-mutant-pair",
    )
    duplicate_mutant_evidence = replace(
        first.mutant.evidence,
        evidence_id="evidence-duplicate-mutant-pair",
        result_group_digest=duplicate_mutant_result.digest,
    )
    duplicate_mutant_record = corpora.ExactExecutionRecord(
        task_spec=duplicate_mutant_task,
        result_group=duplicate_mutant_result,
        evidence=duplicate_mutant_evidence,
    )
    duplicate_witness = replace(
        second,
        baseline=first.baseline,
        mutant=duplicate_mutant_record,
    )
    duplicate_entry = replace(
        manifest.entries[1],
        source=_write_bound(
            tmp_path,
            "evidence/duplicate-mutant-pair.json",
            canonical_json(duplicate_witness),
        ),
        evidence=duplicate_witness,
    )
    duplicate_pair_manifest = replace(
        manifest,
        entries=(manifest.entries[0], duplicate_entry, *manifest.entries[2:]),
    )
    duplicate_pair_result = corpora.validate_candidate_manifest(
        duplicate_pair_manifest, tmp_path
    )
    assert "mutant_execution_pairs_duplicated" in duplicate_pair_result.errors


def test_missing_unknown_and_wrong_category_mutants_fail_closed(tmp_path):
    manifest = _mutant_manifest(tmp_path)
    missing = replace(manifest, entries=manifest.entries[1:])
    result = corpora.validate_candidate_manifest(missing, tmp_path)
    assert any(item.startswith("mutant_witnesses_missing:") for item in result.errors)

    axis_entry = next(
        item for item in manifest.entries if item.entry_id.startswith("axis:")
    )
    with pytest.raises(ValueError, match="kill predicate"):
        replace(
            axis_entry.evidence,
            kill_semantics=corpora.KillSemantics.COMPARATOR_WEAKENING_EXPOSED,
        )


def test_structural_review_never_becomes_independent_approval(tmp_path):
    manifest = _false_positive_manifest(tmp_path)
    manifest_source = _write_bound(
        tmp_path, "review/candidate.json", canonical_json(manifest)
    )
    approval = corpora.IndependentApprovalRecord(
        approval_id="arbitrary-review",
        manifest_digest=manifest.digest,
        manifest_source=manifest_source,
        submitter_id=manifest.submitter_id,
        approver_id="arbitrary-distinct-string",
        reviewed_entry_ids=manifest.entry_ids,
    )
    approval_source = _write_bound(
        tmp_path, "review/approval.json", canonical_json(approval)
    )

    result = corpora.validate_independent_approval(
        manifest, approval, approval_source, tmp_path
    )
    assert result.state is corpora.CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING
    assert result.structurally_reviewed
    assert not result.independently_approved
    assert not result.meets_independently_approved_minimum(1)
    assert not result.root_denominator_eligible(1)
    assert not result.root_authority_eligible
    assert result.root_authority_state == "not_evaluated_root_context_required"


def test_forged_review_hash_payload_self_review_and_duplicate_ids_reject(tmp_path):
    manifest = _false_positive_manifest(tmp_path)
    manifest_source = _write_bound(tmp_path, "candidate.json", canonical_json(manifest))
    approval = corpora.IndependentApprovalRecord(
        approval_id="review",
        manifest_digest="forged-manifest-digest",
        manifest_source=manifest_source,
        submitter_id=manifest.submitter_id,
        approver_id="reviewer",
        reviewed_entry_ids=manifest.entry_ids,
    )
    approval_source = _write_bound(tmp_path, "approval.json", canonical_json(approval))
    result = corpora.validate_independent_approval(
        manifest, approval, replace(approval_source, sha256="0" * 64), tmp_path
    )
    assert result.state is corpora.CorpusValidationState.REJECTED
    assert "approval_manifest_digest_mismatch" in result.errors
    assert not result.independently_approved

    with pytest.raises(ValueError, match="approve itself"):
        replace(
            approval,
            manifest_digest=manifest.digest,
            approver_id=manifest.submitter_id,
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(
            approval,
            manifest_digest=manifest.digest,
            reviewed_entry_ids=(manifest.entry_ids[0], manifest.entry_ids[0]),
        )


def test_absent_required_corpora_and_false_positive_minimum_fail_closed():
    missing = corpora.absent_required_phase6_corpora()
    assert tuple(item.corpus_kind for item in missing) == (
        corpora.CorpusKind.FALSE_POSITIVES,
        corpora.CorpusKind.V1_V2_COMPATIBILITY,
        corpora.CorpusKind.MUTANT_KILL_WITNESSES,
    )
    assert all(item.state is corpora.CorpusValidationState.MISSING_INPUT for item in missing)
    assert all(not item.meets_independently_approved_minimum(1) for item in missing)
    assert all(not item.root_authority_eligible for item in missing)


def test_changed_source_sha_duplicate_json_and_authority_fields_reject(tmp_path):
    manifest = _false_positive_manifest(tmp_path)
    changed = replace(
        manifest,
        entries=(
            replace(
                manifest.entries[0],
                source=replace(manifest.entries[0].source, sha256="0" * 64),
            ),
        ),
    )
    result = corpora.validate_candidate_manifest(changed, tmp_path)
    assert any(item.startswith("entry_source_invalid:") for item in result.errors)

    duplicate_json = '{"manifest_id":"a","manifest_id":"b"}'
    with pytest.raises(corpora.CorpusDecodeError, match="duplicate JSON field"):
        corpora.decode_candidate_manifest(duplicate_json)

    payload = to_primitive(manifest)
    payload["authority_eligible"] = True
    payload["gate_passed"] = True
    with pytest.raises(corpora.CorpusDecodeError, match="fields do not match"):
        corpora.decode_candidate_manifest(canonical_json(payload))


def test_candidate_review_and_specialized_documents_round_trip(tmp_path):
    manifest = _false_positive_manifest(tmp_path)
    assert corpora.decode_candidate_manifest(canonical_json(manifest)) == manifest
    manifest_source = _write_bound(tmp_path, "candidate.json", canonical_json(manifest))
    approval = corpora.IndependentApprovalRecord(
        approval_id="review",
        manifest_digest=manifest.digest,
        manifest_source=manifest_source,
        submitter_id=manifest.submitter_id,
        approver_id="reviewer",
        reviewed_entry_ids=manifest.entry_ids,
    )
    assert corpora.decode_approval_record(canonical_json(approval)) == approval


def test_c0_traversal_alias_and_resolver_failures_reject(tmp_path, monkeypatch):
    for codepoint in range(0x20):
        with pytest.raises(ValueError, match="C0 control"):
            corpora.SourceBinding(f"material/a{chr(codepoint)}b.json", "0" * 64)

    for path in (
        "../escape.json",
        "/absolute.json",
        "material//alias.json",
        "material/./alias.json",
        "material\\alias.json",
    ):
        with pytest.raises(ValueError):
            corpora.SourceBinding(path, "0" * 64)

    outside = tmp_path.parent / "outside-phase6-corpus"
    outside.mkdir(exist_ok=True)
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    binding = corpora.SourceBinding("alias/value.json", "0" * 64)
    with pytest.raises(corpora.CorpusSourceError, match="escapes repository"):
        corpora._read_source(tmp_path, binding)

    def fail_resolve(self, *args, **kwargs):
        raise OSError("resolver unavailable")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    with pytest.raises(corpora.CorpusSourceError, match="cannot be resolved"):
        corpora._resolved_bound_path(tmp_path, "material/value.json")


def test_public_api_freeze_remains_byte_identical():
    assert _sha256(PUBLIC_FREEZE.read_bytes()) == PUBLIC_FREEZE_SHA256
