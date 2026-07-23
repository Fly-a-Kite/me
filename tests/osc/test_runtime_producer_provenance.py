from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path

import pytest

from datadiff_osc._canonical import canonical_envelope
from datadiff_osc.runtime._private_receipts import (
    REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
    TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
    RepositoryTestReceipt,
)
from datadiff_osc.runtime._producer_provenance import (
    PRODUCER_PROVENANCE_SCHEMA_VERSION,
    REPOSITORY_TEST_JUNIT_SCHEMA_VERSION,
    REPOSITORY_TEST_LOG_SCHEMA_VERSION,
    REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION,
    EnvironmentRecord,
    ExpectedProvenanceRecordPolicy,
    ImmutableFileRecord,
    ImmutableTypedOutputRecord,
    ProducerFailureRecord,
    ProducerOriginKind,
    ProducerProvenancePolicy,
    ProducerRecordCategory,
    RawParentRoleBinding,
    RepositoryTestRawContext,
    SourceSnapshotFileBinding,
    SourceSnapshotView,
    build_producer_provenance,
    verify_producer_provenance,
)
from datadiff_osc.runtime._receipt_producers import (
    RepositoryTestObservation,
    TargetPackageObservation,
    TargetVersionObservation,
    build_repository_test_receipt,
    build_target_version_receipt,
)


PUBLIC_FREEZE_SHA256 = (
    "cb505dbe665e517e53314ddae8ba9302d05ac794d326e9be1e792f8bd4e22131"
)


def _file(
    record_id: str,
    subject_kind: str,
    path: str,
    content: bytes,
    *,
    schema_version: str = "test-record-v1",
    origin: ProducerOriginKind = ProducerOriginKind.REAL_EXECUTION,
) -> ImmutableFileRecord:
    return ImmutableFileRecord(
        record_id=record_id,
        subject_kind=subject_kind,
        relative_path=path,
        schema_version=schema_version,
        origin=origin,
        content=content,
    )


def _repository_observation() -> RepositoryTestObservation:
    return RepositoryTestObservation(
        source_digest="source-digest-01",
        test_plan=("collect", "execute"),
        config=(("plugins", "none"), ("pytest", "frozen")),
        environment=(("PYTHONHASHSEED", "0"), ("TZ", "UTC")),
        command=("python", "-m", "pytest", "-q"),
        junit_xml=(
            b'<testsuite tests="1" failures="0" errors="0" skipped="0">'
            b'<testcase file="tests/a.py" name="test_a" />'
            b"</testsuite>"
        ),
        log=b"1 passed in 0.01s\n",
        exit_code=0,
    )


def _repository_receipt() -> RepositoryTestReceipt:
    return build_repository_test_receipt(_repository_observation())


def _raw_context(observation: RepositoryTestObservation) -> bytes:
    context = RepositoryTestRawContext(
        source_digest=observation.source_digest,
        test_plan=observation.test_plan,
        config=observation.config,
        environment=observation.environment,
        command=observation.command,
        exit_code=observation.exit_code,
    )
    return canonical_envelope(
        "RepositoryTestRawContext",
        REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION,
        context,
    ).encode("utf-8")


def _typed(
    *,
    raw_parents: tuple[RawParentRoleBinding, ...] = (
        RawParentRoleBinding("invocation_context", "raw-01"),
        RawParentRoleBinding("junit_xml", "raw-02"),
        RawParentRoleBinding("pytest_log", "raw-03"),
    ),
    origin: ProducerOriginKind = ProducerOriginKind.REAL_EXECUTION,
) -> ImmutableTypedOutputRecord:
    value = _repository_receipt()
    return ImmutableTypedOutputRecord(
        record_id="typed-01",
        subject_kind="repository_test_receipt",
        subject_ids=value.node_ids,
        relative_path="out/receipt.json",
        envelope_type="RepositoryTestReceipt",
        envelope_schema_version=REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
        raw_parent_roles=raw_parents,
        origin=origin,
        content=canonical_envelope(
            "RepositoryTestReceipt",
            REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
            value,
        ).encode("utf-8"),
    )


def _expected_file(
    category: ProducerRecordCategory,
    record: ImmutableFileRecord,
    *,
    frozen_sha: str = "",
) -> ExpectedProvenanceRecordPolicy:
    return ExpectedProvenanceRecordPolicy(
        category=category,
        record_id=record.record_id,
        subject_kind=record.subject_kind,
        subject_ids=(),
        relative_path=record.relative_path,
        schema_version=record.schema_version,
        envelope_type="",
        raw_parent_roles=(),
        origin=record.origin,
        frozen_source_sha256=frozen_sha,
    )


def _expected_typed(
    record: ImmutableTypedOutputRecord,
    *,
    frozen_sha: str = "",
) -> ExpectedProvenanceRecordPolicy:
    return ExpectedProvenanceRecordPolicy(
        category=ProducerRecordCategory.TYPED_OUTPUT,
        record_id=record.record_id,
        subject_kind=record.subject_kind,
        subject_ids=record.subject_ids,
        relative_path=record.relative_path,
        schema_version=record.envelope_schema_version,
        envelope_type=record.envelope_type,
        raw_parent_roles=record.raw_parent_roles,
        origin=record.origin,
        frozen_source_sha256=frozen_sha,
    )


def _fixture(
    *,
    input_origin: ProducerOriginKind = ProducerOriginKind.REAL_EXECUTION,
):
    observation = _repository_observation()
    environment = (
        EnvironmentRecord("PYTHONHASHSEED", "0"),
        EnvironmentRecord("TZ", "UTC"),
    )
    implementation = (
        _file(
            "implementation-01",
            "implementation",
            "src/datadiff_osc/runtime/_receipt_producers.py",
            b"implementation bytes",
        ),
    )
    inputs = (
        _file(
            "input-01",
            "frozen_plan",
            "inputs/plan.json",
            b"plan bytes",
            origin=input_origin,
        ),
    )
    raw_outputs = (
        _file(
            "raw-01",
            "repository_test_context",
            "out/context.json",
            _raw_context(observation),
            schema_version=REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION,
        ),
        _file(
            "raw-02",
            "repository_junit",
            "out/junit.xml",
            observation.junit_xml,
            schema_version=REPOSITORY_TEST_JUNIT_SCHEMA_VERSION,
        ),
        _file(
            "raw-03",
            "repository_log",
            "out/pytest.log",
            observation.log,
            schema_version=REPOSITORY_TEST_LOG_SCHEMA_VERSION,
        ),
    )
    typed_outputs = (_typed(),)
    receipt = build_producer_provenance(
        producer_kind="repository_test_replay",
        invocation_id="invocation-01",
        source_digest="source-digest-01",
        protocol_digest="protocol-digest-01",
        plan_digest="plan-digest-01",
        command=("python", "producer.py", "--frozen"),
        cwd=".",
        environment=environment,
        implementation_files=implementation,
        inputs=inputs,
        raw_outputs=raw_outputs,
        typed_outputs=typed_outputs,
    )
    snapshot = SourceSnapshotView(
        source_digest=receipt.source_digest,
        files=(
            SourceSnapshotFileBinding(
                relative_path=implementation[0].relative_path,
                sha256=implementation[0].sha256,
            ),
        ),
    )
    expected = (
        _expected_file(ProducerRecordCategory.IMPLEMENTATION, implementation[0]),
        _expected_file(
            ProducerRecordCategory.INPUT,
            inputs[0],
            frozen_sha=(inputs[0].sha256 if input_origin is not ProducerOriginKind.REAL_EXECUTION else ""),
        ),
        _expected_file(ProducerRecordCategory.RAW_OUTPUT, raw_outputs[0]),
        _expected_file(ProducerRecordCategory.RAW_OUTPUT, raw_outputs[1]),
        _expected_file(ProducerRecordCategory.RAW_OUTPUT, raw_outputs[2]),
        _expected_typed(typed_outputs[0]),
    )
    policy = ProducerProvenancePolicy(
        producer_kind=receipt.producer_kind,
        invocation_id=receipt.invocation_id,
        source_digest=receipt.source_digest,
        protocol_digest=receipt.protocol_digest,
        plan_digest=receipt.plan_digest,
        command=receipt.command,
        cwd=receipt.cwd,
        environment=receipt.environment,
        expected_records=expected,
    )
    raw = {
        "environment": environment,
        "implementation_files": implementation,
        "inputs": inputs,
        "raw_outputs": raw_outputs,
        "typed_outputs": typed_outputs,
    }
    return receipt, policy, snapshot, raw


def _verify(receipt, policy, snapshot, raw, *, failures=()):
    return verify_producer_provenance(
        receipt,
        policy=policy,
        source_snapshot=snapshot,
        failures=failures,
        **raw,
    )


def _rebuild(raw, receipt, *, failures=()):
    return build_producer_provenance(
        producer_kind=receipt.producer_kind,
        invocation_id=receipt.invocation_id,
        source_digest=receipt.source_digest,
        protocol_digest=receipt.protocol_digest,
        plan_digest=receipt.plan_digest,
        command=receipt.command,
        cwd=receipt.cwd,
        failures=failures,
        exit_code=receipt.exit_code,
        **raw,
    )


def test_exact_typed_producer_provenance_round_trips():
    receipt, policy, snapshot, raw = _fixture()

    verification = _verify(receipt, policy, snapshot, raw)

    assert verification.valid
    assert verification.errors == ()
    assert receipt.schema_version == PRODUCER_PROVENANCE_SCHEMA_VERSION
    assert receipt.typed_outputs[0].raw_parent_record_ids == (
        "raw-01",
        "raw-02",
        "raw-03",
    )


@pytest.mark.parametrize("field_name", ["source_digest", "protocol_digest", "plan_digest"])
def test_forged_source_protocol_or_plan_fails_closed(field_name: str):
    receipt, policy, snapshot, raw = _fixture()
    forged = replace(receipt, **{field_name: f"forged-{field_name}"})

    verification = _verify(forged, policy, snapshot, raw)

    assert not verification.valid
    assert any(field_name in error or "raw_binding_mismatch" in error for error in verification.errors)


def test_category_relabel_cannot_escape_exact_record_policy():
    receipt, policy, snapshot, raw = _fixture()
    changed = dict(raw)
    changed["inputs"] = (replace(raw["raw_outputs"][0], subject_kind="frozen_plan"),)
    changed["raw_outputs"] = (
        replace(
            raw["inputs"][0],
            subject_kind="repository_test_context",
            schema_version=REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION,
        ),
        *raw["raw_outputs"][1:],
    )
    changed["typed_outputs"] = (
        replace(
            raw["typed_outputs"][0],
            raw_parent_roles=(
                RawParentRoleBinding("invocation_context", "input-01"),
                RawParentRoleBinding("junit_xml", "raw-02"),
                RawParentRoleBinding("pytest_log", "raw-03"),
            ),
        ),
    )
    forged = _rebuild(changed, receipt)

    verification = _verify(forged, policy, snapshot, changed)

    assert not verification.valid
    assert any("record_policy_mismatch" in item for item in verification.errors)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("subject_kind", "relabelled_subject"),
        ("relative_path", "elsewhere/receipt.json"),
        ("envelope_schema_version", "osc-root-repository-test-receipt-v999"),
        ("envelope_type", "UnknownReceipt"),
    ],
)
def test_typed_subject_path_schema_and_type_relabel_reject(field_name: str, value: str):
    receipt, policy, snapshot, raw = _fixture()
    changed = dict(raw)
    changed["typed_outputs"] = (replace(raw["typed_outputs"][0], **{field_name: value}),)
    forged = _rebuild(changed, receipt)

    verification = _verify(forged, policy, snapshot, changed)

    assert not verification.valid
    assert any(
        "record_policy_mismatch" in item or "typed_raw_rebuild_failed" in item
        for item in verification.errors
    )


def test_frozen_origin_requires_independent_exact_sha_and_cannot_be_relabeled():
    receipt, policy, snapshot, raw = _fixture(
        input_origin=ProducerOriginKind.ROOT_FROZEN_PUBLIC_SOURCE
    )
    assert _verify(receipt, policy, snapshot, raw).valid

    changed = dict(raw)
    changed["inputs"] = (
        replace(
            raw["inputs"][0],
            origin=ProducerOriginKind.ROOT_FROZEN_CONTROL_CORPUS,
        ),
    )
    forged = _rebuild(changed, receipt)
    verification = _verify(forged, policy, snapshot, changed)
    assert not verification.valid
    assert any("origin_not_allowed" in item for item in verification.errors)

    relabeled_real = dict(raw)
    relabeled_real["inputs"] = (
        replace(raw["inputs"][0], origin=ProducerOriginKind.REAL_EXECUTION),
    )
    real_receipt = _rebuild(relabeled_real, receipt)
    real_verification = _verify(real_receipt, policy, snapshot, relabeled_real)
    assert not real_verification.valid
    assert any("origin_not_allowed" in item for item in real_verification.errors)

    bad_expected = tuple(
        replace(item, frozen_source_sha256="f" * 64)
        if item.record_id == "input-01"
        else item
        for item in policy.expected_records
    )
    bad_policy = replace(policy, expected_records=bad_expected)
    verification = _verify(receipt, bad_policy, snapshot, raw)
    assert not verification.valid
    assert "provenance_frozen_source_hash_mismatch:input-01" in verification.errors


@pytest.mark.parametrize(
    "content",
    [
        b"not-json",
        b'{"canonical_schema_version":"osc-canonical-json-v1"}',
        b'{"canonical_schema_version":"osc-canonical-json-v1", "payload":{},"schema_version":"x","type":"RepositoryTestReceipt"}',
    ],
)
def test_malformed_or_noncanonical_typed_bytes_fail_closed(content: bytes):
    receipt, policy, snapshot, raw = _fixture()
    changed = dict(raw)
    changed["typed_outputs"] = (replace(raw["typed_outputs"][0], content=content),)
    forged = _rebuild(changed, receipt)

    verification = _verify(forged, policy, snapshot, changed)

    assert not verification.valid
    assert any("typed_raw_rebuild_failed" in item for item in verification.errors)


def test_wrong_typed_subject_and_raw_parent_swap_fail_closed():
    receipt, policy, snapshot, raw = _fixture()
    changed_subject = dict(raw)
    changed_subject["typed_outputs"] = (
        replace(raw["typed_outputs"][0], subject_ids=("forged-subject",)),
    )
    subject_receipt = _rebuild(changed_subject, receipt)
    subject_verification = _verify(subject_receipt, policy, snapshot, changed_subject)
    assert not subject_verification.valid
    assert any("subject identity mismatch" in item for item in subject_verification.errors)

    changed_parent = dict(raw)
    changed_parent["typed_outputs"] = (
        replace(
            raw["typed_outputs"][0],
            raw_parent_roles=(
                RawParentRoleBinding("invocation_context", "raw-01"),
                RawParentRoleBinding("junit_xml", "raw-03"),
                RawParentRoleBinding("pytest_log", "raw-02"),
            ),
        ),
    )
    parent_receipt = _rebuild(changed_parent, receipt)
    parent_verification = _verify(parent_receipt, policy, snapshot, changed_parent)
    assert not parent_verification.valid
    assert any("record_policy_mismatch" in item for item in parent_verification.errors)


def test_raw_role_order_extra_malformed_and_missing_rebuilder_fail_closed():
    receipt, policy, snapshot, raw = _fixture()

    reordered = dict(raw)
    reordered["typed_outputs"] = (
        replace(
            raw["typed_outputs"][0],
            raw_parent_roles=(
                RawParentRoleBinding("junit_xml", "raw-02"),
                RawParentRoleBinding("invocation_context", "raw-01"),
                RawParentRoleBinding("pytest_log", "raw-03"),
            ),
        ),
    )
    reordered_receipt = _rebuild(reordered, receipt)
    reordered_verification = _verify(
        reordered_receipt, policy, snapshot, reordered
    )
    assert not reordered_verification.valid
    assert any("role/order mismatch" in item for item in reordered_verification.errors)

    extra_parent = _file(
        "raw-04", "repository_extra", "out/extra.bin", b"extra"
    )
    extra = dict(raw)
    extra["raw_outputs"] = (*raw["raw_outputs"], extra_parent)
    extra["typed_outputs"] = (
        replace(
            raw["typed_outputs"][0],
            raw_parent_roles=(
                *raw["typed_outputs"][0].raw_parent_roles,
                RawParentRoleBinding("extra", "raw-04"),
            ),
        ),
    )
    extra_receipt = _rebuild(extra, receipt)
    extra_verification = _verify(extra_receipt, policy, snapshot, extra)
    assert not extra_verification.valid
    assert any("role/order mismatch" in item for item in extra_verification.errors)

    for index, malformed in ((1, b"<not-junit"), (2, b"not a pytest summary")):
        changed = dict(raw)
        changed["raw_outputs"] = tuple(
            replace(item, content=malformed) if offset == index else item
            for offset, item in enumerate(raw["raw_outputs"])
        )
        changed_receipt = _rebuild(changed, receipt)
        verification = _verify(changed_receipt, policy, snapshot, changed)
        assert not verification.valid
        assert any("typed_raw_rebuild_failed" in item for item in verification.errors)

    missing_kind = "repository_test_replay_without_registered_raw_rebuilder"
    self_consistent_receipt = replace(receipt, producer_kind=missing_kind)
    self_consistent_policy = replace(policy, producer_kind=missing_kind)
    missing_verification = _verify(
        self_consistent_receipt, self_consistent_policy, snapshot, raw
    )
    assert not missing_verification.valid
    assert any("raw rebuilder context is missing" in item for item in missing_verification.errors)

    target_value = build_target_version_receipt(
        TargetVersionObservation(
            source_digest=receipt.source_digest,
            audit_plan=(("policy", "exact"),),
            environment=(("venv", "phase6"),),
            python_runtime=(("version", "3.12"),),
            packages=(
                TargetPackageObservation(
                    distribution_name="pandas",
                    import_name="pandas",
                    installed_version="3.0.3",
                    latest_version="3.0.3",
                    installed_metadata=(
                        b"Metadata-Version: 2.4\nName: pandas\nVersion: 3.0.3\n\n"
                    ),
                    version_source_kind="pypi_json",
                    version_source=(
                        b'{"info":{"name":"pandas","version":"3.0.3"}}'
                    ),
                ),
            ),
        )
    )
    target_record = replace(
        raw["typed_outputs"][0],
        subject_kind="target_version_receipt",
        subject_ids=target_value.package_ids,
        envelope_type="TargetVersionReceipt",
        envelope_schema_version=TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
        content=canonical_envelope(
            "TargetVersionReceipt",
            TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
            target_value,
        ).encode("utf-8"),
    )
    target_raw = dict(raw)
    target_raw["typed_outputs"] = (target_record,)
    target_receipt = _rebuild(target_raw, receipt)
    target_policy = replace(
        policy,
        expected_records=(*policy.expected_records[:-1], _expected_typed(target_record)),
    )
    target_verification = _verify(
        target_receipt, target_policy, snapshot, target_raw
    )
    assert not target_verification.valid
    assert any("raw rebuilder context is missing" in item for item in target_verification.errors)


def test_forged_snapshot_environment_and_raw_bytes_fail_closed():
    receipt, policy, snapshot, raw = _fixture()
    assert not _verify(
        receipt,
        policy,
        replace(snapshot, source_digest="forged-source"),
        raw,
    ).valid

    changed_environment = dict(raw)
    changed_environment["environment"] = (
        EnvironmentRecord("PYTHONHASHSEED", "123"),
        EnvironmentRecord("TZ", "UTC"),
    )
    assert not _verify(receipt, policy, snapshot, changed_environment).valid

    changed_output = dict(raw)
    changed_output["raw_outputs"] = (
        replace(raw["raw_outputs"][0], content=b"substituted"),
        *raw["raw_outputs"][1:],
    )
    assert not _verify(receipt, policy, snapshot, changed_output).valid


def test_structured_failure_material_never_verifies():
    receipt, policy, snapshot, raw = _fixture()
    failure = ProducerFailureRecord(
        failure_id="failure-01",
        stage="decode",
        failure_kind="structured_error",
        detail="failed",
    )
    failed_receipt = _rebuild(raw, receipt, failures=(failure,))

    verification = _verify(
        failed_receipt, policy, snapshot, raw, failures=(failure,)
    )

    assert not verification.valid
    assert "provenance_structured_failure_present" in verification.errors


def test_diagnostic_origin_and_self_declared_authority_cannot_pass():
    receipt, policy, snapshot, raw = _fixture()
    changed = dict(raw)
    changed["typed_outputs"] = (
        replace(raw["typed_outputs"][0], origin=ProducerOriginKind.DIAGNOSTIC),
    )
    forged = _rebuild(changed, receipt)
    verification = _verify(forged, policy, snapshot, changed)
    assert not verification.valid
    assert any("origin_not_allowed" in item for item in verification.errors)

    with pytest.raises(ValueError, match="DIAGNOSTIC"):
        replace(
            policy.expected_records[-1],
            origin=ProducerOriginKind.DIAGNOSTIC,
        )
    with pytest.raises(ValueError, match="one exact origin"):
        replace(
            policy.expected_records[-1],
            origin=(
                ProducerOriginKind.REAL_EXECUTION,
                ProducerOriginKind.ROOT_FROZEN_PUBLIC_SOURCE,
            ),
        )

    field_names = {item.name for item in fields(type(receipt))}
    assert not field_names & {"real", "eligible", "authority", "gate"}
    with pytest.raises(TypeError):
        type(receipt)(**receipt.to_dict(), authority=True)


def test_public_api_freeze_remains_byte_identical():
    path = Path(__file__).parents[2] / "src/datadiff_osc/public_api_freeze.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PUBLIC_FREEZE_SHA256
