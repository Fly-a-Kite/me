from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import datadiff_osc.runtime._phase6_formal_target_version_producer as producer_module
from datadiff_osc.runtime._phase6_formal_target_version_producer import (
    FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION,
    FORMAL_TARGET_VERSION_EVIDENCE_WRITER_MODE,
    FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_MODE,
    FORMAL_TARGET_VERSION_PREPARATION_MODE,
    TargetVersionFutureExecutionAuthority,
    TargetVersionEvidenceWriteRequest,
    TargetVersionPublicSourceBinding,
    TargetVersionRawMaterial,
    inspect_target_version_execution_capability,
    prepare_target_version_replay,
    require_formal_target_version_execution_authority,
    target_version_producer_preview,
    verify_target_version_replay_evidence,
    write_target_version_replay_evidence,
)
from datadiff_osc.runtime._receipt_producers import TargetPackageObservation
from datadiff_osc.runtime._semantic_replay import replay_runtime_admission


_SOURCE_DIGEST = "osc-phase6-source-snapshot-" + "a" * 64
_SHA = "a" * 64
_TARGETS = (
    ("pandas", "pandas", "3.0.5"),
    ("polars", "polars", "1.43.0"),
    ("duckdb", "duckdb", "1.5.5"),
    ("pyarrow", "pyarrow", "25.0.0"),
    ("datafusion", "datafusion", "54.0.0"),
    ("chdb", "chdb", "4.2.1"),
    ("pysqlite3-binary", "sqlite", "0.5.4.post2"),
)
_RAW_SHA = {
    "pandas": "def430f070766d7d545517b3c2a08afbb59f865c4297281d00fafcc4e25a123f",
    "polars": "f9e79734733ed9c25f2c0a822a9e2f3bf8d7765201dc507293cd594d1e25e23c",
    "duckdb": "d2f9d34a9ba3976cae39e4bb69b464227851c907493e9d1395c91710f3a5a721",
    "pyarrow": "752ee190964d03405cf7e53a2ba55c173c9f69a0f87a53383f19cd760cd80982",
    "datafusion": "27287ace58a4003d8e4b5373d7c688cb0266a8e01ddc45d7f48ec803c515c42c",
    "chdb": "25dfd4a21d3b82d9c83cbbe46c2965f37f9d8f8e89c5c2703dae107afd19a2b9",
    "pysqlite3-binary": "250725f2b802e5d21db15739aa8ae3f18477dd27cee6803eef8f263594ab6d0c",
}
_PUBLIC_SOURCE_ROOT = Path(
    "/tmp/phase6_formal_execution_authority_r1.Smql7A/target_version_public_sources"
)
_FORMAL_EVIDENCE_ROOT = Path(
    "/tmp/phase6_formal_latest_target_remediation_r1.20260724/formal_evidence"
)
_CURRENT_SOURCE_SNAPSHOT = Path(
    "/tmp/phase6_formal_target_version_writer_source_snapshot_r1b.JmwKM6/source-snapshot.json"
)
_CURRENT_SOURCE_SNAPSHOT_SHA256 = (
    "52560616ed557bc6a2e7a6c2df8fe80a6de95fbe27ab8b2117480dbde1adf07c"
)
_CURRENT_SOURCE_DIGEST = (
    "osc-phase6-source-snapshot-a46503ed6be524f2b84902bbf813223ef2a476031394025b3c0676070491cda9"
)
_CURRENT_SOURCE_GIT_HEAD = "75a99c6df1f40115edc0803b72694cedf30a6fec"
_TARGET_ENVIRONMENT_ROOT = Path("/tmp/phase6_formal_latest_target_remediation_r1.20260724")
_TARGET_INTERPRETER = _TARGET_ENVIRONMENT_ROOT / "venv/bin/python"
_TARGET_INTERPRETER_SHA256 = (
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
)
_TARGET_LOCK = _TARGET_ENVIRONMENT_ROOT / "source/requirements-final.lock"
_TARGET_REVALIDATION = (
    _TARGET_ENVIRONMENT_ROOT / "target_version_revalidation/target-version-revalidation.json"
)
_TARGET_ENVIRONMENT_AUDIT = _TARGET_ENVIRONMENT_ROOT / "venv/environment-audit.json"
_TARGET_REVALIDATION_SHA256 = (
    "c95676504232c595f1a24408f07ecd80e1c89313502da0076e5b0483210c1bbd"
)
_TARGET_ENVIRONMENT_AUDIT_SHA256 = (
    "e7b5ad5b6e193fb6ffc99d1baa4235d1a3760764aaf33ab04d51002caa065389"
)
_TARGET_LOCK_SHA256 = "5121f122b209545f0f3b127d89faa36b59aa09c72ce0b54f5d582e7f727379dc"
_HISTORICAL_ENVIRONMENT_SOURCE_SNAPSHOT_SHA256 = (
    "4750e6ff7eb05a27883a667d3832b1425af0db0e65311e4905f26e4f99b9f531"
)
_HISTORICAL_ENVIRONMENT_SOURCE_DIGEST = (
    "osc-phase6-source-snapshot-a598ba904fff4300704ce99a4a4f7089e2d16c821c0539bb8dc1124983b4c85d"
)


def _public_raw(name: str) -> bytes:
    raw = (_PUBLIC_SOURCE_ROOT / f"pypi-{name}.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _RAW_SHA[name]
    return raw


def _materials() -> tuple[TargetVersionRawMaterial, ...]:
    return tuple(
        TargetVersionRawMaterial(
            distribution_name=name,
            import_name=import_name,
            installed_metadata=(
                f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n"
            ).encode(),
            public_version_source=_public_raw(name),
        )
        for name, import_name, version in _TARGETS
    )


def _prepare(**overrides: object):
    values: dict[str, object] = {
        "source_digest": _SOURCE_DIGEST,
        "source_snapshot_sha256": _SHA,
        "requirements_lock_sha256": _SHA,
        "target_revalidation_sha256": _SHA,
        "environment_audit_sha256": _SHA,
        "materials": _materials(),
        "environment": (("platform", "linux"), ("venv", "isolated")),
        "python_runtime": (("implementation", "cpython"), ("version", "3.12")),
    }
    values.update(overrides)
    return prepare_target_version_replay(**values)  # type: ignore[arg-type]


def _write_request(preparation, output_root: Path) -> TargetVersionEvidenceWriteRequest:
    return TargetVersionEvidenceWriteRequest(
        output_root=str(output_root),
        source_digest=preparation.source_digest,
        source_snapshot_sha256=preparation.source_snapshot_sha256,
        requirements_lock_sha256=preparation.requirements_lock_sha256,
        target_revalidation_sha256=preparation.target_revalidation_sha256,
        environment_audit_sha256=preparation.environment_audit_sha256,
        receipt_digest=preparation.receipt.digest,
    )


def _execution_authority(
    tmp_path: Path, **overrides: object
) -> TargetVersionFutureExecutionAuthority:
    values: dict[str, object] = {
        "execution_root": str(tmp_path),
        "future_output_root": str(tmp_path / "formal_evidence"),
        "execution_command_sha256": "a" * 64,
        "source_snapshot_path": str(_CURRENT_SOURCE_SNAPSHOT),
        "source_snapshot_sha256": _CURRENT_SOURCE_SNAPSHOT_SHA256,
        "source_digest": _CURRENT_SOURCE_DIGEST,
        "source_git_head": _CURRENT_SOURCE_GIT_HEAD,
        "requirements_lock_path": str(_TARGET_LOCK),
        "requirements_lock_sha256": _TARGET_LOCK_SHA256,
        "target_revalidation_path": str(_TARGET_REVALIDATION),
        "target_revalidation_sha256": _TARGET_REVALIDATION_SHA256,
        "environment_audit_path": str(_TARGET_ENVIRONMENT_AUDIT),
        "environment_audit_sha256": _TARGET_ENVIRONMENT_AUDIT_SHA256,
        "historical_environment_source_snapshot_sha256": (
            _HISTORICAL_ENVIRONMENT_SOURCE_SNAPSHOT_SHA256
        ),
        "historical_environment_source_digest": _HISTORICAL_ENVIRONMENT_SOURCE_DIGEST,
        "target_interpreter": str(_TARGET_INTERPRETER),
        "target_interpreter_sha256": _TARGET_INTERPRETER_SHA256,
        "public_source_root": str(_PUBLIC_SOURCE_ROOT),
        "public_sources": tuple(
            TargetVersionPublicSourceBinding(
                distribution_name=name,
                path=str(_PUBLIC_SOURCE_ROOT / f"pypi-{name}.json"),
                sha256=_RAW_SHA[name],
            )
            for name, _, _ in _TARGETS
        ),
    }
    values.update(overrides)
    authority = TargetVersionFutureExecutionAuthority(**values)  # type: ignore[arg-type]
    if "execution_command_sha256" not in overrides:
        authority = replace(
            authority,
            execution_command_sha256=authority.expected_execution_command_sha256,
        )
    return authority


def test_valid_preparation_is_private_canonical_and_never_authority(tmp_path: Path):
    preparation = _prepare()

    assert preparation.receipt.source_digest == _SOURCE_DIGEST
    assert len(preparation.receipt.packages) == 7
    assert all(item.matches_latest for item in preparation.receipt.packages)
    assert {item.version_source_kind for item in preparation.receipt.packages} == {
        "pypi_json"
    }
    assert preparation.authority_eligible is False
    assert preparation.formal_evidence_created is False
    assert preparation.gate_credit is False
    assert preparation.candidate_confirmed is False
    assert preparation.bug_claimed is False
    payload = json.loads(preparation.payload_json)
    assert payload["mode"] == FORMAL_TARGET_VERSION_PREPARATION_MODE
    assert payload["all_targets_match_latest"] is True
    assert payload["public_raw_sha256"] == [
        _RAW_SHA[name] for name, _, _ in _TARGETS
    ]
    assert list(tmp_path.iterdir()) == []


def test_writer_stages_only_canonical_temporary_tree_and_replays_exactly(
    tmp_path: Path,
):
    preparation = _prepare()
    output_root = tmp_path / "formal_evidence"
    request = _write_request(preparation, output_root)

    result = write_target_version_replay_evidence(
        preparation=preparation,
        request=request,
    )

    assert result.output_root == str(output_root)
    assert result.authority_eligible is False
    assert result.formal_evidence_created is False
    assert result.gate_credit is False
    assert result.candidate_confirmed is False
    assert result.bug_claimed is False
    assert verify_target_version_replay_evidence(
        preparation=preparation,
        request=request,
    ) == result
    assert sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    ) == sorted(
        [
            "raw/target_version_replay/manifest.json",
            "receipts/target_version_replay.json",
            *[
                f"raw/target_version_replay/{name}/{leaf}"
                for name, _, _ in _TARGETS
                for leaf in ("installed-METADATA", "pypi.json")
            ],
        ]
    )
    manifest_raw = (
        output_root / "raw/target_version_replay/manifest.json"
    ).read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest["schema_version"] == FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION
    assert manifest["mode"] == FORMAL_TARGET_VERSION_EVIDENCE_WRITER_MODE
    assert manifest["receipt"]["receipt_digest"] == preparation.receipt.digest
    assert manifest["authority_eligible"] is False
    assert manifest["gate_credit"] is False
    receipt = json.loads(
        (output_root / "receipts/target_version_replay.json").read_text(encoding="utf-8")
    )
    assert replay_runtime_admission(
        envelope_type=receipt["type"],
        schema_version=receipt["schema_version"],
        payload=receipt["payload"],
        subject_kind="target_packages",
        subject_ids=preparation.receipt.package_ids,
    ) == ()
    assert not _FORMAL_EVIDENCE_ROOT.exists()


def test_writer_refuses_stale_bindings_existing_paths_and_symlink_escape(
    tmp_path: Path,
):
    preparation = _prepare()
    stale_root = tmp_path / "stale-output"
    stale_request = _write_request(preparation, stale_root)
    with pytest.raises(ValueError, match="source snapshot SHA-256 mismatch"):
        write_target_version_replay_evidence(
            preparation=preparation,
            request=replace(stale_request, source_snapshot_sha256="b" * 64),
        )
    assert not stale_root.exists()
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        write_target_version_replay_evidence(
            preparation=preparation,
            request=replace(stale_request, receipt_digest="wrong-receipt-digest"),
        )
    assert not stale_root.exists()

    existing_root = tmp_path / "existing-output"
    existing_root.mkdir()
    sentinel = existing_root / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        write_target_version_replay_evidence(
            preparation=preparation,
            request=_write_request(preparation, existing_root),
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"

    link = tmp_path / "linked-parent"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        write_target_version_replay_evidence(
            preparation=preparation,
            request=_write_request(preparation, link / "escaped-output"),
        )
    with pytest.raises(ValueError, match="parent traversal"):
        TargetVersionEvidenceWriteRequest(
            output_root=str(tmp_path / ".." / "escaped-output"),
            source_digest=preparation.source_digest,
            source_snapshot_sha256=preparation.source_snapshot_sha256,
            requirements_lock_sha256=preparation.requirements_lock_sha256,
            target_revalidation_sha256=preparation.target_revalidation_sha256,
            environment_audit_sha256=preparation.environment_audit_sha256,
            receipt_digest=preparation.receipt.digest,
        )


def test_writer_revalidation_detects_raw_or_manifest_mutation(tmp_path: Path):
    preparation = _prepare()
    output_root = tmp_path / "formal_evidence"
    request = _write_request(preparation, output_root)
    write_target_version_replay_evidence(preparation=preparation, request=request)

    metadata_path = output_root / "raw/target_version_replay/pandas/installed-METADATA"
    metadata_path.write_bytes(metadata_path.read_bytes() + b"X")
    with pytest.raises(ValueError, match="installed metadata bytes mismatch"):
        verify_target_version_replay_evidence(
            preparation=preparation,
            request=request,
        )


def test_writer_revalidation_detects_manifest_byte_mutation(tmp_path: Path):
    preparation = _prepare()
    output_root = tmp_path / "formal_evidence"
    request = _write_request(preparation, output_root)
    write_target_version_replay_evidence(preparation=preparation, request=request)

    manifest_path = output_root / "raw/target_version_replay/manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest is not exact canonical binding"):
        verify_target_version_replay_evidence(
            preparation=preparation,
            request=request,
        )


def test_writer_failure_removes_staged_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    preparation = _prepare()
    output_root = tmp_path / "formal_evidence"
    request = _write_request(preparation, output_root)
    original = producer_module._write_new_bytes
    calls = 0

    def fail_after_first(path: Path, raw: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        original(path, raw)

    monkeypatch.setattr(producer_module, "_write_new_bytes", fail_after_first)
    with pytest.raises(OSError, match="injected staging failure"):
        write_target_version_replay_evidence(preparation=preparation, request=request)
    assert not output_root.exists()
    assert list(tmp_path.glob(".formal_evidence.stage-*")) == []


def test_execution_capability_collects_actual_target_venv_without_authority_write(
    tmp_path: Path,
):
    authority = _execution_authority(tmp_path)

    capability = inspect_target_version_execution_capability(authority=authority)

    assert capability.authority_id == authority.authority_id
    assert capability.source_digest == _CURRENT_SOURCE_DIGEST
    assert capability.preparation.receipt.all_match_latest is True
    assert [record.distribution_name for record in capability.metadata_records] == [
        name for name, _, _ in _TARGETS
    ]
    assert [record.import_name for record in capability.metadata_records] == [
        import_name for _, import_name, _ in _TARGETS
    ]
    assert all(record.installed_metadata for record in capability.metadata_records)
    assert all(
        Path(record.metadata_path).is_relative_to(_TARGET_ENVIRONMENT_ROOT / "venv")
        for record in capability.metadata_records
    )
    assert capability.formal_execution_authorized is False
    assert capability.authority_eligible is False
    assert capability.formal_evidence_created is False
    assert capability.gate_credit is False
    assert capability.candidate_confirmed is False
    assert capability.bug_claimed is False
    assert not Path(authority.future_output_root).exists()
    assert not _FORMAL_EVIDENCE_ROOT.exists()
    with pytest.raises(PermissionError, match="separate successor authority"):
        require_formal_target_version_execution_authority(authority)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_snapshot_sha256", "b" * 64),
        ("source_digest", "osc-phase6-source-snapshot-" + "b" * 64),
        ("source_git_head", "b" * 40),
        ("target_revalidation_sha256", "b" * 64),
        ("environment_audit_sha256", "b" * 64),
        ("target_interpreter_sha256", "b" * 64),
    ],
)
def test_execution_capability_rejects_provenance_substitutions_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
):
    authority = _execution_authority(tmp_path, **{field: value})

    def unexpected_probe(*args: object, **kwargs: object) -> object:
        raise AssertionError("metadata probe must not run for a stale binding")

    monkeypatch.setattr(producer_module.subprocess, "run", unexpected_probe)
    with pytest.raises(ValueError, match="mismatch"):
        inspect_target_version_execution_capability(authority=authority)
    assert not Path(authority.future_output_root).exists()


@pytest.mark.parametrize(
    "source_git_head",
    [
        "a" * 39,
        "a" * 41,
        "a" * 64,
        "A" * 40,
        "z" * 40,
        "synthetic-git-head",
    ],
)
def test_execution_capability_rejects_malformed_git_head_before_probe(
    tmp_path: Path, source_git_head: str
):
    with pytest.raises(ValueError, match="source_git_head"):
        _execution_authority(tmp_path, source_git_head=source_git_head)
    assert not (tmp_path / "formal_evidence").exists()


def test_execution_capability_rejects_missing_or_malformed_authority_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def unexpected_probe(*args: object, **kwargs: object) -> object:
        raise AssertionError("metadata probe must not run without valid authority")

    monkeypatch.setattr(producer_module.subprocess, "run", unexpected_probe)
    with pytest.raises(TypeError, match="requires an authority input"):
        inspect_target_version_execution_capability(authority=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not authorized"):
        _execution_authority(tmp_path, formal_execution_authorized=True)
    with pytest.raises(ValueError, match="frozen target"):
        TargetVersionPublicSourceBinding(
            distribution_name="pandas",
            path=str(_PUBLIC_SOURCE_ROOT / "pypi-pandas.json"),
            sha256="b" * 64,
        )
    existing_output = tmp_path / "formal_evidence"
    existing_output.mkdir()
    authority = _execution_authority(tmp_path)
    with pytest.raises(ValueError, match="must be an absent direct child"):
        inspect_target_version_execution_capability(authority=authority)
    assert existing_output.is_dir()


def test_execution_capability_rejects_duplicate_probe_transcript_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authority = _execution_authority(tmp_path)
    malformed = b'{"records":[],"records":[]}'

    monkeypatch.setattr(
        producer_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=malformed,
            stderr=b"",
        ),
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_target_version_execution_capability(authority=authority)
    assert not Path(authority.future_output_root).exists()
    assert not _FORMAL_EVIDENCE_ROOT.exists()


def test_unsupported_pypi_json_alias_fails_closed():
    material = _materials()[0]

    with pytest.raises(ValueError, match="target version source is not authority eligible"):
        TargetPackageObservation(
            distribution_name=material.distribution_name,
            import_name=material.import_name,
            installed_version="3.0.5",
            latest_version="3.0.5",
            installed_metadata=material.installed_metadata,
            version_source_kind="pypi-json",
            version_source=material.public_version_source,
        )


@pytest.mark.parametrize(
    ("materials", "message"),
    [
        (_materials()[::-1], "canonical target sequence"),
        (_materials()[:-1], "canonical target sequence"),
        (_materials() + (_materials()[0],), "canonical target sequence"),
        (
            _materials()[:-1]
            + (
                replace(_materials()[-1], import_name="pysqlite3"),
            ),
            "canonical target sequence",
        ),
    ],
)
def test_target_identity_order_and_sqlite_import_binding_fail_closed(
    materials: tuple[TargetVersionRawMaterial, ...], message: str
):
    with pytest.raises(ValueError, match=message):
        _prepare(materials=materials)


def test_raw_source_metadata_and_latest_mismatches_fail_closed():
    materials = _materials()
    with pytest.raises(ValueError, match="public target version source SHA mismatch: pandas"):
        _prepare(
            materials=(
                replace(materials[0], public_version_source=b'{"info":{"name":"pandas","version":"9.9.9"}}'),
                *materials[1:],
            )
        )
    duplicate_json = b'{"info":{"name":"pandas","version":"3.0.5","version":"9"}}'
    with pytest.raises(ValueError, match="public target version source SHA mismatch: pandas"):
        _prepare(
            materials=(replace(materials[0], public_version_source=duplicate_json), *materials[1:])
        )
    with pytest.raises(ValueError, match="target latest version mismatch"):
        _prepare(
            materials=(
                replace(
                    materials[0],
                    installed_metadata=b"Metadata-Version: 2.4\nName: pandas\nVersion: 0.0.0\n\n",
                ),
                *materials[1:],
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_digest", "", "target source digest"),
        ("source_snapshot_sha256", "x" * 64, "source_snapshot_sha256"),
        ("environment", (("venv", "isolated"), ("platform", "linux")), "uniquely sorted"),
        ("python_runtime", (), "target Python runtime"),
    ],
)
def test_malformed_authority_bindings_fail_closed(
    field: str, value: object, message: str
):
    with pytest.raises(ValueError, match=message):
        _prepare(**{field: value})


def test_preview_and_cli_reject_execution_output_and_plan_flags(capsys: pytest.CaptureFixture[str]):
    preview = target_version_producer_preview()
    assert preview["mode"] == FORMAL_TARGET_VERSION_PREPARATION_MODE
    assert preview["writes_durable_output"] is False
    assert preview["launches_adapter"] is False
    assert preview["launches_subprocess"] is False
    assert preview["makes_network_request"] is False
    assert preview["test_only_staged_writer_available"] is True
    assert preview["test_only_execution_capability_available"] is True
    assert preview["formal_execution_authorized"] is False

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/osc/run_phase6_formal_target_version_producer.py"
    )
    namespace = {
        "__name__": "phase6_target_version_runner_test",
        "__file__": str(script),
    }
    exec(compile(script.read_text(), str(script), "exec"), namespace)
    main = namespace["main"]
    assert main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["writes_durable_output"] is False
    assert output["test_only_staged_writer_available"] is True
    assert output["test_only_execution_capability_available"] is True
    assert output["formal_execution_authorized"] is False
    for forbidden in ("--execute", "--output", "out.json", "--dynamic-plan"):
        with pytest.raises(SystemExit) as raised:
            main([forbidden])
        assert raised.value.code == 2
