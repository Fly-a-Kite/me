from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from datadiff_osc.runtime._phase6_formal_target_version_producer import (
    FORMAL_TARGET_VERSION_PREPARATION_MODE,
    TargetVersionRawMaterial,
    prepare_target_version_replay,
    target_version_producer_preview,
)
from datadiff_osc.runtime._receipt_producers import TargetPackageObservation


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
    for forbidden in ("--execute", "--output", "out.json", "--dynamic-plan"):
        with pytest.raises(SystemExit) as raised:
            main([forbidden])
        assert raised.value.code == 2
