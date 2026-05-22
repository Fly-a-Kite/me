import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_upstream_issue_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_upstream_issue_bundle", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bundle_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_module)

collect_issue_paths = bundle_module.collect_issue_paths
main = bundle_module.main
sha256_file = bundle_module.sha256_file
validate_paths = bundle_module.validate_paths
write_bundle = bundle_module.write_bundle


def test_collect_issue_paths_uses_only_minimal_upstream_files():
    manifest = {
        "primary_artifact": {
            "upstream_issue_draft": "bugs/x/upstream_issue.md",
            "standalone_reproducer": "bugs/x/standalone.py",
            "triage_json": "bugs/x/triage.json",
            "preflight_boundary_script": "scripts/preflight.py",
            "preflight_boundary_output": "reports/preflight.txt",
            "upstream_issue_checklist": "reports/checklist.md",
        },
        "experiments": [
            {"manifest": "runs/large.json"},
        ],
    }

    assert collect_issue_paths(manifest) == [
        "bugs/x/standalone.py",
        "bugs/x/upstream_issue.md",
        "reports/checklist.md",
        "reports/preflight.txt",
        "scripts/preflight.py",
    ]


def test_validate_paths_reports_missing_and_empty_issue_files(tmp_path):
    (tmp_path / "present.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing=missing.py; empty=empty.md"):
        validate_paths(["present.py", "missing.py", "empty.md"], root=tmp_path)


def test_write_bundle_contains_minimal_issue_members(tmp_path):
    (tmp_path / "bugs" / "x").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bugs" / "x" / "upstream_issue.md").write_text("# issue\n", encoding="utf-8")
    (tmp_path / "scripts" / "preflight.py").write_text("print('check')\n", encoding="utf-8")
    output = tmp_path / "issue.tar.gz"

    write_bundle(["bugs/x/upstream_issue.md", "scripts/preflight.py"], output, root=tmp_path)

    with tarfile.open(output, "r:gz") as tar:
        assert tar.getnames() == [
            "bugs/x/upstream_issue.md",
            "scripts/preflight.py",
        ]


def test_write_bundle_is_deterministic_and_normalizes_metadata(tmp_path):
    (tmp_path / "bugs" / "x").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    issue_path = tmp_path / "bugs" / "x" / "upstream_issue.md"
    script_path = tmp_path / "scripts" / "preflight.py"
    issue_path.write_text("# issue\n", encoding="utf-8")
    script_path.write_text("print('check')\n", encoding="utf-8")
    issue_path.chmod(0o600)
    script_path.chmod(0o777)

    output_a = tmp_path / "issue-a.tar.gz"
    output_b = tmp_path / "issue-b.tar.gz"

    write_bundle(["bugs/x/upstream_issue.md", "scripts/preflight.py"], output_a, root=tmp_path)
    write_bundle(["bugs/x/upstream_issue.md", "scripts/preflight.py"], output_b, root=tmp_path)

    assert sha256_file(output_a) == sha256_file(output_b)

    with tarfile.open(output_a, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers()}

    assert members["bugs/x/upstream_issue.md"].mtime == 0
    assert members["bugs/x/upstream_issue.md"].uid == 0
    assert members["bugs/x/upstream_issue.md"].gid == 0
    assert members["bugs/x/upstream_issue.md"].uname == ""
    assert members["bugs/x/upstream_issue.md"].gname == ""
    assert members["bugs/x/upstream_issue.md"].mode == 0o644
    assert members["scripts/preflight.py"].mode == 0o755


def test_repo_upstream_issue_bundle_matches_manifest_and_checksum(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "reports" / "ubuntu-datafusion-artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_members = collect_issue_paths(manifest)

    bundle_path = root / manifest["primary_artifact"]["upstream_issue_bundle"]
    with tarfile.open(bundle_path, "r:gz") as tar:
        assert tar.getnames() == expected_members

    checksum_path = root / manifest["primary_artifact"]["upstream_issue_bundle_sha256"]
    assert checksum_path.read_text(encoding="utf-8") == f"{sha256_file(bundle_path)}  {bundle_path.name}\n"

    rebuilt_path = tmp_path / "rebuilt-upstream-issue-bundle.tar.gz"
    write_bundle(expected_members, rebuilt_path, root=root)
    assert sha256_file(rebuilt_path) == sha256_file(bundle_path)


def test_main_writes_bundle_checksum_and_summary(tmp_path, monkeypatch, capsys):
    (tmp_path / "bugs" / "x").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "bugs" / "x" / "upstream_issue.md").write_text("# issue\n", encoding="utf-8")
    (tmp_path / "bugs" / "x" / "standalone.py").write_text("print('repro')\n", encoding="utf-8")
    (tmp_path / "scripts" / "preflight.py").write_text("print('check')\n", encoding="utf-8")
    (tmp_path / "reports" / "preflight.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "reports" / "checklist.md").write_text("# checklist\n", encoding="utf-8")

    manifest = {
        "primary_artifact": {
            "upstream_issue_draft": "bugs/x/upstream_issue.md",
            "standalone_reproducer": "bugs/x/standalone.py",
            "preflight_boundary_script": "scripts/preflight.py",
            "preflight_boundary_output": "reports/preflight.txt",
            "upstream_issue_checklist": "reports/checklist.md",
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "dist" / "issue.tar.gz"

    monkeypatch.setattr(bundle_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        bundle_module,
        "parse_args",
        lambda: bundle_module.argparse.Namespace(
            manifest=str(manifest_path),
            output=str(output_path),
        ),
    )

    assert main() == 0

    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    assert output_path.exists()
    assert checksum_path.exists()
    assert checksum_path.read_text(encoding="utf-8") == f"{sha256_file(output_path)}  {output_path.name}\n"

    with tarfile.open(output_path, "r:gz") as tar:
        assert tar.getnames() == collect_issue_paths(manifest)

    out = capsys.readouterr().out
    assert f"bundle={output_path}" in out
    assert f"sha256={checksum_path}" in out
    assert "files=5" in out


def test_main_fails_without_writing_outputs_for_missing_inputs(tmp_path, monkeypatch, capsys):
    (tmp_path / "bugs" / "x").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "bugs" / "x" / "upstream_issue.md").write_text("# issue\n", encoding="utf-8")
    (tmp_path / "bugs" / "x" / "standalone.py").write_text("print('repro')\n", encoding="utf-8")
    (tmp_path / "scripts" / "preflight.py").write_text("print('check')\n", encoding="utf-8")
    (tmp_path / "reports" / "checklist.md").write_text("# checklist\n", encoding="utf-8")

    manifest = {
        "primary_artifact": {
            "upstream_issue_draft": "bugs/x/upstream_issue.md",
            "standalone_reproducer": "bugs/x/standalone.py",
            "preflight_boundary_script": "scripts/preflight.py",
            "preflight_boundary_output": "reports/preflight.txt",
            "upstream_issue_checklist": "reports/checklist.md",
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "dist" / "issue.tar.gz"
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")

    monkeypatch.setattr(bundle_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        bundle_module,
        "parse_args",
        lambda: bundle_module.argparse.Namespace(
            manifest=str(manifest_path),
            output=str(output_path),
        ),
    )

    with pytest.raises(FileNotFoundError, match=r"missing=reports/preflight.txt"):
        main()

    assert not output_path.exists()
    assert not checksum_path.exists()
    assert capsys.readouterr().out == ""
