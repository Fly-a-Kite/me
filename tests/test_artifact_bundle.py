import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_ubuntu_artifact_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_ubuntu_artifact_bundle", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bundle_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_module)

collect_manifest_paths = bundle_module.collect_manifest_paths
bundle_members = bundle_module.bundle_members
main = bundle_module.main
sha256_file = bundle_module.sha256_file
validate_paths = bundle_module.validate_paths
write_bundle = bundle_module.write_bundle


def test_collect_manifest_paths_deduplicates_and_sorts():
    manifest = {
        "summary_document": "reports/summary.md",
        "ablation_audit": {"markdown": "reports/audit.md", "csv": "reports/audit.csv"},
        "pattern_analyses": {"null_agg_topk": "reports/pattern.md", "null_agg_topk_csv": "reports/pattern.csv"},
        "environment": {"pip": "reports/pip.txt"},
        "primary_artifact": {"triage": "bugs/x/triage.json"},
        "experiments": [
            {
                "manifest": "runs/e.json",
                "summary": "reports/s.md",
                "analysis": "reports/a.md",
                "aggregate_csv": "reports/s-aggregates.csv",
            },
            {
                "manifest": "runs/e.json",
                "summary": "reports/s.md",
                "analysis": "reports/a2.md",
                "aggregate_csv": "reports/s2-aggregates.csv",
            },
        ],
    }

    assert collect_manifest_paths(manifest) == [
        "bugs/x/triage.json",
        "reports/a.md",
        "reports/a2.md",
        "reports/audit.csv",
        "reports/audit.md",
        "reports/pattern.csv",
        "reports/pattern.md",
        "reports/pip.txt",
        "reports/s-aggregates.csv",
        "reports/s.md",
        "reports/s2-aggregates.csv",
        "reports/summary.md",
        "runs/e.json",
    ]


def test_validate_paths_reports_missing_and_empty_files(tmp_path):
    (tmp_path / "present.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing=missing.txt; empty=empty.txt"):
        validate_paths(["present.txt", "missing.txt", "empty.txt"], root=tmp_path)


def test_bundle_members_expands_directories_without_duplicates(tmp_path):
    bug_dir = tmp_path / "bugs" / "bug_x"
    bug_dir.mkdir(parents=True)
    (bug_dir / "case.json").write_text("{}", encoding="utf-8")
    (bug_dir / "triage.json").write_text("{}", encoding="utf-8")

    assert bundle_members(
        [
            "bugs/bug_x",
            "bugs/bug_x/triage.json",
        ],
        root=tmp_path,
    ) == [
        "bugs/bug_x/case.json",
        "bugs/bug_x/triage.json",
    ]


def test_write_bundle_has_unique_members_when_directory_and_child_are_listed(tmp_path):
    bug_dir = tmp_path / "bugs" / "bug_x"
    bug_dir.mkdir(parents=True)
    (bug_dir / "case.json").write_text("{}", encoding="utf-8")
    (bug_dir / "triage.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "bundle.tar.gz"

    write_bundle(["bugs/bug_x", "bugs/bug_x/triage.json"], output, root=tmp_path)

    with tarfile.open(output, "r:gz") as tar:
        names = tar.getnames()
    assert names == [
        "bugs/bug_x/case.json",
        "bugs/bug_x/triage.json",
    ]


def test_write_bundle_is_deterministic_and_normalizes_metadata(tmp_path):
    bug_dir = tmp_path / "bugs" / "bug_x"
    bug_dir.mkdir(parents=True)
    case_path = bug_dir / "case.json"
    triage_path = bug_dir / "triage.json"
    case_path.write_text("{}\n", encoding="utf-8")
    triage_path.write_text("{}\n", encoding="utf-8")
    case_path.chmod(0o600)
    triage_path.chmod(0o777)

    output_a = tmp_path / "bundle-a.tar.gz"
    output_b = tmp_path / "bundle-b.tar.gz"

    write_bundle(["bugs/bug_x", "bugs/bug_x/triage.json"], output_a, root=tmp_path)
    write_bundle(["bugs/bug_x", "bugs/bug_x/triage.json"], output_b, root=tmp_path)

    assert sha256_file(output_a) == sha256_file(output_b)

    with tarfile.open(output_a, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers()}

    assert members["bugs/bug_x/case.json"].mtime == 0
    assert members["bugs/bug_x/case.json"].uid == 0
    assert members["bugs/bug_x/case.json"].gid == 0
    assert members["bugs/bug_x/case.json"].uname == ""
    assert members["bugs/bug_x/case.json"].gname == ""
    assert members["bugs/bug_x/case.json"].mode == 0o644
    assert members["bugs/bug_x/triage.json"].mode == 0o755


def test_repo_ubuntu_artifact_bundle_matches_manifest_and_checksum(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "reports" / "ubuntu-datafusion-artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_members = bundle_members(collect_manifest_paths(manifest), root=root)

    bundle_path = root / "reports" / "ubuntu-datafusion-artifact-bundle.tar.gz"
    with tarfile.open(bundle_path, "r:gz") as tar:
        assert tar.getnames() == expected_members

    checksum_path = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    assert checksum_path.read_text(encoding="utf-8") == f"{sha256_file(bundle_path)}  {bundle_path.name}\n"

    rebuilt_path = tmp_path / "rebuilt-ubuntu-artifact-bundle.tar.gz"
    write_bundle(collect_manifest_paths(manifest), rebuilt_path, root=root)
    assert sha256_file(rebuilt_path) == sha256_file(bundle_path)


def test_main_writes_bundle_checksum_and_expanded_member_count(tmp_path, monkeypatch, capsys):
    bug_dir = tmp_path / "bugs" / "bug_x"
    bug_dir.mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (tmp_path / "runs").mkdir()

    (tmp_path / "reports" / "summary.md").write_text("# summary\n", encoding="utf-8")
    (tmp_path / "reports" / "roadmap.md").write_text("# roadmap\n", encoding="utf-8")
    (tmp_path / "reports" / "pip.txt").write_text("pkg==1\n", encoding="utf-8")
    (bug_dir / "case.json").write_text("{}\n", encoding="utf-8")
    (bug_dir / "triage.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "runs" / "exp.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "reports" / "exp-summary.md").write_text("# exp\n", encoding="utf-8")
    (tmp_path / "reports" / "exp-analysis.md").write_text("# analysis\n", encoding="utf-8")
    (tmp_path / "reports" / "exp.csv").write_text("a,b\n", encoding="utf-8")

    manifest = {
        "summary_document": "reports/summary.md",
        "methodology_roadmap": "reports/roadmap.md",
        "environment": {"pip_freeze": "reports/pip.txt"},
        "primary_artifact": {
            "directory": "bugs/bug_x",
            "triage_json": "bugs/bug_x/triage.json",
        },
        "experiments": [
            {
                "manifest": "runs/exp.json",
                "summary": "reports/exp-summary.md",
                "analysis": "reports/exp-analysis.md",
                "aggregate_csv": "reports/exp.csv",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "dist" / "artifact.tar.gz"
    expected_paths = collect_manifest_paths(manifest)
    expected_members = bundle_members(expected_paths, root=tmp_path)

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
        assert tar.getnames() == expected_members

    out = capsys.readouterr().out
    assert f"bundle={output_path}" in out
    assert f"sha256={checksum_path}" in out
    assert f"files={len(expected_members)}" in out


def test_main_fails_without_writing_outputs_for_empty_inputs(tmp_path, monkeypatch, capsys):
    bug_dir = tmp_path / "bugs" / "bug_x"
    bug_dir.mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (tmp_path / "runs").mkdir()

    (tmp_path / "reports" / "summary.md").write_text("# summary\n", encoding="utf-8")
    (tmp_path / "reports" / "roadmap.md").write_text("", encoding="utf-8")
    (tmp_path / "reports" / "pip.txt").write_text("pkg==1\n", encoding="utf-8")
    (bug_dir / "triage.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "runs" / "exp.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "reports" / "exp-summary.md").write_text("# exp\n", encoding="utf-8")
    (tmp_path / "reports" / "exp-analysis.md").write_text("# analysis\n", encoding="utf-8")
    (tmp_path / "reports" / "exp.csv").write_text("a,b\n", encoding="utf-8")

    manifest = {
        "summary_document": "reports/summary.md",
        "methodology_roadmap": "reports/roadmap.md",
        "environment": {"pip_freeze": "reports/pip.txt"},
        "primary_artifact": {
            "triage_json": "bugs/bug_x/triage.json",
        },
        "experiments": [
            {
                "manifest": "runs/exp.json",
                "summary": "reports/exp-summary.md",
                "analysis": "reports/exp-analysis.md",
                "aggregate_csv": "reports/exp.csv",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "dist" / "artifact.tar.gz"
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

    with pytest.raises(FileNotFoundError, match=r"empty=reports/roadmap.md"):
        main()

    assert not output_path.exists()
    assert not checksum_path.exists()
    assert capsys.readouterr().out == ""
