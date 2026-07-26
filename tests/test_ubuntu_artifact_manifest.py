import json
import importlib.util
from pathlib import Path
import tarfile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_ubuntu_artifact_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_ubuntu_artifact_bundle", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bundle_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_module)

collect_manifest_paths = bundle_module.collect_manifest_paths


def test_ubuntu_datafusion_artifact_manifest_paths_exist():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "reports" / "ubuntu-datafusion-artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = collect_manifest_paths(manifest)
    bundle_path = root / "reports" / "ubuntu-datafusion-artifact-bundle.tar.gz"

    with tarfile.open(bundle_path, "r:gz") as tar:
        file_members = {
            member.name: member
            for member in tar.getmembers()
            if member.isfile()
        }

    for rel_path in paths:
        member = file_members.get(rel_path)
        if member is not None:
            assert member.size > 0, rel_path
            continue
        directory_prefix = rel_path.rstrip("/") + "/"
        assert any(name.startswith(directory_prefix) for name in file_members), rel_path
