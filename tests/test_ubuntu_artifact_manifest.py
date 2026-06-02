import json
import importlib.util
from pathlib import Path


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

    for rel_path in paths:
        path = root / rel_path
        assert path.exists(), rel_path
        assert path.stat().st_size > 0, rel_path
