#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "ubuntu-datafusion-artifact-manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "datafusion-upstream-issue-bundle.tar.gz"


ISSUE_BUNDLE_KEYS = [
    "upstream_issue_draft",
    "standalone_reproducer",
    "preflight_boundary_script",
    "preflight_boundary_output",
    "upstream_issue_checklist",
]


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = collect_issue_paths(manifest)
    validate_paths(paths, root=PROJECT_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(paths, output_path, root=PROJECT_ROOT)
    digest = sha256_file(output_path)
    digest_path = output_path.with_suffix(output_path.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    print(f"bundle={output_path}")
    print(f"sha256={digest_path}")
    print(f"files={len(paths)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal upstream issue evidence bundle.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def collect_issue_paths(manifest: dict[str, Any]) -> list[str]:
    primary = manifest.get("primary_artifact", {})
    paths = [primary.get(key, "") for key in ISSUE_BUNDLE_KEYS]
    return sorted({path for path in paths if path})


def validate_paths(paths: list[str], *, root: Path) -> None:
    missing = []
    empty = []
    for rel_path in paths:
        path = root / rel_path
        if not path.exists():
            missing.append(rel_path)
        elif path.is_file() and path.stat().st_size == 0:
            empty.append(rel_path)
    if missing or empty:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if empty:
            details.append("empty=" + ",".join(empty))
        raise FileNotFoundError("; ".join(details))


def write_bundle(paths: list[str], output_path: Path, *, root: Path) -> None:
    with output_path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz_handle:
            with tarfile.open(fileobj=gz_handle, mode="w") as tar:
                for rel_path in sorted(paths):
                    tar.add(
                        root / rel_path,
                        arcname=rel_path,
                        recursive=False,
                        filter=normalize_tarinfo,
                    )


def normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    tarinfo.mode = 0o755 if tarinfo.mode & 0o111 else 0o644
    return tarinfo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
