"""Version environment registry for cross-version differential execution.

A *version environment* is an isolated Python interpreter (usually a venv) whose
installed data-processing packages differ from the current process. The registry
maps a stable ``env_id`` to that interpreter plus its probed package versions.

This module is intentionally dependency-free so it can be imported by any
environment, including ones that do not have the project installed.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = "datadiff-version-environments-v1"
DEFAULT_REGISTRY_RELPATH = "experiments/version_environments.json"
DEFAULT_PROBE_PACKAGES: tuple[str, ...] = (
    "pandas",
    "polars",
    "duckdb",
    "pyarrow",
    "datafusion",
    "chdb",
)

_PROBE_CODE = (
    "import importlib.metadata as m, json, sys\n"
    "names = json.loads(sys.argv[1])\n"
    "out = {}\n"
    "for name in names:\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except Exception:\n"
    "        out[name] = ''\n"
    "print(json.dumps(out, sort_keys=True))\n"
)


class VersionEnvironmentError(ValueError):
    """Raised for malformed registries or unusable environments."""


def repository_root() -> Path:
    """Locate the project root (the directory containing ``pyproject.toml``)."""

    override = os.environ.get("DATADIFF_REPOSITORY_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class VersionEnvironment:
    env_id: str
    interpreter: str
    packages: Mapping[str, str] = field(default_factory=dict)
    label: str = ""
    repository_root: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "env_id": self.env_id,
            "interpreter": self.interpreter,
            "packages": dict(sorted(self.packages.items())),
            "label": self.label,
            "repository_root": self.repository_root,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VersionEnvironment":
        env_id = str(payload.get("env_id", "") or "").strip()
        interpreter = str(payload.get("interpreter", "") or "").strip()
        if not env_id:
            raise VersionEnvironmentError("version environment requires a non-empty env_id")
        if not interpreter:
            raise VersionEnvironmentError(f"version environment {env_id!r} requires an interpreter")
        packages = payload.get("packages") or {}
        if not isinstance(packages, Mapping):
            raise VersionEnvironmentError(f"version environment {env_id!r} packages must be a mapping")
        return cls(
            env_id=env_id,
            interpreter=interpreter,
            packages={str(k): str(v) for k, v in packages.items()},
            label=str(payload.get("label", "") or ""),
            repository_root=str(payload.get("repository_root", "") or ""),
        )

    @property
    def interpreter_path(self) -> Path:
        return Path(self.interpreter)

    def require_interpreter(self) -> Path:
        path = self.interpreter_path
        if not path.is_file():
            raise VersionEnvironmentError(
                f"version environment {self.env_id!r} interpreter is missing: {path}"
            )
        return path


def probe_package_versions(
    interpreter: str | Path,
    packages: Sequence[str] = DEFAULT_PROBE_PACKAGES,
    *,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """Return the versions of ``packages`` installed in ``interpreter``."""

    if not packages:
        return {}
    completed = subprocess.run(
        [str(interpreter), "-c", _PROBE_CODE, json.dumps(list(packages))],
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise VersionEnvironmentError(
            f"version probe failed for {interpreter}: {completed.stderr.decode('utf-8', 'replace')[:400]}"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise VersionEnvironmentError(f"version probe returned invalid JSON: {exc}") from exc
    return {str(k): str(v) for k, v in payload.items()}


def environment_from_interpreter(
    env_id: str,
    interpreter: str | Path,
    *,
    label: str = "",
    packages: Sequence[str] = DEFAULT_PROBE_PACKAGES,
    repository_root_path: str | Path | None = None,
) -> VersionEnvironment:
    versions = probe_package_versions(interpreter, packages)
    return VersionEnvironment(
        env_id=env_id,
        interpreter=str(interpreter),
        packages=versions,
        label=label,
        repository_root=str(repository_root_path or repository_root()),
    )


def registry_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("DATADIFF_VERSION_ENVIRONMENTS", "").strip()
    if override:
        return Path(override)
    return repository_root() / DEFAULT_REGISTRY_RELPATH


def load_registry(path: str | Path | None = None) -> dict[str, VersionEnvironment]:
    target = registry_path(path)
    if not target.is_file():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise VersionEnvironmentError("version environment registry must be a JSON object")
    raw = payload.get("environments")
    if not isinstance(raw, list):
        raise VersionEnvironmentError("version environment registry requires an 'environments' list")
    registry: dict[str, VersionEnvironment] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise VersionEnvironmentError("each version environment entry must be an object")
        environment = VersionEnvironment.from_dict(entry)
        if environment.env_id in registry:
            raise VersionEnvironmentError(f"duplicate version environment id: {environment.env_id!r}")
        registry[environment.env_id] = environment
    return registry


def write_registry(
    environments: Iterable[VersionEnvironment],
    path: str | Path | None = None,
) -> Path:
    target = registry_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "environments": [environment.to_dict() for environment in environments],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe and register version environments.")
    parser.add_argument("--interpreter", action="append", default=[], help="interpreter path to probe")
    parser.add_argument("--env-id", action="append", default=[], help="env_id for the matching --interpreter")
    parser.add_argument("--label", action="append", default=[], help="optional label for the matching --interpreter")
    parser.add_argument("--out", default="", help="registry output path")
    args = parser.parse_args(argv)

    if len(args.interpreter) != len(args.env_id):
        parser.error("--interpreter and --env-id must be provided the same number of times")
    labels = list(args.label) + [""] * (len(args.interpreter) - len(args.label))
    environments = [
        environment_from_interpreter(env_id, interpreter, label=label)
        for interpreter, env_id, label in zip(args.interpreter, args.env_id, labels)
    ]
    if args.out:
        target = write_registry(environments, args.out)
        print(f"wrote {target}")
    else:
        print(json.dumps([e.to_dict() for e in environments], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
