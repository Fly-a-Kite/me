"""Sidecar worker: execute one case in the current interpreter and emit canonical JSON.

Invoked by :mod:`datadiff.backends.version_subprocess` as
``<version-environment-python> -m datadiff.version_worker``. The worker imports the
project from ``PYTHONPATH`` and the data-processing libraries from the environment,
so the same backend code runs against a different installed package version.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Mapping

SCHEMA_VERSION = "datadiff-version-worker-v1"


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    from datadiff.backends import make_backend
    from datadiff.dsl import Case
    from datadiff.normalizer import normalize_result

    case = Case.from_dict(dict(request["case"]))
    backend_name = str(request["backend"])
    timeout_s = float(request.get("timeout_s", 5.0))
    enable_normalizer = bool(request.get("enable_normalizer", True))

    backend = make_backend(backend_name)
    result = backend.run(list(case.tables), case.program, timeout_s=timeout_s)
    normalized = normalize_result(result, case.program, enable_normalizer=enable_normalizer)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend_name,
        "status": normalized.status,
        "interpreter": sys.executable,
        "normalized": normalized.to_dict(),
    }


def main() -> int:
    raw = sys.stdin.buffer.read()
    request: dict[str, Any] = {}
    try:
        request = json.loads(raw.decode("utf-8"))
        payload = execute(request)
    except Exception as exc:  # pragma: no cover - defensive: surface worker failures
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend": str(request.get("backend", "")),
            "status": "worker_error",
            "interpreter": sys.executable,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
