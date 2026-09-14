#!/usr/bin/env bash
# Run the frozen interpreter against this worktree's source, so long-running
# controllers execute the code under development rather than the base venv's
# editable install. Override the base interpreter with DATADIFF_PYTHON_BASE.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${DATADIFF_PYTHON_BASE:-/data1/lbw/xjx/datadiff_fuzz_lab/.venv/bin/python}" "$@"
