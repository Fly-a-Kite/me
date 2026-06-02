#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Compatibility wrapper: the final paper harness uses the authority/freeze-aware
# closed-loop launcher as the only 24h entrypoint.
export DATADIFF_ROOT_DIR="${DATADIFF_ROOT_DIR:-${ROOT_DIR}}"
export DATADIFF_TMUX_SESSION="${DATADIFF_TMUX_SESSION:-datadiff-live-24h}"

exec bash "${ROOT_DIR}/scripts/start_closed_loop_24h_tmux.sh"
