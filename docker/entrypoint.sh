#!/usr/bin/env bash
# First run: download the MaleCNS files and build the graph cache into the /data volume.
# Later runs: files are present, skip straight to serving.
set -euo pipefail

MIN_SYN="${FLYPAINT_MIN_SYNAPSES:-5}"
PORT="${FLYPAINT_PORT:-8000}"

if [ "${1:-serve}" = "serve" ]; then
  if [ ! -f "/data/cache/brain_traced_min${MIN_SYN}.npz" ]; then
    echo "flypaint: preparing connectome (first run: ~1.1 GB download + <1 min build, ~2 GB RAM)"
    flypaint --min-synapses "$MIN_SYN" prepare
  fi
  echo "flypaint: serving on http://0.0.0.0:${PORT}"
  exec flypaint --min-synapses "$MIN_SYN" serve --host 0.0.0.0 --port "$PORT" --runs-dir /app/runs/web
fi

# any other command runs the flypaint CLI, e.g.  docker compose run --rm flypaint validate
exec flypaint --min-synapses "$MIN_SYN" "$@"
