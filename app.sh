#!/bin/bash
# Entry point — packages and runs the benchmark

set -e

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

echo "Traces:    ${MLFLOW_TRACKING_URI:-(not set)}"

exec $PYTHON benchmark.py --config config.yaml
