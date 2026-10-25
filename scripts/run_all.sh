#!/usr/bin/env bash
# Full pipeline: train -> evaluate -> robustness -> figures
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PYTHON:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

EPOCHS="${1:-10}"

$PYTHON -m src.train --models all --epochs "$EPOCHS"
$PYTHON -m src.evaluate --models all
$PYTHON -m src.robustness --models cnn_bilstm proposed
$PYTHON -m src.visualize --all
