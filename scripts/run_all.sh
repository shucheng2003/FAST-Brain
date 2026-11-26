#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m fast_brain.run --dataset rnn "$@"
python -m fast_brain.run --dataset sddes "$@"
python -m fast_brain.summarize --results-dir results
