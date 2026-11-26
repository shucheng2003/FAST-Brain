#!/usr/bin/env bash
#SBATCH --job-name=fast-rnn
#SBATCH --output=results/fast-rnn-%j.out
#SBATCH --error=results/fast-rnn-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m fast_brain.run --dataset rnn "$@"
