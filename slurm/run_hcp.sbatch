#!/usr/bin/env bash
#SBATCH --job-name=fast-hcp
#SBATCH --output=results/fast-hcp-%j.out
#SBATCH --error=results/fast-hcp-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m fast_brain.run --dataset hcp "$@"
