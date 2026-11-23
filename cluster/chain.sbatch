#!/bin/bash
# Runs after tuning has finished (dependency set by `submit.sh all`): exports the tuned configs
# and submits the final comparison, the probes and the analysis from inside this job.
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0-01:00
#SBATCH -o cluster/logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
bash cluster/submit.sh after-tuning
