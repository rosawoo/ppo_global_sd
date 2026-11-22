#!/bin/bash
# One Optuna tuning worker (= one array task). Each trial trains both sigma arms x 2 tuning
# seeds = 4 single-threaded runs in parallel, hence -c 4. Workers of the same env share one
# journal file, so they cooperate on one study. Submitted by cluster/submit.sh (stage "tune").
#   $1 = env id, $2 = trials for this worker
#SBATCH -c 4
#SBATCH -o cluster/logs/%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/config.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=offscreen MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) env=$1 trials=$2 restarts=${SLURM_RESTART_COUNT:-0}"
"$PY" scripts/tune.py worker --env "$1" --n-trials "$2" --jobs "${SLURM_CPUS_PER_TASK:-4}" --study-dir "$TUNE_DIR"
