#!/bin/bash
# One task of a job array = line $SLURM_ARRAY_TASK_ID of the task file passed as $1
# (a training run or a probe; see scripts/run_task.py). Submitted by cluster/submit.sh,
# which sets the partition, time limit, memory, array range and job name.
#SBATCH -c 1
#SBATCH -o cluster/logs/%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit.sh always submits from the repo root
source cluster/config.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=offscreen MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) partition=${SLURM_JOB_PARTITION} restarts=${SLURM_RESTART_COUNT:-0}"
echo "cpu: $(lscpu | grep -m1 'Model name' | cut -d: -f2 | xargs)"
"$PY" scripts/run_task.py "$1" "$SLURM_ARRAY_TASK_ID"
