#!/bin/bash
# All analyses (figures/, results/, demos/). Submitted by cluster/submit.sh (stage "analysis"),
# usually with a dependency on the training / probe arrays.
#SBATCH -c 1
#SBATCH -o cluster/logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/config.sh
export OMP_NUM_THREADS=1 SDL_VIDEODRIVER=offscreen MPLBACKEND=Agg PYTHONUNBUFFERED=1
bash cluster/run_analysis.sh
