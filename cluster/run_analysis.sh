#!/bin/bash
# Run every analysis on whatever has finished. Safe to re-run. Small enough for a login node
# while runs are still coming in; for the full study use the sbatch wrapper
# (cluster/analysis.sbatch, which `submit.sh analysis` submits).
set -uo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
export MPLBACKEND=Agg SDL_VIDEODRIVER=offscreen
run() { echo; echo ">>> $*"; "$@" || echo "!!! failed (continuing): $*"; }

analyze_root() {  # runs_root out_dir fig_dir
  local root=$1 out=$2 fig=$3
  [ -d "$root" ] || { echo "(no $root, skipped)"; return 0; }
  run "$PY" analysis/curves.py       --runs-root "$root" --out-dir "$out" --fig-dir "$fig"
  run "$PY" analysis/sigma.py        --runs-root "$root" --out-dir "$out" --fig-dir "$fig"
  run "$PY" analysis/probe_report.py --runs-root "$root" --out-dir "$out" --fig-dir "$fig"
  [ -d "$root/parking-v0" ] && run "$PY" analysis/parking_deepdive.py --runs-root "$root" --out-dir "$out" \
      --fig-dir "$fig/parking_deepdive"
  [ -d "$root/parking-nowalls-v0" ] && run "$PY" analysis/parking_deepdive.py --env parking-nowalls-v0 \
      --runs-root "$root" --out-dir "$out" --fig-dir "$fig/parking_deepdive"
  return 0
}

# 1) tuning: best configs, paired global-vs-state comparison across all tuned configs, sensitivities
[ -d "$TUNE_DIR" ] && run "$PY" scripts/tune.py report --study-dir "$TUNE_DIR" --out-dir results --fig-dir figures/tuning

# 2) final comparison at the shared tuned config (the primary result) ...
analyze_root "$RUNS_ROOT" results figures
# ... with each arm at its own best config, and at the spec's untuned default config
analyze_root "$PERARM_ROOT" results/perarm figures/perarm
analyze_root "$DEFAULT_ROOT" results/default_config figures/default_config

# 3) demo videos of training seed 1 of the primary runs
run "$PY" analysis/demos.py --runs-root "$RUNS_ROOT" --envs parking-v0 Pendulum-v1 --seed 1
run "$PY" analysis/demos.py --runs-root "$RUNS_ROOT" --envs simglucose-adolescent1-v0 --seed 1 --frame-skip 4

# 4) hyper-parameter ablations, each in its own runs root
for d in "$ABL_ROOT"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  run "$PY" analysis/curves.py --runs-root "$d" --out-dir "results/abl_$name" --fig-dir "figures/abl_$name"
  run "$PY" analysis/sigma.py  --runs-root "$d" --out-dir "results/abl_$name" --fig-dir "figures/abl_$name"
done
echo; echo ">>> done: results/*.md (start with results/summary.md and results/tuning_summary.md) and figures/"
