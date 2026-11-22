#!/bin/bash
# Re-create the preliminary local results in prelim/ (the runs themselves are not in git).
# Pendulum: runs/ (6 seeds per mode, full length). Pilots: runs_pilot/ (1 seed per mode,
# Simglucose 400k steps, parking 300k steps; the full runs are 1M / 2M).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export MPLBACKEND=Agg SDL_VIDEODRIVER=offscreen
run() { echo ">>> $*"; "$@" || echo "!!! failed: $*"; }
for pair in "runs:pendulum" "runs_pilot:pilots"; do
  root=${pair%%:*}; out=prelim/${pair##*:}
  run "$PY" analysis/curves.py       --runs-root "$root" --out-dir "$out" --fig-dir "$out/figures"
  run "$PY" analysis/sigma.py        --runs-root "$root" --out-dir "$out" --fig-dir "$out/figures"
  run "$PY" analysis/probe_report.py --runs-root "$root" --out-dir "$out" --fig-dir "$out/figures"
done
run "$PY" analysis/parking_deepdive.py --runs-root runs_pilot --out-dir prelim/pilots --fig-dir prelim/pilots/figures/parking_deepdive
