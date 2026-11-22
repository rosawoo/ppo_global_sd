# Global vs. state-dependent σ in PPO — report

Status: **implementation complete, smoke test + Pendulum pilot done, full sweep not yet
launched** (user decision, 2026-11-22: third env = LunarLanderContinuous-v3; sweep to be
started by the user; `state_separate` ablation to be decided after the main results).
Sections marked _pending_ are filled in from `results/summary.md` and
`results/sigma_summary.md` once the sweep finishes.

Prior work: Andrychowicz et al. 2021, "What Matters in On-Policy Reinforcement
Learning? A Large-Scale Empirical Study" (arXiv:2006.05990) found no clear benefit of
per-state σ on MuJoCo locomotion. This study tests it in tasks with a precision
gradient over the state space.

## Setup

See `CLAUDE.md` (spec) and `DECISIONS.md` (every deviation). One PPO config for all
envs; the only CLI difference between the arms is `--std-mode`.

## Compute

Device: CPU (Apple Silicon, 10 cores), 1 torch thread per run, runs parallelised by
`scripts/sweep.py`.

| env | steps / run | train SPS (1 core) | wall-clock / run | source |
|---|---|---|---|---|
| parking-v0 | 2M | ≈ 460 | ≈ 85 min est. (72 train + 13 eval) | extrapolated from 50k-step smoke run (2.2 min) |
| Pendulum-v1 | 500k | ≈ 3,700 (4 runs in parallel) | 2.2 min measured | `runs_pilot/` |
| LunarLanderContinuous-v3 | 1M | ≈ 3,400 | ≈ 5–6 min est. | extrapolated from smoke run |

Estimated full sweep (3 envs × 2 modes × 10 seeds) ≈ 30 core-hours, ≈ 4 h wall-clock
with 8 workers, dominated by parking.

## Smoke test (1 seed, 50k steps, `runs_smoke/`) — not a result

All 6 runs finished, no NaNs. Stochastic eval return, step 0 → 50k:

| env | global | state |
|---|---|---|
| parking-v0 | −50.4 → −14.4 (success 0) | −50.7 → −15.6 (success 0) |
| Pendulum-v1 | −1275 → −1190 | −1270 → −1148 |
| LunarLanderContinuous-v3 | −183 → +110 | −196 → +58 |

## Pendulum pilot (2 seeds, full 500k steps, `runs_pilot/`) — sanity check only

Purpose: confirm both modes learn before committing compute (D26). n = 2, so no
claims: global ended at ≈ −200 on both seeds; state at ≈ −300 and ≈ −590, with mean
log σ ≈ −3.2 at the end vs. −1.4 for global. The full 10-seed sweep is what tests
this.

## How to run the rest

```
source .venv/bin/activate
python -m pytest -q                               # 31 tests
python scripts/sweep.py --workers 8               # full sweep -> runs/ (resumable; re-run the same command)
python analysis/curves.py                         # -> figures/, results/summary.md      (H1, H2)
python analysis/sigma.py                          # -> figures/, results/sigma_summary.md (H3, H4)
python analysis/demos.py --seed 1                 # -> demos/{env}/{mode}/*.mp4
# ablations from the spec (both modes get the same extra flags):
python scripts/sweep.py --runs-root runs_abl/ent0.01 --extra "--ent-coef 0.01"
python scripts/sweep.py --envs parking-v0 --runs-root runs_abl/lam0.9 --extra "--gae-lambda 0.9"
```

## H1 — performance _(pending)_

## H2 — optimization difficulty _(pending)_

## H3 — does σ(s) track precision requirements? _(pending)_

## H4 — interpretability statements _(pending)_

## Demos _(pending)_
