# Preliminary local results (2 CPU cores, 2026-11-22)

**Not the study's results.** Produced on a 2-core VM, before hyper-parameter tuning, at the spec's
shared default config. They validate the pipeline and give early evidence on mechanisms. The real
experiment is the cluster pipeline in `cluster/README.md`. Interpretation is in `REPORT.md`
("Preliminary results").

| folder | runs | notes |
|---|---|---|
| `pendulum/` | Pendulum-v1, global and state, seeds 1–6, full 500k steps | 6 of the planned 10 seeds; noise probes: 80 states × 6 repeats per run |
| `pilots/` | Simglucose, 1 seed per mode, 400k steps (planned 1M); parking-v0, 1 seed per mode, 300k steps (planned 2M) | pilots, not comparisons; probes with 60 states × 4 repeats |

Each folder has `summary.md` (H1/H2 tables), `sigma_summary.md` (H3), `probe_summary.md` (noise
cost by state), and `figures/`. `pilots/parking_deepdive.md` and `pilots/figures/parking_deepdive/`
hold the parking deep dive.

Known limitation of these particular runs: they were trained before a fix to the logging of two
diagnostics, so `mean_outside_bounds_frac` and `action_clipped_frac` in their `updates.csv` read 0
(DECISIONS.md D34). The same quantities computed from the eval dumps by `sigma.py` (executed noise,
fraction of states with the mean outside the action box) are correct.

Re-create (needs the local `runs/` and `runs_pilot/`, which are not in git):

```bash
bash prelim/make_prelim.sh
```
