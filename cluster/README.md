# Running the study on Harvard's FASRC Cannon cluster

Everything below is copy-paste. Lines starting with `#` are comments. Replace `USERNAME`
with your FASRC username (not `USERNAME@fasrc`).

**Why CPUs, not GPUs.** Each run is one single-threaded process: the actor (64-64) and
critic (256-256) are tiny, and the time goes into the simulators (highway-env, the
simglucose ODE, Box2D), which are CPU-bound Python. A GPU would sit idle while costing far
more fairshare per hour than a core. Throughput comes from running many runs at once as
Slurm job arrays.

**What runs, in order** (one command, `submit.sh all`, chains all of it):

1. **Tuning**: one paired Optuna study per environment. Every trial trains *both* σ arms at
   the same hyper-parameters (2 seeds each), so the search finds good settings for each
   env *and* shows whether one arm wins consistently across settings (DECISIONS.md D39).
2. **Final comparison**: 10 fresh seeds × {global, state} × 3 envs at the best *shared*
   tuned config (primary result), and again with each arm at its *own* best config.
3. **Noise-tolerance probes** of the 60 primary runs (what action noise costs in each state).
4. **Analysis**: every figure and table, plus demo videos.

## 0. Get the code onto GitHub (on your laptop, once)

```bash
cd ~/path/to/ppo_global_sd          # your local clone of rosawoo/ppo_global_sd
git checkout main && git pull
git checkout -b cluster-sweep
git am ~/Downloads/ppo_global_sd_changes.patch   # the patch file you were given
git push -u origin cluster-sweep
```

(If you prefer not to use git: copy the zip to the cluster with
`scp ~/Downloads/ppo_global_sd.zip USERNAME@login.rc.fas.harvard.edu:~/` and `unzip` it there.)

## 1. Log in and fetch the code

```bash
ssh USERNAME@login.rc.fas.harvard.edu     # password, then the OpenAuth 6-digit code
git clone https://github.com/rosawoo/ppo_global_sd.git
cd ppo_global_sd
git checkout cluster-sweep
spart                                     # shows the partitions you can submit to
```

If `spart` does not list `shared` or `sapphire`, put partitions you do have in
`cluster/config.sh` (`PARTITION=...`), or prefix every submit command with
`PARTITION=<yours> `. If you have a lab account, set `ACCOUNT=<lab>` the same way.

## 2. Create the Python environment (once, ~10 min)

```bash
salloc -p test -c 4 --mem=8G -t 0-01:00   # a short interactive job; don't install on the login node
bash cluster/setup_env.sh                 # Python 3.11 env at ~/envs/ppo_sd, CPU-only torch, runs the unit tests
exit                                      # leave the interactive job
```

It must end with `passed` (no failures) and `environment ready`. If `mamba` complains about
`conda init`, delete the `conda initialize` block from `~/.bashrc` (FASRC's advice) and log
in again.

## 3. Smoke test (6 short runs, ~15 min, free on the test partition)

```bash
bash cluster/submit.sh smoke
squeue --me                               # wait until the 6 smoke tasks are gone
tail -n 3 cluster/logs/smoke_*.out        # each should end with "...: done in X min"
~/envs/ppo_sd/bin/python scripts/sweep.py --smoke --dry-run    # "6 finished (skipped), 0 to run"
```

## 4. The whole study, one command

```bash
bash cluster/submit.sh all
```

This submits the three tuning arrays (8 workers per env × 4 cores each) and a small "chain" job
that waits for them. When tuning ends, the chain job exports `configs/tuned.json` and submits
the final comparison (120 runs), then the probes (60), then the analysis job. Rough
wall-clock with a reasonably free cluster: tuning ≈ 12–24 h (parking dominates), final runs
≈ 3–4 h, probes + analysis ≈ 2 h.

**Recommended, in parallel:** the spec's untuned default config, as a robustness check
(does tuning change the conclusion?). It does not depend on tuning, so it can start now:

```bash
bash cluster/submit.sh main
```

Watching it:

```bash
squeue --me                                    # what is queued / running
bash cluster/submit.sh status                  # runs per (env, mode, status) + tuning trial counts
tail -f cluster/logs/tune_parking_<JOBID>_1.out    # a tuning worker (Optuna prints every finished trial)
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS -X   # finished tasks (time, memory)
```

If anything fails or times out, run the same submit command again: finished runs are skipped,
unfinished ones restart from step 0 (NaN runs are kept as results and never retried), and
extra tuning workers just add trials to the existing studies. If the chain job reports that
tuning is not finished for every env, wait (or add workers with `bash cluster/submit.sh tune`)
and then run `bash cluster/submit.sh final`, then `probe`, then `analysis`.

## 5. Extra environments and ablations (optional, after tuning)

These use the tuned shared config automatically once `configs/tuned.json` exists (the two
variants inherit their base env's config). If default parking never learns to park (it can
settle on crashing into a wall, see DECISIONS.md D37), give the no-walls variant its own study
first: `TUNE_ENVS="parking-nowalls-v0" bash cluster/submit.sh tune`, wait, then
`bash cluster/submit.sh final` to re-export.

```bash
bash cluster/submit.sh extra        # parking without walls, Simglucose with a log-scale insulin action, LunarLander (60 runs)
bash cluster/submit.sh ablations    # σ-head variants (state_detach, state_separate), entropy 0.01, GAE λ 0.9 on parking, clip 0.25 (210 runs)
# when those are finished:
bash cluster/submit.sh probe        # probes every finished run that has no probe yet
bash cluster/submit.sh analysis     # re-makes all figures and tables
```

To chain these too, pass the job ids that `extra` / `ablations` print, colon-separated:
`AFTER=<id1>:<id2> bash cluster/submit.sh probe`.

## 6. Results

* `results/tuning_summary.md`, `figures/tuning/`: best configs per env, hyper-parameter
  importance per arm, and the paired global-vs-state comparison across every config tried
* `results/summary.md`: H1/H2 at the tuned shared config. IQM [95% CI] of return, success,
  crash rate, time in range, noise cost; state − global difference; optimisation diagnostics
* `results/sigma_summary.md`: H3/H4. σ vs. precision variables, executed noise, Spearman ρ
* `results/probe_summary.md`: what noise costs in each region; learned σ vs. required precision
* `results/parking_deepdive.md`, `figures/parking_deepdive/`: the parking deep dive
* `results/perarm/`, `results/default_config/`: the same tables with each arm at its own
  best config, and at the spec's untuned config
* `figures/`, `demos/`: every figure and the demo videos

Copy them to your laptop (run this **on your laptop**):

```bash
rsync -av USERNAME@login.rc.fas.harvard.edu:ppo_global_sd/{results,figures,demos,configs} ./cluster_results/
```

## Cost summary

| stage | what | core-hours (approx., upper bound) |
|---|---|---|
| smoke | 6 short runs | 0.5 |
| tune | parking 32 trials, Pendulum 64, Simglucose 40; 4 runs per trial | 350–500 (pruning saves ~30%) |
| final (in `all`) | 120 runs + 60 probes | 160 |
| main (untuned defaults) | 60 runs | 75 |
| extra | 60 runs | 85 |
| ablations | 210 runs | 275 |

Budgets are set in `cluster/config.sh` (`TUNE_WORKERS`, `TUNE_TRIALS`, time limits). Per task:
training 1 core / 3 GB; tuning worker 4 cores / 12 GB; probe 1 core / 4 GB; analysis 16 GB.

## Troubleshooting

* `Invalid partition` / `Invalid account`: set `PARTITION` / `ACCOUNT` (step 1).
* Tasks pending with `(Priority)`: normal. `serial_requeue` adds a lot of capacity:
  `PARTITION=shared,sapphire,serial_requeue bash cluster/submit.sh main`. A pre-empted task is
  requeued automatically and its run restarts from scratch (a pre-empted tuning trial is just
  abandoned; the worker starts new trials).
* `ModuleNotFoundError` in a log: the environment is not at `~/envs/ppo_sd`; set `ENV_PREFIX`
  in `cluster/config.sh` (or re-run `setup_env.sh`).
* Outputs go to `runs*/` and `tuning/` inside the repo (a few GB in total; home quota is 100 GB).
  To use scratch instead, set `RUNS_ROOT`, `PERARM_ROOT`, `DEFAULT_ROOT`, `ABL_ROOT` and `TUNE_DIR`
  to paths under `/n/netscratch/<lab>/Lab/$USER/` in `cluster/config.sh`.
