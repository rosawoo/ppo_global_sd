# Global vs. state-dependent σ in PPO

Does a PPO Gaussian policy do better with a **global** log-std (a learned vector, the CleanRL /
SB3 default) or a **state-dependent** one (a head that outputs log σ(s)), in tasks where the
right amount of action noise differs across states? Three domains:

| env id | domain | where precision matters |
|---|---|---|
| `parking-v0` (highway-env), **deep dive** | goal-conditioned vehicle control | the last metres before the spot |
| `Pendulum-v1` (gymnasium) | classic control | (premise) holding it upright |
| `simglucose-adolescent1-v0` | insulin dosing, type-1 diabetes | low blood glucose: an overdose causes hypoglycaemia |

The two policy classes are identical except for how σ is computed (`ppo.py --std-mode
global|state`). See `CLAUDE.md` (spec), `DECISIONS.md` (every design choice), `REPORT.md`
(results and the reasoning about *why*).

## Quick start (laptop)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps -r requirements-simglucose.txt   # --no-deps matters, see DECISIONS.md D27
python -m pytest -q                                        # unit tests
python ppo.py --env-id Pendulum-v1 --std-mode state        # one run -> runs/Pendulum-v1/state/seed1/
python scripts/sweep.py --smoke                            # 1 seed, 50k steps, 3 envs x 2 modes
```

## Full experiment (FASRC Cannon)

`cluster/README.md` has the copy-paste commands. In short:

```bash
bash cluster/setup_env.sh        # once, inside `salloc -p test -c 4 --mem=8G -t 0-01:00`
bash cluster/submit.sh smoke     # 6 short runs
bash cluster/submit.sh all       # per-env hyper-parameter tuning -> final comparison at the tuned configs
                                 #   (10 seeds) -> noise probes -> all figures and tables
bash cluster/submit.sh main      # optional, in parallel: the spec's untuned config (robustness check)
```

## Layout

```
ppo.py                       single-file PPO (CleanRL-derived); --std-mode global|state|state_separate|state_detach
envs/wrappers.py             env construction, goal-relative logging, shared obs/reward normaliser
envs/simglucose_env.py       basal-insulin control env built from simglucose's simulator
scripts/sweep.py             (env x mode x seed) grid, locally or as Slurm task files; --config uses tuned settings
scripts/tune.py              paired Optuna tuning per env (both sigma arms per trial) -> configs/tuned.json
scripts/run_task.py          one task of a Slurm array (resumable)
scripts/eval.py              periodic evaluation + per-step dumps (obs, goal-relative state, mu, sigma, action)
analysis/curves.py           H1/H2: IQM + 95% CIs (rliable), noise cost, optimisation diagnostics
analysis/sigma.py            H3/H4: sigma vs state, executed-action noise, heatmaps, sigma over training
analysis/probe.py            noise-tolerance probe: what action noise costs in each state (env cloning)
analysis/probe_report.py     precision requirement c(s) vs learned sigma(s), heterogeneity index
analysis/parking_deepdive.py parking: outcomes, trajectories, approach profile, final-state CDFs
analysis/demos.py            rendered rollouts with the sigma ellipse drawn on every frame
cluster/                     FASRC scripts: setup_env.sh, submit.sh, array.sbatch, config.sh, README.md
prelim/                      preliminary local results (2 CPU cores; see REPORT.md)
```
