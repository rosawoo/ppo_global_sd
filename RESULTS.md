# Results: global vs. state-dependent σ in PPO (Cannon study, 22–23 Nov 2026)

The full study from `cluster/README.md` (`submit.sh all` + `submit.sh main`) ran on FASRC Cannon
(CPU only) from 22 Nov 16:43 to 23 Nov 12:03 EST. 180 training runs, 136 tuning trials and 60
noise probes; nothing failed. The decision rules are the ones fixed in advance in `REPORT.md`.

## In plain words

**The question.** PPO explores by adding Gaussian noise to the action it would otherwise take. The
noise has a size, σ. Does it help to let σ depend on the state, so that the agent can explore
widely where precision doesn't matter and act carefully where it does?

**The two arms** (`--sigma-mode global` / `state`):
* **Global σ** (standard PPO): one learned noise level per action dimension, the same in every state.
* **σ(s)**: the policy network outputs a noise level for each state.

Everything else is the same: network, hyper-parameters, the 10 training seeds and the held-out
evaluation episodes (20 every 25k steps, reset seeds from 100,000).

**The three tasks**, each with a "precision gradient" (states where a small mistake costs a lot):
* **Pendulum** (`Pendulum-v1`, 500k steps): swing up and balance. Main metric: return.
* **Parking** (`parking-v0`, 2M steps): park a car in a marked spot. Main metric: share of
  evaluation episodes that end parked.
* **Simglucose** (`simglucose-adolescent1-v0`, 1M steps): set the basal insulin rate for a
  simulated adolescent with type 1 diabetes. Main metric: return. Also reported: the share of
  episodes meeting the clinical targets (over 70% of time in 70–180 mg/dL, under 4% below 70).

**Three comparisons:**
* **Main**: both arms at the best *shared* tuned configuration, so they differ only in σ.
* **Per-arm**: each arm at its own best tuned configuration.
* **Untuned**: the spec's default hyper-parameters.

**How to read the numbers.** Each number is the IQM (interquartile mean: the mean of the middle
half) of the 10 runs' final scores, where final = the mean of the last 3 evaluations. The interval
is the 95% bootstrap confidence interval of the difference σ(s) − global (20,000 resamples). A
difference counts ("ahead") only if that interval leaves out zero; otherwise it is "no clear
difference with 10 seeds".

## H1: does one arm win? Main comparison (tuned, shared config)

| task | main metric | global σ | σ(s) | difference [95% CI] | verdict |
|---|---|---|---|---|---|
| Pendulum | return | −184.9 | −176.5 | **+8.4** [+0.2, +18.6] (+4.6%) | **σ(s) ahead**, barely |
| Parking | parked | 48.3% | 36.4% | −11.9 pp [−47.2, +28.9] | no clear difference |
| Simglucose | return | 474.8 | 465.4 | **−9.3** [−34.2, −3.2] (−2.0%) | **global σ ahead** |

On Simglucose the gap shows most in safety: episodes meeting the clinical targets **96% → 61%**
(−35 pp [−61, −12]), and episodes where glucose fell below 54 mg/dL at least once **0% → 14%**.

### Does the verdict hold up?

| comparison | Pendulum (return) | Parking (parked) | Simglucose (return) |
|---|---|---|---|
| main (shared tuned config) | σ(s) ahead: −185 → −176 | no clear difference: 48% → 36% | global ahead: 475 → 465 |
| per-arm (each arm its own config) | no clear difference: −185 → −181, +4.1 [−3.9, +14.4] | same config for both, so the same runs: 48% → 36% | global ahead by 0.16%: 475.1 → 474.4, −0.8 [−1.4, −0.2] |
| untuned defaults | **global ahead**: −193 → −378, −185 [−342, −48] | neither arm ever parks: 0% → 0% | no clear difference: 363 → 297, −66 [−146, +39] |
| tuning trials: σ(s) scored higher in | 56% of 25 configs (Wilcoxon p = 0.69) | 57% of 21 (p = 0.34) | 50% of 26 (p = 0.62) |

* **Simglucose's big gap is mostly a tuning effect.** With its own configuration (a higher
  learning rate and longer rollouts), σ(s) meets the clinical targets in 99% of episodes vs 96%
  for global σ (+2.5 pp [−0.6, +4.4]), with no low-glucose events. Only a 0.16% return gap is left.
* **Pendulum's σ(s) win is fragile.** Its interval barely leaves out zero, and it disappears at
  σ(s)'s own best config.
* **Tuning matters more than the σ parameterisation.** Untuned, σ(s) is far worse on Pendulum and
  nothing parks. Tuned, both arms reach similar levels in every task.
* **No arm is consistently better across configurations.** In the paired tuning trials each arm
  wins about half the time. One hint: in parking's top 6 configurations, σ(s) scored higher in all 6
  (p = 0.03), but the final runs at the best of them did not confirm it.

**H1 verdict:** no consistent winner. With tuned hyper-parameters a state-dependent σ is neither
clearly better nor clearly worse than a global σ. The one clear loss (Simglucose, shared config)
largely goes away when σ(s) gets its own hyper-parameters.

## H2: is σ(s) harder to train?

| | Pendulum | Parking | Simglucose |
|---|---|---|---|
| runs that never reached the task's bar (global σ → σ(s)) | 0 → 0 of 10 | 4 → 6 of 10 | 0 → **3** of 10 |
| spread of final return across runs (std) | 12.7 → 4.7 | 1.82 → 1.84 | 0.7 → **26.5** |
| updates with a KL spike (> 0.02) | 31% → 41% | 81% → 81% | 0% → **25%** |
| runs where σ hit its floor | 0 → 0 | 0 → 0 | 0 → **10** of 10 |

The bar is a return of −300 on Pendulum, and the success criterion in at least half the episodes
on Parking and Simglucose.

**H2 verdict:** yes on Simglucose at the shared config: σ collapses to its floor in some states in
every σ(s) run, and training is much less stable. No on Pendulum, where σ(s) runs are *more*
consistent. Parking is a wash. Source: `results/summary.md`, H2 tables.

## H3: does σ(s) put its noise where noise is cheap?

The noise probe measured, in states visited by the trained policies, how much return each unit of
noise variance costs, c(s). If c is uneven, an ideal σ(s) is small where c is large.

* **There was something to gain.** The cost of noise is very uneven across states. The ratio of
  c's arithmetic to geometric mean, the factor by which an ideal σ(s) could cut the cost of
  noise, is 7 (Pendulum), 54 (parking) and 167 (Simglucose) in the global-σ runs.
* **σ(s) does become state-dependent.** 73–99% of the variance of log σ is across states within
  a run.
* **Where noise is expensive, and what σ(s) did:**

  | task · action | where noise costs more | what σ(s) does | ρ(log σ, log c): median, runs < 0 |
  |---|---|---|---|
  | Pendulum · torque | away from the top: per unit variance, noise near upright costs ~0.03–0.1× what it costs in the swing-up | σ 1.3× *larger* near the top | −0.22, 9 of 10 |
  | Parking · steering | near the spot (2× in the global-σ runs) | σ cut ~2× within 3 m | −0.20, 8 of 10 |
  | Parking · acceleration | (acceleration noise is almost free near the spot) | σ 1.5× larger near the spot | −0.09, 6 of 10 |
  | Simglucose · basal insulin | not at low glucose (0.65×) | σ 2× larger at low glucose | −0.04, 7 of 10 |

**H3 verdict:** σ(s) moves noise in the right direction (more where it is cheap, less where it is
costly), but only weakly: |ρ| ≤ 0.22. A large part of what reaches the environment is set by the
action clip, not by σ. σ(s) policies keep the mean action outside the action box in most states
(Pendulum 89% vs 10% for global σ; parking 83% vs 65%), with very large σ (Pendulum: σ ≈ 7 for a
torque range of ±2). Sources: `results/sigma_summary.md`, `results/probe_summary.md`.

## H4: the plain-English statements from REPORT.md

1. *"In parking, σ(s) cuts steering noise within 3 m of the spot while keeping acceleration noise
   roughly constant."* **Partly supported.** Steering σ is 0.84 within 3 m vs 1.51 beyond (×1.8
   cut), but acceleration σ *rises* near the spot (6.2 vs 4.1). (`results/parking_deepdive.md`)
2. *"The global-σ policy is just as precise in the last metres."* **Supported.** Within 3 m its
   executed noise is lower than σ(s)'s: acceleration 0.27 vs 0.66, steering 0.15 vs 0.24. It
   isn't explained by slowing down: the approach speed is the same (13.1 vs 13.3 m/s).
3. *"In Pendulum, σ(s) is smallest at the top although noise costs more during the swing-up."*
   **Refuted at the tuned config.** Noise does cost more in the swing-up, but σ(s) is largest at
   the top, which is the efficient choice. The untuned preliminary runs had shown the opposite.
4. *"In Simglucose, both policies stop insulin at low BG; executed noise there is ~0."* Not
   checked here; see `figures/simglucose-adolescent1-v0_effective_noise.png`.

## What is where

* `results/summary.md`: H1/H2 tables for the main comparison. Also `results/perarm/` and
  `results/default_config/` for the other two comparisons.
* `results/sigma_summary.md`, `results/probe_summary.md`, `results/parking_deepdive.md`,
  `results/tuning_summary.md`: H3, the probe (main comparison only), the parking deep dive and the
  tuning analysis. All but the tuning summary also come as `.json`.
* `results/tuning_trials.csv`: every tuning trial with both arms' scores and its hyper-parameters.
* `results/eval_curves.csv`: every evaluation of every run (180 runs; columns config, env, arm,
  seed, then each run's `evals.csv`).
* `figures/`: 111 figures, the same layout as `results/`.
* `configs/tuned.json`: the tuned configurations (shared and per arm).
* Not in git (gitignored, on Cannon under `~/ppo_global_sd`): `runs*/` (checkpoints and per-update
  logs), `tuning/` (Optuna journals) and `demos/*.mp4`.

## How it was run, and what changed

* Tuning: Optuna TPE, paired trials (both arms at the same configuration, 2 seeds each). 64,
  32 and 40 trials for Pendulum, parking and Simglucose. Best shared trials: 53, 30 and 33.
* Final runs: 10 seeds × 2 arms × 3 tasks at the shared config (`runs_final`) and at each arm's own
  config (`runs_final_perarm`), plus the untuned defaults (`runs`).
* For parking, tuning picked the same trial (30) as best for both arms. So the per-arm parking runs
  repeat the main ones, and they reproduced them exactly, every evaluation identical.
* **No hyper-parameters, seeds, environments or evaluation settings were changed.**
* Three fixes were needed to run:
  * `cluster/config.sh`: smoke-test partition.
  * `requirements-simglucose.txt`: missing dependencies of `requests`.
  * `analysis/sigma.py`: a division by zero that crashed the σ analysis of the untuned runs.

  See the commit messages.
* Not run: the optional `extra` and `ablations` stages of `cluster/submit.sh`.
