# Decisions

Every non-obvious choice, with the reason. All of them apply identically to both arms
(`--std-mode global` / `state`) unless explicitly stated.

## Implementation

**D1. Base file.** `ppo.py` starts from CleanRL `ppo_continuous_action.py` (master,
fetched 2026-11-22; committed unmodified in the first commit so the diff is auditable).
Removed: wandb / HF-hub / video capture. Kept: single file, `Args` + `tyro`,
TensorBoard, `SyncVectorEnv`. Added: per-update `updates.csv` and per-eval `evals.csv`
next to the TensorBoard logs so the analysis does not have to parse event files.

**D2. Autoreset mode = `SAME_STEP`.** CleanRL's rollout loop was written for the
pre-1.0 vector API, where the obs returned on a `done` step is the first obs of the
next episode. Gymnasium ≥1.0 defaults to `NEXT_STEP`, which would insert one bogus
transition (ignored action, reward 0) per episode into the PPO batch. With 100-step
parking episodes that is ≥1% junk data. `SAME_STEP` restores the semantics the
algorithm assumes.

**D3. Normalization lives in `VecNormalizer`, not in per-env wrappers.** CleanRL wraps
every sub-env in its own `NormalizeObservation`/`NormalizeReward`, so with 8 envs there
are 8 different running statistics and none can be handed to the eval env. We need
(a) a separate eval env that sees exactly the training normalization, frozen, and
(b) raw observations in the `.npz` dumps. So one shared running mean/std (same update
rule as gymnasium's `RunningMeanStd`, obs clip ±10, return-based reward scaling with
clip ±10) is applied in `ppo.py` and saved in the checkpoint. The reward-return
accumulator is reset on `terminated or truncated` (the gym-0.2x behaviour CleanRL's
results were produced with).

**D4. Time-limit truncation bootstraps the value (`--bootstrap-truncated`, default on).**
CleanRL treats truncation as termination. Pendulum *only* ever truncates (200 steps)
and parking truncates at 100 steps, with time not in the observation; treating that as
a true terminal state makes the value target non-Markov. On truncation (not
termination) we add `γ·V(s_final)` to the last reward, using `final_obs` from the
vector env. This is a deviation from CleanRL; the flag restores the original behaviour.

**D5. σ-head init: spec'd `std=0.01`, so init equivalence is approximate (≤8%), not
bit-exact.** With orthogonal gain 0.01 each weight row has norm 0.01 and `‖h‖ ≤ 8`,
so `|log σ(s) − log σ_init| ≤ 0.08` for every state (typically ~0.01). The unit test
asserts this bound and that the mean / critic / trunk weights are *bit-identical*
across modes for the same seed (shared modules are constructed first so they consume
the same RNG draws). `--logstd-head-init-std 0` gives exact equality if wanted. After
building the agent the torch RNG is re-seeded so both arms also draw the same
exploration-noise stream.

**D6. `Agent(obs_dim, act_dim, ...)` instead of `Agent(envs, ...)`.** Lets
`scripts/eval.py` and `analysis/demos.py` rebuild the policy from a checkpoint without
constructing a vector env.

**D7. log σ clamp is a hard `torch.clamp` to [−5, 2]** (as in the spec). Its gradient
is zero outside the range, so a σ that hits a bound can stick there; the fraction of
(state, dim) pairs at each bound is logged every update (`logstd_frac_at_min/max`) and
is part of the H2 diagnostics.

**D8. No action rescaling.** The policy outputs raw actions that the env wrapper clips
to the action bounds (Pendulum: ±2, others: ±1). log-prob is of the unclipped sample.

**D9. Gradient clipping is global** (one norm over actor + critic, as in CleanRL), so
the critic's gradient shares the 0.5 budget with the actor. `vf_coef=0.5` is kept
from CleanRL. Per-group *pre-clip* gradient norms (mean head, σ parameters, actor
trunk, critic) are logged as averages over the minibatches of each update.

**D10. Device = CPU, 1 torch thread per process.** The networks are tiny; the sweep
parallelises over runs instead.

## Protocol

**D11. Eval seeds.** Eval episode *i* always resets with seed `100000 + i`
(i = 0..19), for both the stochastic and deterministic episodes, every eval, every run,
both modes. Stochastic eval noise comes from a private `torch.Generator(seed 0)`, so
evaluation does not perturb the training RNG stream (unit-tested).

**D12. Eval schedule.** Evals run at step 0 and at the first update boundary at or
after each multiple of 25k steps, plus once after the last update. `evals.csv` stores
the true `global_step`; AUC is a trapezoid over those steps divided by the step range
(= time-averaged eval return).

**D13. Resuming = restarting incomplete runs from scratch.** `scripts/sweep.py` skips
runs whose `status.json` says `done` or `nan` and re-runs anything else from step 0
(deleting the partial directory). Runs are deterministic given the seed, and exact
mid-run resume would require serialising simulator state.

**D14. NaN policy.** Any non-finite policy output, loss or parameter stops the run and
writes `status: "nan"` with the step. The sweep never retries it; the analysis counts
them per mode.

**D15. "Learned" thresholds (H2: fraction of seeds that never learn)**, fixed before
the sweep: parking — stochastic-policy eval success rate ≥ 0.5 at any eval;
Pendulum — stochastic eval return ≥ −300; LunarLanderContinuous — stochastic eval
return ≥ 100. Defined in `envs/wrappers.py:ENV_SPECS`.

**D16. Goal-relative logging.** `info["goal_rel"]` / `info["goal_dist"]` are produced
at every env step by the wrappers. Every *eval* step is stored in the `.npz`
(this is what the σ analysis uses). For *training* steps only the per-rollout mean
distance and per-episode final distance are logged (storing 2M × 7 floats per run for
40 runs was judged not worth it; nothing in §5 needs it).
  - parking: `(achieved_goal − desired_goal) × scales` → metres, m/s; distance = ‖(dx, dy)‖.
  - Pendulum: `[θ, θ̇]`, θ wrapped to (−π, π], 0 = upright; distance = |θ|.
  - LunarLander: first 6 obs dims (pad is the origin); distance = ‖(x, y)‖.

**D17. Per-env sizes** (the only per-env settings; in `scripts/sweep.py`):
parking 8×256, 2M steps; Pendulum 4×512, 500k; third env 8×256, 1M.

## Analysis

**D18. Runs that stop early (NaN) stay in the statistics.** Their eval curve is padded
with the last recorded value before computing AUC / final scores, and they are counted
separately (`n_nan_runs`). Dropping them would bias the unstable arm upwards.

**D19. Scores.** AUC = trapezoid of the eval curve over steps ÷ step range (time-
averaged eval return); final = mean of the last 3 evals. Per env and per metric we
report the IQM over seeds with a 95% stratified-bootstrap CI from `rliable` (one task
per call, so "stratified" = resampling seeds), and IQM(state) − IQM(global) with a
bootstrap CI that resamples seeds independently in each arm. Envs are not aggregated
into one normalized score: H1 is a per-domain claim and returns are not commensurable.

**D20. σ analysis uses the final eval dump only, stochastic episodes** (`--last-k 1
--policy stoch`). Pooling several evals mixes policies from different training steps;
in the smoke test that made a *global* σ look state-dependent (std of log σ > 0).
The stochastic episodes are used because that is the state distribution the σ was
trained on.

**D21. "Fraction of variance of log σ explained by state".** For one policy σ is a
deterministic function of the state, so the literal fraction is 1 unless σ is constant.
We therefore report three things per seed and action dim: (a) the *amount* of state-
driven variation, `std_s(log σ)` and σ p95/p5 — ≈0 / ≈1 means the head collapsed to a
global value and H3 fails; (b) how much of it the *precision variables* explain: η²
(correlation ratio) per binned variable and R² of a joint linear fit; (c) pooled over
seeds, the within-seed (state) share of Var[log σ] vs. the between-seed share.

**D22. Spearman ρ is reported per seed; inference is across seeds.** Steps within an
episode are strongly autocorrelated, so per-step p-values are not used; we report the
median ρ and how many seeds have ρ > +0.1 / < −0.1.

**D23. Fine-control regions** (for the near/far σ ratio and the episode annotations),
fixed before the sweep: parking — within 3 m of the goal; Pendulum — |θ| < 0.3 rad;
LunarLander — height < 0.3 obs units. Defined in `analysis/sigma.py:SIGMA_SPECS`.

**D24. Demos** use the stochastic policy on reset seeds 200000+ (disjoint from the
eval seeds). The first 5 are rendered as-is; only if none is a failure are further
seeds scanned and the first failure swapped in for the 5th clip.

**D25. `pandas` pinned to 2.3.3**: `arch` 7.2.0 (a `rliable` dependency) fails to
import under pandas 3.x.

**D26. Pre-sweep pilot.** The 50k-step smoke test showed only a small Pendulum
improvement (≈ −1275 → −1150; the LR anneals to 0 over just 50k steps), so before any
full sweep a full-length Pendulum pilot (500k steps, seeds 1–2, both modes,
`runs_pilot/`) was run to confirm the shared config learns: global reached ≈ −200 on
both seeds, state ≈ −300 and ≈ −590. No hyperparameter was changed in response.

## User decisions (2026-11-22)

**U1. Third environment: `LunarLanderContinuous-v3`.**
**U2. Full sweep not launched yet** — the user will start it; nothing under `runs/` exists.
**U3. `state_separate` ablation: decide after the main results.**

## Additions (2026-11-22, second session): third domain, diagnostics, probe, cluster

**U4. Third core environment = Simglucose, not LunarLander.** The project goal names
Simglucose (insulin dosing for type-1 diabetes) as the third domain, so it replaces U1.
LunarLanderContinuous-v3 is kept as an optional fourth env (`bash cluster/submit.sh extra`)
because it was already implemented and costs little. *Confirm with the user.*

**D27. Simglucose is installed without dependency resolution**
(`pip install --no-deps -r requirements-simglucose.txt`). simglucose 0.2.11 declares
`gymnasium~=0.29.1` and `gym==0.9.4` for its own wrappers; a normal install silently
downgraded gymnasium 1.4 → 0.29 and broke `ppo.py` (no `AutoresetMode`). We only use its
simulator classes. `setuptools<81` is pinned because gym 0.9.4 / simglucose import
`pkg_resources`.

**D28. Simglucose env construction** (`envs/simglucose_env.py`). Built from simglucose's own
components with the package defaults: patient `adolescent#001` (the gym default),
`CGMSensor('Dexcom')` (3-min samples → one env step = 3 simulated minutes), `InsulinPump('Insulet')`,
`RandomScenario` (random daily meals, unannounced), random start hour, random initial BG,
basal-only control (bolus 0, as in the package's gym env), termination at BG < 10 or > 600
mg/dL. Every random component is re-drawn from the gymnasium `np_random` at each `reset`
(the package's `T1DSimGymnaisumEnv.reset` replays the same day forever). The patient's pandas
parameter Series is replaced by a plain-attribute copy (`_FastParams`): the ODE right-hand
side reads ~40 parameters per call and pandas attribute access dominated the run time;
this is 8.6× faster with bit-identical trajectories (unit-tested).

**D29. Simglucose action.** Policy action a ∈ [−1, 1] (clipped by the wrapper as in every env)
→ basal rate = (a + 1)/2 · 10 · u_b, where u_b is the patient's steady-state basal
(u2ss·BW/6000 U/min, 0.0139 U/min = 0.84 U/h for adolescent#001). The native action box,
0–30 U/min, is ~2000× u_b: any Gaussian policy over it overdoses. 10× u_b ≈ 8.4 U/h covers the
scenario's largest meal (80 g at CR 12 g/U ≈ 6.7 U) within about an hour on top of basal.
Consequences, kept on purpose because they are the precision structure under study: the box
midpoint (where the untrained mean sits) is 5× basal, which drives this patient into severe
hypoglycaemia within ~12 h; a constant 2× basal (a = −0.6) gives ~100% time in range.
Noise in a is *absolute* insulin noise (σ = 0.5 ↔ ±2.5× basal per 3-min step).

**D30. Simglucose reward = 1 − risk(BG)/100**, with risk the Magni/Kovatchev risk index of the
true BG (0 at 112.5 mg/dL, ~8 at 70 and 180, ~22 at 50 and 250, ~100 at 20 and 600). The
default `risk_diff` telescopes to risk(BG₀) − risk(BG_T) and carries no per-step signal. The
+1 alive bonus removes the incentive a pure penalty would create to end the episode (BG < 10
terminates); losing the remaining bonus is the termination penalty. Max return 480.

**D31. Simglucose observation, horizon and metrics.** Obs (42-d) = last 20 CGM readings (1 h),
last 20 insulin rates (× basal; history filled with 1.0 = basal steady state at reset),
sin/cos of the time of day. CGM alone is far from Markov (insulin acts for hours, meals are
unannounced). Time limit 480 steps = 24 h (truncation, value bootstrapped per D4).
`is_success` = consensus targets (Battelino et al. 2019): time in 70–180 mg/dL ≥ 70% and time
< 70 mg/dL < 4%. `crashed` = any BG < 54 mg/dL (level-2 hypoglycaemia) during the episode.
Episode TIR / TBR / TBR54 / TAR / mean BG / mean risk are recorded in every eval.
goal_rel = (CGM, CGM trend over 15 min, true BG, hour, mean insulin over 1 h);
goal_dist = Magni risk of the true BG. Fine-control region = CGM < 100 mg/dL (where an
overdose risks hypoglycaemia); "learned" threshold = stochastic success rate ≥ 0.5.

**D32. Simglucose log-action ablation** (`simglucose-adolescent1-logact-v0`): rate = u_b · 10^a
(0.1–10× basal, a = 0 → basal). Gaussian noise in a is then *multiplicative*: small in absolute
units whenever the mean dose is small (low BG). Hypothesis it tests: when the action
parameterisation already scales the noise with the required precision, a global σ inherits
state-dependent precision for free and σ(s) should matter less.

**D33. `--std-mode state_detach`** (optional ablation): the σ head reads the shared trunk
features through `detach()`, so σ gradients cannot reshape the trunk; same parameters and
init as `state`. With `state_separate` it separates "σ(s) is hard to learn" from "σ(s)
gradients interfere with the mean's features".

**D34. Read-only diagnostics** in `ppo.py` (`--grad-diagnostics`, default on), logged per update:
(a) gradient signal-to-noise ratio ‖ḡ‖²/Var[ḡ] of the actor loss over 16 minibatches at the
pre-update parameters (≈1 = pure noise, any dimension), per group: mean head, σ level
(global parameter / head bias), σ state-dependence (head weights), actor trunk;
(b) in `state` mode, cosine and norm ratio between the trunk gradient arriving via the σ head
and via the mean head (interference); (c) fraction of rollout states whose mean lies outside
the action box and fraction of sampled actions that are clipped (saturation), measured
against the env's own box (`eval_env.unwrapped.action_space`: gymnasium's `ClipAction` wrapper
advertises an unbounded space, so the first local runs logged 0 for (c); `sigma.py` computes
the same quantities correctly from the eval dumps). The diagnostics
use a private numpy Generator and `torch.autograd.grad` only; training is bit-identical with
and without them (checked on Pendulum, both modes, and unit-tested).

**D35. Noise-tolerance probe** (`analysis/probe.py`). Measures the premise "the appropriate
noise level differs across states" instead of assuming it. States are drawn from each run's
own stochastic episodes on reset seeds 300000+ (disjoint from eval 100000+ and demo 200000+),
stratified 50/50 inside / outside the fine-control region with inverse-inclusion weights.
At each state the env is deep-copied (exact for parking, Pendulum, Simglucose; Box2D cannot be
copied, so LunarLander is skipped) and the discounted (γ = 0.99) return over W steps is compared
between the mean policy and the mean policy plus N(0, (η·half-width)²) noise for the first H
steps, η ∈ {0.05, 0.1, 0.2, 0.4}, 6 common-random-number repeats; plus the policy's own σ, and
(2-d actions) noise on one action dimension at η = 0.2. H / W: parking 5 / 100 (1 s of noise,
rest of the episode), Pendulum 10 / 100, Simglucose 20 / 160 (1 h of noise, 8 h of effect).
c(s) = Σ L η² / Σ η⁴ (least-squares fit of L ≈ c η², the Q-curvature the σ gradient sees).
Heterogeneity index = AM/GM of c over on-policy states (c floored at its 5th percentile): by
AM–GM, the factor by which an ideal σ(s) ∝ c(s)^−½ with the same mean log σ would cut the
quadratic noise cost relative to the best global σ.

**D36. Cluster = CPU job arrays, one single-threaded run per task.** The actor (64-64) and critic
(256-256) are tiny and the simulators (highway-env, the simglucose ODE, Box2D) are CPU-bound
Python, so a GPU would idle while costing far more fairshare. Tasks are resumable under
requeue / resubmission (D13 semantics in `scripts/run_task.py`). Time limits are generous upper
bounds (parking 8 h, Simglucose 4 h, Pendulum 1 h).

**D37. parking-v0 episodes are up to 500 steps, and crashing can pay.** `duration=100` is in
*seconds*; the env clock advances 1/policy_frequency = 0.2 s per step, so the time limit is
500 policy steps (CLAUDE.md §2 said 100 steps; corrected). The episode terminates on success or
on a crash (wall), and a crash adds −5 once. With the per-step reward −√(Σ wᵢ|Δᵢ|) ≈ −0.5 at the
start, a 500-step time-out costs ≈ −150 to −250 while driving into the nearest wall costs
≈ −10 to −15, about the same as a slow successful park. Return is therefore not monotone in
parking competence, and "crash early" is a local optimum (the smoke test's −50 → −14 with 0%
success fits it). Consequences: the parking deep dive leads with success / crash / time-out
rates, not return; the default env stays the primary parking env (spec: keep the default
config). Optional variant `parking-nowalls-v0` = default config with `add_walls=False`, so an
episode can only end by parking or timing out; it isolates the precision question from the
crash shortcut (`cluster/submit.sh extra`). Probe window for parking: 150 steps (30 s).

**D38. Headless rendering: `SDL_VIDEODRIVER=offscreen`, never `dummy`.** highway-env's
`EnvViewer` sets `enabled=False` when `SDL_VIDEODRIVER=dummy` and then returns all-black frames
without any warning. The cluster scripts use the `offscreen` driver, and `analysis/demos.py`
re-enables the viewer if it finds it disabled.

**D39. Hyper-parameters are tuned per environment with a *paired* search (user request,
2026-11-22: "test hyperparameters that are good in each case").** One Optuna study per env
(`scripts/tune.py`; TPE, multivariate, constant-liar for parallel workers; median pruning on a
4-eval moving average after 30% of training). Every trial trains **both** arms with the **same**
sampled hyper-parameters on the same tuning seeds (101, 102) at the full training length, so
each trial is a paired comparison and the spec's invariant (arms differ only in `--std-mode`)
holds inside the search. Searched: learning rate (log-uniform 5e-5–3e-3), γ (env-specific grid
matched to the time scale: Pendulum {0.9 … 0.99}, parking {0.95 … 0.995}, Simglucose
{0.98 … 0.999}), GAE λ {0.8, 0.9, 0.95, 0.98, 0.99}, clip {0.1, 0.2, 0.25, 0.3}, entropy
{0, 1e-4, 1e-3, 1e-2}, epochs {3, 5, 10, 20}, minibatch size {64 … 512}, rollout length
(env-specific), initial σ {0.25, 0.5, 1.0}. Fixed: network sizes, grad clip, normalisation,
log σ clamp, training budget, eval protocol. The first trials are enqueued known-good configs:
the spec's shared default, Andrychowicz et al.'s λ = 0.9 / clip 0.25, and for Pendulum the RL
Baselines3 Zoo's tuned PPO (γ 0.9, lr 1e-3, 1024-step rollouts; without gSDE).
Objective = mean over the two arms of each arm's score; a run's score = mean over the last 25% of
its evaluations of: parking, stochastic success rate − 0.002 · final distance (m, capped at 30;
breaks ties while success is 0 without rewarding crashes); Pendulum and Simglucose, stochastic
return. Tuning uses 10 + 10 eval episodes on reset seeds 500000+, no eval dumps and no gradient
diagnostics; none of it is reused in the final comparison.
Outputs (`tune.py export` → `configs/tuned.json`): the best **shared** config per env (primary
final comparison, `$RUNS_ROOT`), and each arm's **own** best config (secondary comparison,
`$PERARM_ROOT`), both re-run with 10 fresh seeds and the standard eval seeds. The paired trials
also answer "is one arm *consistently* better?" across all configs tried, separately for the
uniform random start-up trials (`results/tuning_summary.md`). Tuning scores use 2 seeds and are
optimistically biased (winner's curse); only the fresh final runs are reported as results.
The variants `parking-nowalls-v0` and `simglucose-adolescent1-logact-v0` reuse the tuned config
of their base env. Ablations are run on top of the tuned shared config. The spec's untuned
default config (`submit.sh main` → `$DEFAULT_ROOT`) is kept as a robustness check: does tuning
change the conclusion?
