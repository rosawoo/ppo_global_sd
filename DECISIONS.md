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
