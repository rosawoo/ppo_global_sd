# Project: Global vs. state-dependent σ in PPO (continuous control)

You are implementing and running a controlled empirical study. Treat this file as the
spec. When something is ambiguous, pick the option that keeps the two policy classes
*identical except for how σ is computed*, and log the decision in `DECISIONS.md`.

## 1. Research question

PPO with a diagonal Gaussian policy usually learns the mean μ(s) with a network and the
log-std as a **global** parameter vector (CleanRL, SB3, Baselines all do this). The
alternative is a **state-dependent** head: the network outputs log σ(s).

We test, in three domains where the required action precision varies across states:

- H1: state-dependent σ outperforms global σ in tasks where the appropriate noise level
  differs across states.
- H2: state-dependent σ is harder to optimize (more variance across seeds, σ collapse,
  unstable KL / clip-fraction).
- H3: the state-dependent policy actually *learns* σ that varies with the state in a
  way that tracks task precision requirements (e.g. small σ near the goal).
- H4: the learned σ(s) is interpretable to a human when visualized over the state space.

Prior work: Andrychowicz et al. 2021 ("What Matters in On-Policy RL", arXiv:2006.05990)
found *no clear benefit* of per-state σ on MuJoCo locomotion (Hopper, Walker2d,
HalfCheetah, Ant, Humanoid). Our contribution is testing it outside locomotion, in
tasks with a precision gradient over the state space. Cite them in the writeup.

## 2. Environments

Deep dive: **highway-env `parking-v0`** (vehicle control, goal-conditioned, 2-D action
[acceleration, steering] in [-1, 1]). Imprecise actions are fine far from the spot;
precise ones are needed close to it. Plus two other domains:

| id | domain | why precision varies with state |
|---|---|---|
| `parking-v0` (highway-env) | vehicle | fine control only near the parking spot |
| `Pendulum-v1` (gymnasium) | classic control | coarse during swing-up, fine at the top |
| THIRD ENV — ask the user before running the full sweep. Default to `LunarLanderContinuous-v3` (Box2D; precise at touchdown) if no answer. Alternative: `PandaReach-v3` (panda-gym). |

Notes on `parking-v0` (do not skip):
- `pip install highway-env`; `import highway_env; gym.register_envs(highway_env)`.
- Observation is a `Dict` with `observation`, `achieved_goal`, `desired_goal`
  (`KinematicsGoal`, features `[x, y, vx, vy, cos_h, sin_h]`, scales
  `[100, 100, 5, 5, 1, 1]`). Write a wrapper that concatenates
  `observation` and `desired_goal` into a flat Box (do **not** include
  `achieved_goal`; it duplicates `observation`). Normalize observations with a
  running mean/std (CleanRL `NormalizeObservation`-style).
- Reward is a dense negative weighted p-norm of the goal error
  (`reward_weights=[1, 0.3, 0, 0, 0.02, 0.02]`, p=0.5), `success_goal_reward=0.12`,
  `collision_reward=-5` (walls are on), `policy_frequency=5`, `duration=100` steps.
  Episodes terminate on crash or success. PPO is **not** the standard algorithm here
  (official examples use HER + SAC/TQC); it is expected to learn because the reward is
  dense, but success rate may be modest. Report **both** return and `info["is_success"]`.
- Log the goal-relative state at every step (`achieved_goal - desired_goal`, and the
  Euclidean distance) — this is the x-axis for the σ-vs-state analysis.
- Keep the default config. Do not tune the env to make PPO look good.

## 3. Base implementation

Start from **CleanRL `ppo_continuous_action.py`** (vwxyzjn/cleanrl, master). Copy it
into `ppo.py` and modify it. Do not use SB3 (its only "state-dependent" option is gSDE,
which is a different mechanism). Keep CleanRL's structure: single file, `Args`
dataclass via `tyro`, TensorBoard logging, vectorized `gym.vector.SyncVectorEnv`.

### 3.1 The two policy classes

```python
class Agent(nn.Module):
    def __init__(self, envs, std_mode: str, log_std_init: float):
        # std_mode in {"global", "state"}
        self.critic = MLP(obs_dim -> 256 -> 256 -> 1, tanh)          # separate, wider
        self.actor_trunk = MLP(obs_dim -> 64 -> 64, tanh)
        self.actor_mean = layer_init(nn.Linear(64, act_dim), std=0.01)  # 100x smaller
        if std_mode == "global":
            self.actor_logstd = nn.Parameter(torch.full((1, act_dim), log_std_init))
        else:
            self.actor_logstd_head = layer_init(nn.Linear(64, act_dim), std=0.01)
            # bias initialised so that logstd_head(x) ≈ log_std_init at init
            nn.init.constant_(self.actor_logstd_head.bias, log_std_init)

    def dist(self, x):
        h = self.actor_trunk(x)
        mean = self.actor_mean(h)
        if self.std_mode == "global":
            logstd = self.actor_logstd.expand_as(mean)
        else:
            logstd = self.actor_logstd_head(h)
        logstd = torch.clamp(logstd, LOGSTD_MIN, LOGSTD_MAX)   # -5, 2
        return Normal(mean, logstd.exp()), logstd
```

Invariants that make the comparison fair:
- Same trunk, same mean head, same critic, same init, same `log_std_init` (default
  `log(0.5) ≈ -0.69`, following Andrychowicz's finding that initial σ≈0.5 is best).
  At step 0 both policies must produce (numerically) the same σ for every state;
  assert this in a unit test.
- Same clamp on log σ for both (prevents the exp-NaN failure Andrychowicz reported).
- Same optimizer, LR, and parameter count up to the one extra `64×act_dim` head.
- Entropy bonus is the same for both (default 0.0; sweep 0.0 / 0.01 as an ablation,
  because a σ(s) head can collapse faster without it — this is part of H2).

Optional ablation (only after the main sweep): `std_mode="state_separate"` — a
separate 64-64 trunk for log σ, to test whether gradient interference through the
shared trunk explains any difference.

### 3.2 PPO configuration (Andrychowicz-informed defaults, one config for all envs)

| hp | value | source |
|---|---|---|
| clip ε | 0.2 (try 0.25) | And. §3.1 |
| GAE λ | 0.95 (And. recommend 0.9; sweep both on parking) | And. §3.4 |
| γ | 0.99 | And. §3.6 |
| value loss | MSE, **no** value clipping, **no** Huber | And. §3.4 |
| optimizer | Adam, lr 3e-4, linear decay to 0, β1=0.9 | And. §3.7 |
| epochs / minibatches | 10 / 32 | CleanRL default |
| obs norm | running mean/std, clip ±10 | And. §3.3 |
| reward norm | on (CleanRL `NormalizeReward`) | CleanRL |
| adv norm | per-minibatch | CleanRL |
| grad clip | 0.5 | And. §3.3 |
| action handling | sample, clip to action bounds in the wrapper (log-prob on the unclipped sample) | CleanRL |
| activations | tanh | And. §3.2 |
| num_envs × num_steps | 8 × 256 (parking), 4 × 512 (pendulum) | tune |
| total steps | parking 2M, pendulum 500k, third env 1M | |

Put every hp in `Args`; nothing hard-coded. The only CLI difference between the two
arms of the experiment must be `--std-mode`.

## 4. Experimental protocol

- Seeds: **10 per (env, std_mode)**. Seed everything (torch, numpy, env reset seed,
  `torch.use_deterministic_algorithms` where possible).
- Evaluation: every 25k steps, run 20 episodes with the **stochastic** policy and 20
  with the **deterministic** (mean) policy, on a separate eval env with a fixed seed set.
  Record return, episode length, success (parking), and for every eval step the tuple
  `(obs, goal-relative-state, μ, σ, action)` to a `.npz` — this is the raw material for
  the σ analysis.
- Primary metric: area under the eval-return curve (as Andrychowicz) **and** final
  return (mean of last 3 evals). Report **IQM with 95% stratified-bootstrap CIs** using
  `rliable`; also show per-seed curves. Do not report mean±std only.
- Optimization-difficulty diagnostics (H2), logged every update: mean/min log σ,
  policy entropy, approx KL, clip fraction, explained variance, gradient norm of the σ
  parameters vs. the mean parameters, and fraction of seeds that never exceed a
  "learned" threshold.
- Run order: smoke test (1 seed, 50k steps, both modes, all envs) → fix bugs →
  full sweep. Use `scripts/sweep.py` that writes one run per `(env, mode, seed)` under
  `runs/{env}/{mode}/seed{N}/` and is resumable.

## 5. σ-vs-state analysis (H3) and interpretability (H4)

For every trained state-dependent policy (and global as the flat reference line):

1. **Binned σ curves.** Bin eval states by the task's precision variable and plot
   mean σ per action dim with CIs across seeds:
   - parking: distance to goal (m) and |heading error|;
   - pendulum: |θ| (angle from upright) and |θ̇|;
   - third env: height above ground / distance to target.
   Also report Spearman ρ between σ and the precision variable, per seed.
2. **Variance decomposition.** Fraction of variance of log σ explained by state
   (across the eval dataset) — if ≈0, the head collapsed to a global value and H3 fails
   even if H1 holds.
3. **Heatmaps.** parking: σ_accel and σ_steer over the (x, y) plane relative to the
   goal (goal at origin, aligned heading), aggregated over eval rollouts; pendulum:
   σ over (θ, θ̇). One figure per seed plus a seed-averaged figure.
4. **σ over an episode.** Overlay σ(t) on a few trajectories; annotate the moment the
   car enters the "fine control" region.
5. **Demos.** Render 5 episodes per env/mode with `render_mode="rgb_array"`, draw the
   current σ as an ellipse/bar in the corner of each frame (and the mean action), and
   save as MP4/GIF under `demos/`. Include at least one failure case.
6. **Interpretability check.** Write 3–5 plain-English statements the plots support
   or refute, e.g. "the policy becomes 4× more precise in steering within 3 m of the
   spot but keeps acceleration noise constant". Be honest when it isn't interpretable.

## 6. Repository layout

```
ppo.py                  # single-file PPO, CleanRL-derived, --std-mode {global,state}
envs/wrappers.py        # parking Dict->Box wrapper, goal-relative logging, action clip
scripts/sweep.py        # launches / resumes the (env × mode × seed) grid
scripts/eval.py         # periodic eval + .npz dump
analysis/curves.py      # rliable IQM plots, per-seed curves
analysis/sigma.py       # binned σ, variance decomposition, heatmaps, Spearman
analysis/demos.py       # rendered rollouts with σ overlay
tests/test_policy.py    # init equivalence, shapes, log-prob correctness, clamp
DECISIONS.md            # every non-obvious choice, with the reason
REPORT.md               # results, filled in as runs finish
```

## 7. Working rules

- Python 3.10+, `torch`, `gymnasium>=1.0`, `highway-env`, `tyro`, `rliable`,
  `tensorboard`, `imageio[ffmpeg]`. Pin versions in `requirements.txt`.
- Run the smoke test and the unit tests before any long job. Never launch the full
  sweep until the smoke test shows learning in both modes on Pendulum.
- Before changing a hyperparameter, write the reason in `DECISIONS.md`. Never change
  an hp for one std_mode only.
- Commit after each milestone (skeleton runs; smoke test passes; sweep launched;
  analysis done). Small, descriptive commits.
- If a run NaNs, do not silently restart it with a different seed — record it; NaN
  frequency is itself an H2 result.
- Log compute (wall-clock per run, device) in `REPORT.md`.
- Ask the user, not yourself, about: the third environment, the total compute budget,
  and whether to run the `state_separate` ablation.
