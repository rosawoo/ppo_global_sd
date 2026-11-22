# Global vs. state-dependent σ in PPO: report

**Status (2026-11-22, end of the second session).** All three domains of the project goal
are implemented and tested: highway-env `parking-v0` (the deep dive), `Pendulum-v1`, and
Simglucose (`simglucose-adolescent1-v0`, new). New tools explain *why* one parameterisation
wins: read-only optimisation diagnostics, a noise-tolerance probe that measures what action
noise costs in each state, and a parking deep dive. Results below come from **2 local CPU cores
and are preliminary**: Pendulum with 6 seeds per mode, plus 1-seed pilots of Simglucose and
parking. The full design (10 seeds × 3 domains, probes, ablations) is set up for FASRC Cannon:
see `cluster/README.md` (copy-paste commands).

Prior work: Andrychowicz et al. 2021, "What Matters in On-Policy Reinforcement Learning? A
Large-Scale Empirical Study" (arXiv:2006.05990), found no clear benefit of a state-dependent
σ on MuJoCo locomotion. This study tests the choice in tasks with a precision gradient over the
state space.

## Preliminary results (local, 2 CPU cores, spec's untuned default config)

These are **not** the study's results: Pendulum has 6 seeds per mode instead of 10, the other two
domains have 1-seed pilots shorter than the planned runs, and nothing is tuned yet (DECISIONS.md
D39). They are here because they already show the mechanisms the full study has to confirm or
refute. Files: `prelim/` (`prelim/make_prelim.sh` re-creates them).

### Pendulum-v1: 6 seeds per mode, full 500k steps (`prelim/pendulum/`)

**H1: global σ wins clearly.** Final stochastic eval return IQM −194 [95% CI −203, −186] for global
vs −422 [−614, −245] for σ(s); difference −228 [−421, −56], P(state > global) = 0.00 over bootstrap
draws. The two arms are identical until ~75k steps; then the global arm learns the swing-up and
plateaus at about −200 by 120k steps, while σ(s) improves slowly. The gap is the same for the
*deterministic* policy (−190 vs −415), so σ(s) hurts how well the mean is learned, not only the
evaluation noise.

**H2: σ(s) is much harder to optimise.** Spread of the final return across seeds: std 197 vs 10.
3 of 6 σ(s) seeds never reach −300 (0 of 6 global). Mean approx-KL 0.013 vs 0.005, clip fraction
0.13 vs 0.06, 8.5% of updates with KL > 0.02 vs 0%. Final mean log σ −3.4 (σ ≈ 0.03) vs −1.5
(σ ≈ 0.21). As σ(s) collapses, the mean head's gradient grows ~5× (the score ∂log π/∂μ = ε/σ
scales as 1/σ) and the update's KL / clip fraction stay 2–3× higher throughout, the signature of
mechanism 3.2 (σ sets the mean's PPO step). Trunk interference is small *late* in training (the
trunk gradient arriving through the σ head is orthogonal to the mean's, cosine ≈ 0, and ~6% of
its norm), but in the first ~50k steps, while the swing-up is being learned, the two are about
equally large. The `state_detach` ablation tests whether that early interference matters.

**H3: σ(s) does become state-dependent, in the "intuitive" direction.** 84% of the pooled variance
of log σ is within-run (state) variation. σ is ~4× larger far from upright than near it (median
far/near 4.4). Spearman ρ(σ, |θ|) is +0.66 (5/6 seeds > +0.1) and ρ(σ, |θ̇|) +0.67 (6/6).

**Why it still loses: the probe says the intuition is backwards for Pendulum.** In the global-σ
runs, noise is ~100× *cheaper* near the top than in the swing-up (precision requirement c: 0.2
vs 27, median over seeds). The upright controller corrects small errors within a few steps;
a mistimed push in the swing-up costs a whole extra swing. So the premise "holding it upright
needs precise corrections" is wrong *in terms of return*. σ(s) puts its noise exactly where it
is expensive: ρ(log σ, log c) = +0.34. Two further effects:
* *Saturation gives global σ precision for free where it matters.* During the swing-up the
  torque often saturates at ±2, so the executed noise of the global-σ policy there is only
  0.70× its value at the top.
* *Co-adaptation.* The σ(s) policies' controllers become fragile at the top: noise there
  costs them 26 per unit variance vs 0.2 for the global-σ controllers. A mean trained with
  almost no noise does not learn to recover from perturbations.

### Simglucose: 1-seed pilot, 400k of the planned 1M steps (`prelim/pilots/`)

Global σ is far ahead: stochastic return 445 vs 264, time in range 69% vs 38%, time below
70 mg/dL 29% vs 62%. Both policies are still overdosing at this point, so the comparison will
change with the full budget and tuning. The σ(s) run shows the opposite of the premise and the
self-trapping widening of mechanism 3.3:
* σ is ~5× **larger** below 100 mg/dL than above it (far/near 0.21), and the executed noise
  there is 2.4× larger.
* σ spans the whole clamp range (log σ 5th percentile at the −5 floor, 95th at the +2
  ceiling; 9% of states at the floor and 7% at the ceiling at the end). The hard clamp has no
  gradient beyond its bounds (D7), so those states stay there.
* σ grows with the insulin delivered over the last hour (Spearman +0.27; median σ 0.06 → 0.34
  from the lowest to the highest quartile). It widens exactly when an overdose is underway and
  the right action, suspending insulin, lies far below the mean.
* CGM itself explains little of it (η² ≈ 0.02). The σ head learned a nonlinear, idiosyncratic
  function rather than the clinical pattern.
* The probe cannot test the clinical premise yet, because both means are still far from good
  dosing. For the global-σ policy, extra insulin noise slightly *helps* on average (c < 0 in
  28% of states). For the σ(s) policy, noise around its mean costs ~200× more above
  100 mg/dL than below. Its *own* noise costs return mostly at low BG (0.24 vs 0.15 per state),
  where σ ballooned. Wait for the converged, tuned 10-seed runs.

### parking-v0: 1-seed pilot, 300k of the planned 2M steps (`prelim/pilots/`)

**Both arms fell into the crash shortcut (DECISIONS.md D37).** By 300k steps both policies
drive at full throttle toward the goal's row and hit the wall just past it at 12–13 m/s.
100% of stochastic eval episodes end in a crash; the σ(s) mean policy parks in 1.7% of
deterministic episodes and the global one in none. Return is about −11.5 for both, close to
what a slow successful park would earn. That is why return cannot separate parking competence
and success rate is the primary metric (`figures/parking_deepdive/trajectories.png`,
`approach_profile.png`).

* **Saturation makes the precision question moot so far.** The global-σ policy's mean
  acceleration is beyond the action bound in 91% of states (steering 26%). The executed
  acceleration noise is ~0 far from the goal for both arms. The fine manoeuvre where σ(s)
  should help never happens, because neither policy tries to stop.
* **σ(s) still learns an interpretable pattern:** steering noise shrinks with speed
  (Spearman −0.60). A steering error at 12 m/s throws the car much further sideways per step
  than at 3 m/s. σ is also somewhat smaller closer to the goal (ρ(σ, distance) +0.39 for
  acceleration, +0.20 for steering).
* The probe has nothing to measure yet: no sampled state is within 3 m of the spot, and noise
  is nearly free elsewhere (c ≈ 0.1–0.2, negative in 35–40% of states, since noise only changes
  where the car hits the wall).
* For the full study this means: tuning (its parking objective rewards success and, while
  success is 0, ending closer to the spot) has to find a config that escapes the shortcut, and
  `parking-nowalls-v0`, where a crash is impossible, is the cleaner test of the precision
  hypothesis. Give it its own tuning study (`cluster/README.md` §5).

### What this suggests for the full study

So far the global σ wins wherever it has been tested, and the mechanisms are the ones in the
section below: σ(s) follows *visitation and exploration demand*, not the cost of noise; global σ
gets precision for free through saturation and feedback; and σ(s)'s collapse or blow-up damages
learning of the mean. The real tests are still ahead: tuned configs (does a better learning
rate, entropy bonus or initial σ rescue σ(s)?), 10 seeds, full budgets, and parking, where
the precise actions are interior and the crash shortcut is absent in the no-walls variant.


## Why σ(s) should or should not help: the mechanics

### 1. What gradient trains σ in PPO

For one action dimension, π(a|s) = N(μ(s), σ(s)²) and ε = (a − μ)/σ. The score functions are

  ∂ log π / ∂μ = ε/σ  and  ∂ log π / ∂ log σ = ε² − 1.

Expanding the advantage of a sampled action to second order in the noise,
A(s, μ + σε) ≈ σε·∂ₐQ + ½σ²(ε² − 1)·∂²ₐQ, gives (using E[ε²] = 1, E[ε(ε² − 1)] = 0,
E[(ε² − 1)²] = 2) the expected policy-gradient directions

  g_μ ≈ ∂ₐQ(s, μ)          g_log σ ≈ σ² · ∂²ₐQ(s, μ).

So **σ is trained by the curvature of Q in the action**. It shrinks where Q is concave in a
(deviating from the mean costs return, so precision pays) and grows where Q is convex (for
example while the mean is still far from a good action). A state-dependent σ(s) can follow the
curvature state by state; a global σ follows its batch average. The probe
(`analysis/probe.py`) measures exactly this quantity: c(s), the return lost per unit of
injected noise variance in state s.

### 2. Why a global σ is often good enough

The hypothesis needs more than "precision matters in some states". A global-σ policy has
several ways to be precise where it matters without a state-dependent σ:

1. **Saturation at the action bounds.** The env clips the sampled action to the box. Where the
   right action sits at a bound (full throttle, full steering, insulin suspended), the mean
   can move past the bound and the *executed* action becomes nearly deterministic whatever σ
   is. Precision at a bound is free. `sigma.py` plots the executed-action noise (σ after
   clipping), and `ppo.py` logs the fraction of states whose mean lies outside the box.
2. **The action parameterisation.** If noise scales with the action (Simglucose's log-scale
   insulin variant: rate = basal · 10^a), small doses automatically get small absolute noise.
3. **Feedback.** Noise only costs return if its effects persist. A good closed-loop
   controller corrects small perturbations within a few steps. Near Pendulum's upright
   equilibrium, an error of 0.05 rad costs a few θ² ≈ 0.0025 penalties before the controller
   removes it.
4. **Slow dynamics low-pass the noise.** Independent per-step noise averages out when the
   state integrates the action over many steps. Insulin acts over hours, so ±2.5× basal of
   noise on 3-min steps is ±0.6× basal over an hour.
5. **A noise-robust mean.** μ and σ are learned jointly. With a large global σ the best mean
   can simply be more conservative (approach the parking spot slowly) so that the same noise
   does less harm.

### 3. Why σ(s) is harder to optimise

1. **Its signal is second-order and gets split across states.** The per-sample estimator of
   g_log σ is A·(ε² − 1). Its expectation is O(σ²) while its variance is O(Var A), against
   O(1) signal for the mean. A global σ averages this weak signal over the whole batch. A σ
   head has to resolve it region by region, so each region learns from its own samples
   only. Rarely visited regions (the last metres before the spot) get little signal, and
   frequently visited ones get a lot. σ(s) therefore tends to track
   **visitation × curvature**, not curvature alone. The `gsnr_sigma_weight` diagnostic measures
   the signal-to-noise ratio of the state-dependent part (≈ 1 means pure noise).
2. **σ is PPO's step size for the mean.** For a mean shift Δμ, the log probability ratio of a
   sample is ≈ εΔμ/σ, so the ±0.2 ratio clip limits each update to Δμ ≲ 0.2σ. Where σ(s)
   collapses, μ(s) freezes. A global σ collapses (or not) everywhere at once. A σ head can
   collapse early in frequently visited states and lock in whatever the mean does there,
   including a bad local optimum.
3. **σ(s) also tracks where the mean is wrong, not only where precision is needed.** The
   curvature formula assumes the mean is already close to a good action. Early in training,
   in states where the mean is badly wrong and the better actions sit in one tail of the
   Gaussian, the σ gradient is positive and σ(s) widens *locally*. Those are often the
   states where mistakes are most costly, which is exactly where the hypothesis wants σ to be
   small. Widening can also trap itself: once σ is large compared with the action box, most
   samples are clipped to a bound, the executed action hardly depends on μ, and the mean loses
   the signal it needs to improve. A global σ cannot widen in one region only. (The
   Simglucose pilot shows this at low BG; see the results above.)
4. **Interference through the shared trunk.** The σ head's gradient also reshapes the
   features the mean uses. `trunk_cos_sigma_vs_mean` and `trunk_gnorm_sigma_over_mean`
   measure it. The `state_detach` and `state_separate` ablations remove it.
5. **Evaluation decides what σ is worth.** With deterministic (mean-action) evaluation, σ
   matters only through learning (exploration, gradient quality, step size). With stochastic
   evaluation, σ also costs return directly. Both are reported, and det − stoch is plotted
   as the "noise cost".

### 4. What this predicts for the three domains

| domain | where noise is expensive | what a global σ gets for free | prediction |
|---|---|---|---|
| Pendulum | to be measured: the premise is "at the top", but feedback corrects noise there, and a mistimed push in the swing-up costs a whole extra swing | swing-up torques saturate (±2); the upright controller corrects noise | σ(s) shrinks at the top because that is where most samples are, not necessarily where noise is most expensive; σ(s) collapse can freeze an imperfect swing-up. *Prelim (6 seeds): as predicted; noise is ~100× cheaper at the top* |
| parking | the last metres: braking and gentle steering, both *interior* actions, and over-shooting hits the wall behind the spot | far away, full throttle / full steering saturate | the largest potential gain from σ(s), but the default env's crash shortcut (D37) may dominate both arms |
| Simglucose | low BG: extra insulin → hypoglycaemia | the right action at low BG is "suspend", which is the lower bound (saturation); insulin kinetics low-pass the noise | little difference between arms; with the log-scale action map, even less. *Prelim (1 seed, 400k): global far ahead, because σ(s) widened at low BG (mechanism 3.3), which this table did not anticipate* |

## Parking deep dive: how to read the figures

* `figures/parking_deepdive/outcomes_over_training.png`: success, crash and time-out rates for
  the stochastic and the mean policy. **Read this before any return curve.** In the default
  env, return is not monotone in competence: a 500-step time-out costs ≈ −150 to −250, while
  driving into the nearest wall costs ≈ −10 to −15, about the same as a slow successful park
  (DECISIONS.md D37).
* `trajectories.png`: paths in the goal frame (spot at the origin, its heading along +x),
  coloured by σ. A useful σ(s) shows as dark (small σ) near the spot and light far away.
* `approach_profile.png`: speed, |steering mean|, σ and *executed* noise against distance to
  the spot. It shows whether a global-σ policy gets precision from saturation (executed noise
  below σ) or from slowing down.
* `figures/parking-v0_probe_cost.png` / `_probe_sigma_vs_c.png`: whether noise really costs
  more near the spot, separately for steering and acceleration, and whether σ(s) follows it.
* `figures/parking-v0_sigma_over_training.png`: when σ starts to differ between near and far.
  Compare with the success curve. If σ(s) differentiates only after the policy can park, it
  is a consequence of competence, not its cause.

## What the cluster experiments test

| stage (`cluster/submit.sh …`) | runs | question |
|---|---|---|
| `all` ① tuning | ≈ 136 paired trials (4 runs each) | good hyper-parameters per env (DECISIONS.md D39); across *every* config tried, is one arm consistently better? which hyper-parameters matter for which arm? |
| `all` ② final, shared tuned config | 60 + 60 probes | H1–H4 in the three domains at the best shared config (arms differ only in σ); is the precision requirement actually uneven (probe), and does σ(s) follow it? |
| `all` ② final, per-arm tuned configs | 60 | does the verdict survive giving each arm its own best config? |
| `main` (spec's untuned default) | 60 | does the verdict survive *not* tuning? |
| `extra`: `parking-nowalls-v0` | 20 | parking precision without the crash shortcut (D37) |
| `extra`: `simglucose-adolescent1-logact-v0` | 20 | does a multiplicative action map make σ(s) unnecessary (D32)? |
| `extra`: `LunarLanderContinuous-v3` | 20 | the previous third domain, kept as a cheap fourth |
| `ablations`: `state_detach`, `state_separate` | 60 | is σ(s)'s cost interference in the shared trunk, or the σ head itself? |
| `ablations`: entropy 0.01 | 60 | does an entropy bonus stop σ(s) collapse (and help one arm more)? |
| `ablations`: GAE λ = 0.9 on parking | 20 | Andrychowicz et al.'s recommended λ |
| `ablations`: clip 0.25 | 60 | a looser clip lets the mean move further per update when σ(s) is small (mechanism 3.2) |

Decision rules, fixed in advance:
* **H1:** a mode wins on a metric only if the 95% bootstrap CI of IQM(state) − IQM(global)
  excludes 0 at the tuned shared config (`results/summary.md`). Otherwise the result is "no
  detectable difference with 10 seeds". For parking, success rate is the primary metric.
  "Consistently" is judged from the tuning trials (fraction of configs where each arm wins,
  Wilcoxon test over configs, `results/tuning_summary.md`) together with the per-arm and
  untuned comparisons.
* **H2:** seed spread (IQR, std), never-learned fraction, NaN count, σ at the floor, gradient
  SNR, and trunk interference (`results/summary.md`, H2 table).
* **H3:** Spearman ρ(σ, precision variable) per seed, σ far/near ratio, and the probe's
  ρ(log σ, log c) (`results/sigma_summary.md`, `results/probe_summary.md`).
* **H4:** 3–5 plain-English statements checked against the figures (template below).

## H4: interpretability statements (to be filled from the full results)

Template, each one marked *supported*, *refuted* or *unclear* with the figure that decides it:
1. "In parking, the σ(s) policy cuts steering noise by ×k within 3 m of the spot while keeping
   acceleration noise roughly constant." (`parking-v0_sigma_binned.png`)
2. "The global-σ policy is just as precise in the last metre, because it slows down / saturates
   steering." (`approach_profile.png`)
3. "In Pendulum, σ(s) is smallest at the top although noise costs more during the swing-up."
   (`Pendulum-v1_probe_cost.png`)
4. "In Simglucose, both policies stop insulin at low BG; the executed noise there is ~0 for
   both." (`simglucose-adolescent1-v0_effective_noise.png`)

## Compute

| env | steps / run | this VM (2.1 GHz Xeon, 1 core, contended) | notes |
|---|---|---|---|
| parking-v0 | 2M | ≈ 155–230 steps/s | ≈ 300–450 steps/s expected on Cannon cores; eval cost depends on episode length (≤ 500 steps) |
| Pendulum-v1 | 500k | ≈ 1,300 steps/s, ≈ 7 min / run | |
| Simglucose | 1M | ≈ 440–650 steps/s | evaluation (40 × 480-step episodes every 25k steps) is ~40% of the run time |

Earlier sessions: smoke test (1 seed, 50k steps, all envs) and a 2-seed Pendulum pilot on an
Apple-silicon laptop; see `DECISIONS.md` D26. Estimate for Cannon: tuning ≈ 350–500 core-hours, the final comparison ≈ 160, every optional
stage together ≈ 435 more (see `cluster/README.md`).
