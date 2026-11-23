"""Noise-tolerance probe: what does action noise actually cost in each state?

This measures the premise of the study directly instead of assuming it. For a trained run
we sample states from its own stochastic episodes (fixed reset seeds), clone the
environment at each state (copy.deepcopy; exact for parking, Pendulum and Simglucose) and
compare discounted returns over the next W steps:

  V0(s)       follow the run's mean policy mu
  Q_eta(s)    add N(0, (eta * half-width)^2) noise to the executed action for the first H
              steps, then follow mu
  Q_own(s)    add the policy's OWN noise sigma(s_t) for the first H steps, then follow mu
  Q_dim(s)    (multi-dim actions) noise at one reference level on a single action dim,
              to tell which dimension needs the precision (parking: steering vs. acceleration)

with common random numbers across noise levels. probe_report.py turns these into
  L(s, eta) = V0 - Q_eta        the cost of noise at s
  c(s)      = sum L eta^2 / sum eta^4   local precision requirement (L ~ c eta^2 for small eta;
                                         ~ curvature of Q(s, .) accumulated over H steps)
  L_own(s)  = V0 - Q_own        what the policy's own sigma costs at s
and compares the learned sigma(s) with c(s).

States are drawn in two strata (inside / outside the env's fine-control region, see
sigma.py SIGMA_SPECS) so that the rarely-visited precise region is well covered; each state
stores the inverse inclusion weight, so on-policy averages can still be computed.

    python analysis/probe.py --runs-root runs --envs parking-v0 --seeds 1 2 3 --workers 4
    python analysis/probe_report.py --runs-root runs

Writes {run_dir}/probe.npz. LunarLander is skipped (Box2D worlds cannot be deep-copied).
"""
from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import tyro

# per-env defaults: H = steps of injected noise, W = return window (steps)
PROBE_DEFAULTS = {
    "parking-v0": {"horizon": 5, "window": 150},  # 1 s of noise at 5 Hz; 30 s window (episodes: <= 500 steps = 100 s)
    "parking-nowalls-v0": {"horizon": 5, "window": 150},
    "Pendulum-v1": {"horizon": 10, "window": 100},  # 0.5 s of noise at 20 Hz; 5 s window
    "simglucose-adolescent1-v0": {"horizon": 20, "window": 160},  # 1 h of noise; 8 h window (insulin acts for hours)
    "simglucose-adolescent1-logact-v0": {"horizon": 20, "window": 160},
}


@dataclass
class ProbeArgs:
    runs_root: str = "runs"
    envs: list[str] = field(default_factory=lambda: ["parking-v0", "Pendulum-v1", "simglucose-adolescent1-v0"])
    modes: list[str] = field(default_factory=lambda: ["global", "state"])
    seeds: list[int] = field(default_factory=lambda: list(range(1, 11)))
    n_states: int = 120
    """probe states per run (half inside the fine-control region when available)"""
    episodes: int = 20
    """stochastic episodes (reset seeds seed_base + i) the states are sampled from"""
    seed_base: int = 300000
    """first reset seed of the state-sampling episodes (disjoint from eval and demo seeds)"""
    noise_levels: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    """injected noise std, in units of the action box half-width"""
    repeats: int = 6
    """noise draws per level (common random numbers across levels)"""
    dim_level: float = 0.2
    """reference level of the per-dimension probes (multi-dim actions only)"""
    horizon: int = 0
    """steps of injected noise (0 = per-env default, PROBE_DEFAULTS)"""
    window: int = 0
    """return window in steps (0 = per-env default)"""
    gamma: float = 0.99
    workers: int = 1
    overwrite: bool = False
    run_dir: str = ""
    """probe only this run directory (used by the Slurm array; ignores envs/modes/seeds)"""
    emit_tasks: str = ""
    """write one JSON task per expected run directory (finished or not) to this file and exit;
    the array task fails cleanly if its run has no checkpoint yet"""
    extra: str = ""
    """with --emit-tasks: extra probe.py arguments stored in every task"""
    only_finished: bool = False
    """with --emit-tasks: only runs that already have a checkpoint"""


def _policy(agent, normalizer, obs_batch):
    x = torch.as_tensor(normalizer.obs(np.asarray(obs_batch), update=False))
    with torch.no_grad():
        d, _ = agent.dist(x)
    return d.mean.numpy().astype(np.float64), d.stddev.numpy().astype(np.float64)


def sample_episodes(agent, normalizer, env, a: ProbeArgs):
    """Stochastic episodes with per-episode noise generators (so each can be replayed exactly)."""
    from sigma import SIGMA_SPECS

    spec = SIGMA_SPECS[a._env]
    fine_fn = spec.variables[spec.fine_var][0]
    records = []  # (episode, t, in_fine)
    for i in range(a.episodes):
        rng = np.random.default_rng(a.seed_base + i)
        obs, info = env.reset(seed=a.seed_base + i)
        t, done, rel, dist, obs_hist = 0, False, [], [], []
        while not done:
            mu, sd = _policy(agent, normalizer, obs[None])
            rel.append(info["goal_rel"])
            dist.append(info["goal_dist"])
            obs_hist.append(obs)
            act = mu[0] + sd[0] * rng.standard_normal(mu.shape[1])
            obs, _, term, trunc, info = env.step(act.astype(np.float32))
            done, t = term or trunc, t + 1
        rel = np.asarray(rel)
        fine = fine_fn({"goal_rel": rel, "goal_dist": np.asarray(dist), "obs": np.asarray(obs_hist)}) < spec.fine_threshold
        records += [(i, k, bool(fine[k])) for k in range(len(rel))]
    return records


def choose_states(records, n, rng):
    """Up to n/2 states from each stratum; returns [(episode, t, stratum, weight)]."""
    out = []
    strata = [[r for r in records if r[2]], [r for r in records if not r[2]]]
    quota = [min(len(strata[0]), n // 2)]
    quota.append(min(len(strata[1]), n - quota[0]))
    for s, (pop, q) in enumerate(zip(strata, quota)):
        if q == 0:
            continue
        for j in rng.choice(len(pop), size=q, replace=False):
            ep, t, _ = pop[j]
            out.append((ep, t, s, len(pop) / q))
    return sorted(out)


def snapshots(agent, normalizer, env, chosen, a: ProbeArgs):
    """Replay the chosen episodes and deep-copy the env at the chosen steps."""
    by_ep = {}
    for ep, t, s, w in chosen:
        by_ep.setdefault(ep, []).append((t, s, w))
    snaps = []
    for ep, targets in sorted(by_ep.items()):
        want = {t: (s, w) for t, s, w in targets}
        rng = np.random.default_rng(a.seed_base + ep)
        obs, info = env.reset(seed=a.seed_base + ep)
        t, done = 0, False
        while not done and t <= max(want):
            mu, sd = _policy(agent, normalizer, obs[None])
            if t in want:
                snaps.append({"env": copy.deepcopy(env), "obs": obs.copy(), "goal_rel": np.asarray(info["goal_rel"]),
                              "goal_dist": float(info["goal_dist"]), "mean": mu[0], "std": sd[0],
                              "stratum": want[t][0], "weight": want[t][1], "episode": ep, "t": t})
            act = mu[0] + sd[0] * rng.standard_normal(mu.shape[1])
            obs, _, term, trunc, info = env.step(act.astype(np.float32))
            done, t = term or trunc, t + 1
    return snaps


def continuation_returns(agent, normalizer, snap, conds, z, half_width, H, W, gamma):
    """Lock-step rollouts from clones of one snapshot. conds[k] = (kind, eta, repeat, dim) with
    kind in {'det', 'eta', 'own', 'dim'}; z[repeat, t] = standard-normal noise shared across levels."""
    n = len(conds)
    envs = [copy.deepcopy(snap["env"]) for _ in range(n)]
    obs = np.stack([snap["obs"]] * n)
    active, ret, disc = np.ones(n, bool), np.zeros(n), np.ones(n)
    for t in range(W):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        mu, sd = _policy(agent, normalizer, obs[idx])
        for j, k in enumerate(idx):
            kind, eta, rep, dim = conds[k]
            act = mu[j].copy()
            if t < H and kind == "eta":
                act += eta * half_width * z[rep, t]
            elif t < H and kind == "own":
                act += sd[j] * z[rep, t]
            elif t < H and kind == "dim":
                act[dim] += eta * half_width[dim] * z[rep, t, dim]
            o, r, term, trunc, _ = envs[k].step(act.astype(np.float32))
            ret[k] += disc[k] * r
            disc[k] *= gamma
            obs[k] = o
            if term or trunc:
                active[k] = False
    return ret


def probe_run(run_dir: str, a: ProbeArgs):
    from envs.wrappers import make_env
    from scripts.eval import load_run

    out_path = os.path.join(run_dir, "probe.npz")
    if os.path.exists(out_path) and not a.overwrite:
        return f"{run_dir}: exists, skipped"
    torch.set_num_threads(1)
    t0 = time.time()
    agent, normalizer, args = load_run(run_dir)
    env = make_env(args["env_id"])
    lo, hi = env.unwrapped.action_space.low, env.unwrapped.action_space.high
    half_width = (hi - lo) / 2.0
    defaults = PROBE_DEFAULTS[args["env_id"]]
    H, W = a.horizon or defaults["horizon"], a.window or defaults["window"]
    rng = np.random.default_rng(12345 + args["seed"])

    records = sample_episodes(agent, normalizer, env, a)
    chosen = choose_states(records, a.n_states, rng)
    snaps = snapshots(agent, normalizer, env, chosen, a)

    levels = list(a.noise_levels)
    act_dim = len(half_width)
    dims = list(range(act_dim)) if act_dim > 1 else []
    conds = [("det", 0.0, 0, -1)] + [("eta", e, r, -1) for e in levels for r in range(a.repeats)] \
        + [("own", 0.0, r, -1) for r in range(a.repeats)] \
        + [("dim", a.dim_level, r, d) for d in dims for r in range(a.repeats)]
    n_eta, n_own = len(levels) * a.repeats, a.repeats
    V0, Qeta, Qown, Qdim = [], [], [], []
    for i, snap in enumerate(snaps):
        z = np.random.default_rng(777 + i).standard_normal((a.repeats, H, act_dim))
        ret = continuation_returns(agent, normalizer, snap, conds, z, half_width, H, W, a.gamma)
        V0.append(ret[0])
        Qeta.append(ret[1:1 + n_eta].reshape(len(levels), a.repeats))
        Qown.append(ret[1 + n_eta:1 + n_eta + n_own])
        Qdim.append(ret[1 + n_eta + n_own:].reshape(len(dims), a.repeats))
    keys = ("goal_rel", "goal_dist", "mean", "std", "stratum", "weight", "episode", "t", "obs")
    np.savez_compressed(
        out_path, V0=np.asarray(V0), Q_eta=np.asarray(Qeta), Q_own=np.asarray(Qown), Q_dim=np.asarray(Qdim),
        dim_level=a.dim_level, noise_levels=np.asarray(levels), half_width=half_width, horizon=H, window=W,
        gamma=a.gamma,
        n_records=len(records), n_fine_records=sum(r[2] for r in records),
        **{k: np.asarray([s[k] for s in snaps]) for k in keys})
    env.close()
    return f"{run_dir}: {len(snaps)} states in {(time.time() - t0) / 60:.1f} min"


def _job(args):
    run_dir, a = args
    try:
        return probe_run(run_dir, a)
    except Exception as e:  # keep the other runs going
        return f"{run_dir}: FAILED {type(e).__name__}: {e}"


def main(a: ProbeArgs):
    if a.run_dir:
        env = a.run_dir.rstrip("/").split(os.sep)[-3]
        if env not in PROBE_DEFAULTS:
            print(f"skip {a.run_dir}: no probe defaults for {env}")
            return
        b = copy.copy(a)
        b._env = env
        print(_job((a.run_dir, b)), flush=True)
        return
    if a.emit_tasks:
        import json

        os.makedirs(os.path.dirname(os.path.abspath(a.emit_tasks)), exist_ok=True)
        n = 0
        with open(a.emit_tasks, "w") as f:
            for env in [e for e in a.envs if e in PROBE_DEFAULTS]:
                for mode in a.modes:
                    for seed in a.seeds:
                        d = os.path.join(a.runs_root, env, mode, f"seed{seed}")
                        if a.only_finished and not os.path.exists(os.path.join(d, "checkpoint.pt")):
                            continue
                        if a.overwrite or not os.path.exists(os.path.join(d, "probe.npz")):
                            f.write(json.dumps({"kind": "probe", "run_dir": d, "extra": a.extra}) + "\n")
                            n += 1
        print(f"wrote {n} probe tasks to {a.emit_tasks}")
        return
    jobs = []
    for env in a.envs:
        if env not in PROBE_DEFAULTS:
            print(f"skip {env}: no probe defaults (Box2D envs cannot be cloned)")
            continue
        for mode in a.modes:
            for seed in a.seeds:
                d = os.path.join(a.runs_root, env, mode, f"seed{seed}")
                if os.path.exists(os.path.join(d, "checkpoint.pt")):
                    b = copy.copy(a)
                    b._env = env
                    jobs.append((d, b))
    print(f"{len(jobs)} runs to probe with {a.workers} workers")
    if a.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_job, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(_job(job), flush=True)


if __name__ == "__main__":
    main(tyro.cli(ProbeArgs))
