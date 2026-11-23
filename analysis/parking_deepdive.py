"""Deep dive on highway-env parking-v0: what the two sigma parameterisations do differently.

    python analysis/parking_deepdive.py --runs-root runs

Uses evals.csv (every eval) and the eval dumps (final eval by default) of every finished
parking run. Writes to --fig-dir (default figures/parking_deepdive/):

  outcomes_over_training.png  success / crash / time-out fractions of the stochastic and the
                              deterministic policy over training (how each mode fails)
  trajectories.png            car paths in the goal frame (goal at the origin, goal heading
                              = +x), coloured by the policy's steering / acceleration sigma
  approach_profile.png        speed, |steering mean|, sigma and executed-action noise as a
                              function of the distance to the spot (the fine-manoeuvre phase)
  final_state.png             CDFs of the final distance and the final heading error
and results/parking_deepdive.{md,json} with the per-mode numbers (IQM over seeds).

Success in parking-v0 needs sqrt(sum_i w_i |achieved_i - goal_i|) < 0.12 with weights
(1, 0.3, 0, 0, 0.02, 0.02) on (x/100, y/100, vx/5, vy/5, cos h, sin h), i.e. roughly
|dx| < 1.4 m along x, |dy| < 4.8 m along y, a heading error of a few tens of degrees,
and the error budget is shared between them. Velocity does not count, but the episode
must still end with the car inside the tolerance.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tyro

from common import INK, INK2, MODE_COLORS, MODE_LABELS, MODE_ORDER, MUTED, SEQ, boot_ci, eval_npz_paths, load_runs, \
    savefig, set_style
from sigma import _parking_goal_frame, _parking_heading_err, clipped_normal_std

ACT = ("acceleration", "steering")


@dataclass
class DeepDiveArgs:
    env: str = "parking-v0"
    """parking-v0 or the optional no-walls variant parking-nowalls-v0"""
    runs_root: str = "runs"
    out_dir: str = "results"
    fig_dir: str = "figures/parking_deepdive"
    traj_seeds: int = 3
    """training seeds whose final-eval trajectories are drawn (the first k, no cherry-picking)"""
    policy: str = "stoch"
    """episodes for the trajectory / profile figures: stoch | det"""


def iqm(x, axis=0):
    return scipy.stats.trim_mean(x, 0.25, axis=axis)


def final_dump(run, policy):
    paths = eval_npz_paths(run["dir"])
    if not paths:
        return None
    with np.load(paths[-1]) as z:
        d = {k: z[k] for k in z.files}
    keep = d["deterministic"] == (policy == "det")
    steps = {k: v[keep] for k, v in d.items() if not k.startswith("ep_")}
    ep_keep = d["ep_deterministic"] == float(policy == "det")
    eps = {k: v[ep_keep] for k, v in d.items() if k.startswith("ep_")}
    return steps, eps


def outcome_curves(runs, tag):
    """[seed, eval] success, crash and time-out fractions."""
    n = min(len(r["evals"]) for r in runs)
    steps = runs[0]["evals"]["global_step"].to_numpy()[:n]
    succ = np.stack([r["evals"][f"{tag}_success_rate"].to_numpy()[:n] for r in runs])
    crash = np.stack([r["evals"][f"{tag}_crash_rate"].to_numpy()[:n] for r in runs])
    return steps, succ, crash, 1.0 - succ - crash


def plot_outcomes(by_mode, a):
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True, sharey=True)
    for i, tag in enumerate(("stoch", "det")):
        for mode, runs in by_mode.items():
            steps, succ, crash, tout = outcome_curves(runs, tag)
            for j, (arr, lab) in enumerate(((succ, "success"), (crash, "crash (hit a wall)"), (tout, "time-out"))):
                ax = axes[i][j]
                for row in arr:
                    ax.plot(steps, row, color=MODE_COLORS[mode], lw=0.5, alpha=0.25)
                ax.plot(steps, iqm(arr), color=MODE_COLORS[mode], lw=2, label=f"{MODE_LABELS[mode]} (n={len(arr)})")
                ax.set_title(f"{lab} - {'stochastic' if tag == 'stoch' else 'deterministic (mean)'} policy")
                ax.set_ylim(-0.02, 1.02)
                if i == 1:
                    ax.set_xlabel("environment steps")
    axes[0][0].set_ylabel("fraction of eval episodes")
    axes[1][0].set_ylabel("fraction of eval episodes")
    axes[0][0].legend(loc="best")
    fig.suptitle("how parking episodes end over training (bold: IQM over seeds, thin: seeds)", y=1.0)
    fig.tight_layout()
    savefig(fig, os.path.join(a.fig_dir, "outcomes_over_training.png"))


def plot_trajectories(dumps, a):
    """dumps: {mode: {seed: (steps, eps)}} -> one column per mode, rows: sigma_accel, sigma_steer."""
    modes = list(dumps)
    allstd = np.concatenate([s["std"] for m in modes for s, _ in dumps[m].values()])
    fig, axes = plt.subplots(2, len(modes), figsize=(5.4 * len(modes), 9.4), squeeze=False, sharex=True, sharey=True)
    for j, mode in enumerate(modes):
        for i, act in enumerate(ACT):
            ax = axes[i][j]
            vmin, vmax = np.percentile(allstd[:, i], 2), np.percentile(allstd[:, i], 98)
            for seed in sorted(dumps[mode])[: a.traj_seeds]:
                st, ep = dumps[mode][seed]
                gx, gy = _parking_goal_frame(st)
                for e in np.unique(st["episode_id"]):
                    m = st["episode_id"] == e
                    sc = ax.scatter(gx[m], gy[m], c=st["std"][m, i], cmap=SEQ, vmin=vmin, vmax=vmax, s=5, lw=0)
                    ax.plot(gx[m], gy[m], color=MUTED, lw=0.3, alpha=0.5)
                    k = np.flatnonzero(m)[-1]
                    ok = ep["ep_success"][list(np.unique(st["episode_id"])).index(e)] > 0.5
                    crashed = ep["ep_crashed"][list(np.unique(st["episode_id"])).index(e)] > 0.5
                    ax.plot(gx[k], gy[k], marker="o" if ok else ("x" if crashed else "s"), ms=5 if ok else 6,
                            color="#1baf7a" if ok else ("#d64545" if crashed else INK2), mfc="none" if not ok else None)
            ax.add_patch(plt.Rectangle((-2.0, -1.0), 4.0, 2.0, fill=False, ec=INK, lw=1.2))  # ~car-size target box
            ax.plot([0], [0], marker="+", color=INK, ms=12, mew=1.5)
            ax.set_xlim(-25, 25)
            ax.set_ylim(-25, 25)
            ax.set_aspect("equal")
            ax.set_title(f"{MODE_LABELS[mode]}: colour = σ ({act})", fontsize=9)
            if i == 1:
                ax.set_xlabel("along goal heading (m)")
            if j == 0:
                ax.set_ylabel("across goal heading (m)")
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=f"σ {act}")
    fig.suptitle(f"parking-v0: final-policy {a.policy} eval trajectories of the first {a.traj_seeds} seeds in the goal "
                 "frame\n(end marker: o success, x crash, □ time-out; box ≈ the spot)", y=1.0)
    fig.tight_layout()
    savefig(fig, os.path.join(a.fig_dir, "trajectories.png"))


def plot_profile(dumps, a):
    edges = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20, 30.0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = [("speed (m/s)", lambda s: np.hypot(s["goal_rel"][:, 2], s["goal_rel"][:, 3])),
            ("|heading error| (rad)", _parking_heading_err),
            ("|mean steering|", lambda s: np.abs(s["mean"][:, 1])),
            ("|mean acceleration|", lambda s: np.abs(s["mean"][:, 0])),
            ("σ steering", lambda s: s["std"][:, 1]),
            ("σ acceleration", lambda s: s["std"][:, 0]),
            ("executed-noise steering", lambda s: clipped_normal_std(s["mean"][:, 1], s["std"][:, 1], -1, 1)),
            ("executed-noise acceleration", lambda s: clipped_normal_std(s["mean"][:, 0], s["std"][:, 0], -1, 1))]
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    for ax, (lab, fn) in zip(axes.flat, rows):
        for mode, seeds in dumps.items():
            per_seed = []
            for st, _ in seeds.values():
                x, y = st["goal_dist"], fn(st)
                idx = np.digitize(x, edges) - 1
                per_seed.append([y[idx == b].mean() if (idx == b).sum() >= 3 else np.nan for b in range(len(centers))])
            per_seed = np.asarray(per_seed)
            with np.errstate(all="ignore"):
                mean = np.nanmean(per_seed, axis=0)
            lo, hi = boot_ci(per_seed, stat=np.nanmean)
            ax.fill_between(centers, lo, hi, color=MODE_COLORS[mode], alpha=0.18, lw=0)
            ax.plot(centers, mean, "-o", ms=3, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        ax.axvline(3.0, color=MUTED, lw=1, ls=":")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("distance to the spot (m, symlog)")
    axes[0][0].legend(loc="best")
    fig.suptitle(f"parking-v0: approach profile of the final policy ({a.policy} eval episodes; mean over seeds, 95% CI; "
                 "dotted = 3 m fine-control boundary)", y=1.0)
    fig.tight_layout()
    savefig(fig, os.path.join(a.fig_dir, "approach_profile.png"))


def plot_final_state(dumps, a):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for mode, seeds in dumps.items():
        dist, head = [], []
        for st, ep in seeds.values():
            last = np.r_[np.flatnonzero(np.diff(st["episode_id"])), len(st["episode_id"]) - 1]
            dist.append(ep["ep_final_goal_dist"])
            head.append(_parking_heading_err({k: v[last] for k, v in st.items()}))
        for ax, x in zip(axes, (np.concatenate(dist), np.concatenate(head))):
            xs = np.sort(x)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=MODE_COLORS[mode], lw=2,
                    label=f"{MODE_LABELS[mode]} ({len(seeds)} seeds)")
    axes[0].set_xscale("symlog", linthresh=0.5)
    axes[0].set_xlabel("final distance to the spot (m)")
    axes[1].set_xlabel("final |heading error| (rad)")
    for ax in axes:
        ax.set_ylabel("fraction of episodes ≤ x")
    axes[0].axvline(1.4, color=MUTED, ls=":", lw=1)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"parking-v0: where the final policy's {a.policy} episodes end", y=1.02)
    savefig(fig, os.path.join(a.fig_dir, "final_state.png"))


def mode_numbers(by_mode, dumps):
    out = {}
    for mode, runs in by_mode.items():
        rows = {}
        for tag in ("stoch", "det"):
            steps, succ, crash, tout = outcome_curves(runs, tag)
            rows[f"{tag}_success_final"] = succ[:, -3:].mean(1)
            rows[f"{tag}_crash_final"] = crash[:, -3:].mean(1)
            rows[f"{tag}_timeout_final"] = tout[:, -3:].mean(1)
            rows[f"{tag}_first_step_success_ge_0.5"] = np.array(
                [steps[np.argmax(s >= 0.5)] if (s >= 0.5).any() else np.nan for s in succ])
        if mode in dumps:
            fine_frac, sfine, scoarse, efine, ecoarse, tpark, vfine = [], [], [], [], [], [], []
            for st, ep in dumps[mode].values():
                near = st["goal_dist"] < 3.0
                eff = clipped_normal_std(st["mean"], st["std"], -1, 1)
                fine_frac.append(near.mean())
                sfine.append(st["std"][near].mean(0) if near.any() else np.full(2, np.nan))
                scoarse.append(st["std"][~near].mean(0))
                efine.append(eff[near].mean(0) if near.any() else np.full(2, np.nan))
                ecoarse.append(eff[~near].mean(0))
                ok = ep["ep_success"] > 0.5
                tpark.append(ep["ep_length"][ok].mean() if ok.any() else np.nan)
                vfine.append(np.hypot(st["goal_rel"][near, 2], st["goal_rel"][near, 3]).mean() if near.any() else np.nan)
            sfine, scoarse, efine, ecoarse = map(np.asarray, (sfine, scoarse, efine, ecoarse))
            rows["frac_steps_within_3m"] = np.asarray(fine_frac)
            rows["steps_to_park_successful_eps"] = np.asarray(tpark)
            rows["speed_within_3m"] = np.asarray(vfine)
            for i, act in enumerate(ACT):
                rows[f"sigma_{act}_within_3m"] = sfine[:, i]
                rows[f"sigma_{act}_beyond_3m"] = scoarse[:, i]
                rows[f"exec_noise_{act}_within_3m"] = efine[:, i]
                rows[f"exec_noise_{act}_beyond_3m"] = ecoarse[:, i]
        out[mode] = rows
    return out


def main(a: DeepDiveArgs):
    set_style()
    runs = load_runs(a.runs_root).get(a.env, {})
    by_mode = {m: runs[m] for m in MODE_ORDER if m in runs and runs[m]}
    if not by_mode:
        raise SystemExit(f"no finished {a.env} runs under {a.runs_root}")
    if a.env != "parking-v0":
        a.fig_dir = a.fig_dir.rstrip("/") + "_" + a.env
    tag = "" if a.env == "parking-v0" else "_" + a.env
    os.makedirs(a.fig_dir, exist_ok=True)
    plot_outcomes(by_mode, a)
    dumps = {m: {r["seed"]: final_dump(r, a.policy) for r in rs} for m, rs in by_mode.items()}
    dumps = {m: {s: v for s, v in d.items() if v is not None} for m, d in dumps.items()}
    dumps = {m: d for m, d in dumps.items() if d}
    if dumps:
        plot_trajectories(dumps, a)
        plot_profile(dumps, a)
        plot_final_state(dumps, a)
    nums = mode_numbers(by_mode, dumps)
    keys = list(next(iter(nums.values())))
    md = ["### parking-v0 deep dive", "", f"IQM over seeds [95% bootstrap CI of the mean]; final = mean of the last 3 "
          f"evals; σ / executed noise from the final eval's {a.policy} episodes", "",
          "| quantity | " + " | ".join(MODE_LABELS[m] for m in nums) + " |", "|---|" + "---|" * len(nums)]
    for k in keys:
        cells = []
        for m in nums:
            v = np.asarray(nums[m].get(k, [np.nan]), float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                cells.append("–")
                continue
            lo, hi = boot_ci(v) if len(v) > 1 else (v[0], v[0])
            cells.append(f"{iqm(v):.3g} [{lo:.3g}, {hi:.3g}] (n={len(v)})")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    os.makedirs(a.out_dir, exist_ok=True)
    md[0] = f"### {a.env} deep dive"
    with open(os.path.join(a.out_dir, f"parking_deepdive{tag}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out_dir, f"parking_deepdive{tag}.json"), "w") as f:
        json.dump({m: {k: [float(x) for x in v] for k, v in r.items()} for m, r in nums.items()}, f, indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    main(tyro.cli(DeepDiveArgs))
