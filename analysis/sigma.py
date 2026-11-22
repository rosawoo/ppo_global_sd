"""H3 / H4: does the learned sigma vary with the state, and does it track precision needs?

    python analysis/sigma.py --runs-root runs --out-dir results

Uses the per-step eval dumps (last `--last-k` evals of every finished run, stochastic
episodes by default). For each env it produces
  figures/{env}_sigma_binned.png        binned sigma vs. precision variables (CIs across seeds)
  figures/{env}_sigma_heatmap_mean.png  seed-averaged sigma over the state plane
  figures/{env}_sigma_heatmap_seeds.png one row per seed
  figures/{env}_sigma_episodes.png      sigma(t) along a few fixed-seed episodes
and results/sigma_summary.{json,md}: Spearman rho per seed, variance decomposition,
near/far sigma ratios.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tyro

from common import (INK2, MODE_COLORS, MODE_LABELS, MUTED, SEQ, boot_ci, eval_npz_paths, load_runs, savefig, set_style)
from envs.wrappers import ENV_SPECS


def _parking_heading_err(d):
    o = d["obs"]
    a, g = np.arctan2(o[:, 5], o[:, 4]), np.arctan2(o[:, 11], o[:, 10])
    return np.abs(np.arctan2(np.sin(a - g), np.cos(a - g)))


def _parking_goal_frame(d):
    """Position relative to the goal, rotated so the goal heading is the +x axis."""
    dx, dy = d["goal_rel"][:, 0], d["goal_rel"][:, 1]
    cg, sg = d["obs"][:, 10], d["obs"][:, 11]
    return dx * cg + dy * sg, -dx * sg + dy * cg


@dataclass(frozen=True)
class SigmaSpec:
    variables: dict  # name -> (fn(data) -> [N], bin edges)
    heat_xy: object  # fn(data) -> (x, y)
    heat_labels: tuple
    heat_range: tuple  # ((xmin, xmax), (ymin, ymax))
    fine_var: str  # precision variable defining the "fine control" region
    fine_threshold: float  # region = fine_var < threshold


SIGMA_SPECS = {
    "parking-v0": SigmaSpec(
        variables={
            "distance to goal (m)": (lambda d: d["goal_dist"], np.array([0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30.0])),
            "|heading error| (rad)": (_parking_heading_err, np.linspace(0, np.pi, 9)),
        },
        heat_xy=_parking_goal_frame,
        heat_labels=("along goal heading (m)", "across goal heading (m)"),
        heat_range=((-30, 30), (-30, 30)),
        fine_var="distance to goal (m)", fine_threshold=3.0,
    ),
    "Pendulum-v1": SigmaSpec(
        variables={
            "|θ| from upright (rad)": (lambda d: np.abs(d["goal_rel"][:, 0]), np.linspace(0, np.pi, 9)),
            "|θ̇| (rad/s)": (lambda d: np.abs(d["goal_rel"][:, 1]), np.linspace(0, 8, 9)),
        },
        heat_xy=lambda d: (d["goal_rel"][:, 0], d["goal_rel"][:, 1]),
        heat_labels=("θ (rad, 0 = upright)", "θ̇ (rad/s)"),
        heat_range=((-np.pi, np.pi), (-8, 8)),
        fine_var="|θ| from upright (rad)", fine_threshold=0.3,
    ),
    "LunarLanderContinuous-v3": SigmaSpec(
        variables={
            "height above pad (obs units)": (lambda d: d["goal_rel"][:, 1], np.linspace(0, 1.5, 9)),
            "distance to pad (obs units)": (lambda d: d["goal_dist"], np.linspace(0, 1.6, 9)),
        },
        heat_xy=lambda d: (d["goal_rel"][:, 0], d["goal_rel"][:, 1]),
        heat_labels=("x (pad at 0)", "height y"),
        heat_range=((-1, 1), (-0.1, 1.6)),
        fine_var="height above pad (obs units)", fine_threshold=0.3,
    ),
}


@dataclass
class SigmaArgs:
    runs_root: str = "runs"
    out_dir: str = "results"
    fig_dir: str = "figures"
    last_k: int = 1
    """pool the last k eval dumps of each run (1 = final policy only; pooling several evals
    mixes policies from different training steps, which shows up as fake state-dependence)"""
    policy: str = "stoch"
    """which eval episodes to use: stoch | det | both"""
    heat_bins: int = 24
    min_count: int = 3
    """minimum samples for a bin to be shown"""
    episodes: int = 4
    """episodes in the sigma(t) figure (the first fixed eval seeds; no cherry-picking)"""
    episode_run_seeds: list[int] = field(default_factory=lambda: [1])
    """training seeds whose final eval is used for the sigma(t) figure"""


def load_eval_steps(run, last_k, policy):
    paths = eval_npz_paths(run["dir"])[-last_k:]
    if not paths:
        return None
    chunks = [dict(np.load(p)) for p in paths]
    keys = [k for k in chunks[0] if not k.startswith("ep_")]
    # make episode ids unique across the pooled dumps
    for i, c in enumerate(chunks):
        c["episode_id"] = c["episode_id"] + 1000 * i
    d = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    if policy != "both":
        keep = d["deterministic"] == (policy == "det")
        d = {k: v[keep] for k, v in d.items()}
    return d


def binned_mean(x, y, edges, min_count):
    idx = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_count:
            out[b] = y[m].mean()
    return out


def eta_squared(x, y, edges):
    """Fraction of Var[y] explained by the bin of x (correlation ratio)."""
    idx = np.digitize(x, edges) - 1
    tot = y.var()
    if tot < 1e-12:
        return 0.0
    between = sum((idx == b).sum() * (y[idx == b].mean() - y.mean()) ** 2 for b in np.unique(idx)) / len(y)
    return float(between / tot)


def per_seed_stats(env, d):
    spec, n_act = SIGMA_SPECS[env], d["std"].shape[1]
    logs = np.log(d["std"])
    out = {"n_steps": int(len(logs)), "dims": []}
    xs = {name: fn(d) for name, (fn, _) in spec.variables.items()}
    near = xs[spec.fine_var] < spec.fine_threshold
    for a in range(n_act):
        s = d["std"][:, a]
        row = {
            "action": ENV_SPECS[env].action_names[a],
            "sigma_mean": float(s.mean()),
            "logsigma_std_across_states": float(logs[:, a].std()),
            "sigma_p95_over_p5": float(np.percentile(s, 95) / np.percentile(s, 5)),
            "sigma_near": float(s[near].mean()) if near.sum() >= 5 else np.nan,
            "sigma_far": float(s[~near].mean()) if (~near).sum() >= 5 else np.nan,
            "n_near": int(near.sum()),
            "spearman": {}, "eta2": {},
        }
        row["far_over_near"] = row["sigma_far"] / row["sigma_near"]
        for name, (_, edges) in spec.variables.items():
            if logs[:, a].std() < 1e-9:
                rho, p = np.nan, np.nan  # constant sigma (global mode): rho undefined
            else:
                rho, p = scipy.stats.spearmanr(xs[name], s)
            row["spearman"][name] = {"rho": float(rho), "p": float(p)}
            row["eta2"][name] = eta_squared(xs[name], logs[:, a], edges)
        # R^2 of a linear fit of log sigma on all precision variables jointly
        X = np.column_stack([np.ones(len(s))] + list(xs.values()))
        resid = logs[:, a] - X @ np.linalg.lstsq(X, logs[:, a], rcond=None)[0]
        row["r2_linear_precision_vars"] = float(1 - resid.var() / logs[:, a].var()) if logs[:, a].var() > 1e-12 else 0.0
        out["dims"].append(row)
    return out


def plot_binned(env, data, a: SigmaArgs):
    spec, names = SIGMA_SPECS[env], ENV_SPECS[env].action_names
    fig, axes = plt.subplots(len(names), len(spec.variables), figsize=(4.8 * len(spec.variables), 3.0 * len(names)),
                             squeeze=False, sharey="row")
    for j, (vname, (fn, edges)) in enumerate(spec.variables.items()):
        centers = 0.5 * (edges[:-1] + edges[1:])
        for i, act in enumerate(names):
            ax = axes[i][j]
            for mode, seeds in data.items():
                c = MODE_COLORS[mode]
                per_seed = np.stack([binned_mean(fn(d), d["std"][:, i], edges, a.min_count) for d in seeds.values()])
                for row in per_seed:
                    ax.plot(centers, row, color=c, lw=0.6, alpha=0.3)
                ok = np.sum(~np.isnan(per_seed), axis=0) >= max(2, len(per_seed) // 2)
                mean = np.where(ok, np.nanmean(per_seed, axis=0), np.nan)
                lo, hi = boot_ci(per_seed, stat=np.nanmean)
                ax.fill_between(centers, np.where(ok, lo, np.nan), np.where(ok, hi, np.nan), color=c, alpha=0.18, lw=0)
                ax.plot(centers, mean, "-o", color=c, ms=4, label=f"{MODE_LABELS[mode]} (n={len(per_seed)})")
            ax.set_ylim(bottom=0)
            ax.set_xlabel(vname)
            if j == 0:
                ax.set_ylabel(f"σ ({act})")
            if spec.fine_var == vname:
                ax.axvline(spec.fine_threshold, color=MUTED, lw=1, ls=":")
                ax.annotate("fine-control region", (spec.fine_threshold, 0), xytext=(3, 4), textcoords="offset points",
                            fontsize=7, color=INK2)
    axes[0][0].legend(loc="best")
    fig.suptitle(f"{env}: σ vs. precision variables (bold: mean over seeds, band: 95% bootstrap CI, thin: seeds)", y=1.0)
    fig.tight_layout()
    savefig(fig, os.path.join(a.fig_dir, f"{env}_sigma_binned.png"))


def heat_grid(env, d, a: SigmaArgs):
    spec = SIGMA_SPECS[env]
    x, y = spec.heat_xy(d)
    (x0, x1), (y0, y1) = spec.heat_range
    cnt, xe, ye = np.histogram2d(x, y, bins=a.heat_bins, range=[[x0, x1], [y0, y1]])
    grids = []
    for i in range(d["std"].shape[1]):
        tot, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=d["std"][:, i])
        grids.append(np.where(cnt >= a.min_count, tot / np.maximum(cnt, 1), np.nan))
    return np.stack(grids), xe, ye


def plot_heatmaps(env, seeds: dict, mode, a: SigmaArgs):
    spec, names = SIGMA_SPECS[env], ENV_SPECS[env].action_names
    grids = {s: heat_grid(env, d, a) for s, d in seeds.items()}
    stack = np.stack([g[0] for g in grids.values()])  # [seed, act, nx, ny]
    xe, ye = next(iter(grids.values()))[1:]
    with np.errstate(all="ignore"):
        avg = np.where(np.sum(~np.isnan(stack), axis=0) >= max(1, len(stack) // 2), np.nanmean(stack, axis=0), np.nan)
    vlim = [(np.nanpercentile(stack[:, i], 2), np.nanpercentile(stack[:, i], 98)) for i in range(len(names))]

    def draw(ax, g, i, title):
        im = ax.pcolormesh(xe, ye, g.T, cmap=SEQ, vmin=vlim[i][0], vmax=vlim[i][1])
        ax.plot([0], [0], marker="+", color="#0b0b0b", ms=10, mew=1.5)
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        return im

    fig, axes = plt.subplots(1, len(names), figsize=(4.9 * len(names), 4.0), squeeze=False)
    for i, act in enumerate(names):
        im = draw(axes[0][i], avg[i], i, f"σ ({act}), mean over {len(stack)} seeds")
        axes[0][i].set_xlabel(spec.heat_labels[0])
        axes[0][i].set_ylabel(spec.heat_labels[1])
        fig.colorbar(im, ax=axes[0][i], label="σ")
    fig.suptitle(f"{env} [{MODE_LABELS[mode]}]: σ over the state plane (+ = goal; blank = <{a.min_count} visits)", y=1.02)
    savefig(fig, os.path.join(a.fig_dir, f"{env}_sigma_heatmap_mean_{mode}.png"))

    fig, axes = plt.subplots(len(names), len(stack), figsize=(2.4 * len(stack) + 0.8, 2.5 * len(names)), squeeze=False,
                             sharex=True, sharey=True)
    for k, s in enumerate(grids):
        for i, act in enumerate(names):
            im = draw(axes[i][k], stack[k, i], i, f"seed {s}" if i == 0 else "")
            if k == 0:
                axes[i][k].set_ylabel(f"σ ({act})\n{spec.heat_labels[1]}")
            if k == len(stack) - 1:
                fig.colorbar(im, ax=list(axes[i]), label="σ", fraction=0.015, pad=0.01)
    fig.suptitle(f"{env} [{MODE_LABELS[mode]}]: σ over the state plane, per seed (x: {spec.heat_labels[0]})", y=1.0)
    savefig(fig, os.path.join(a.fig_dir, f"{env}_sigma_heatmap_seeds_{mode}.png"))


def plot_episodes(env, runs_by_mode, a: SigmaArgs):
    """sigma(t) above the precision variable, sharing the time axis (no dual y-axes)."""
    spec, names = SIGMA_SPECS[env], ENV_SPECS[env].action_names
    fn = spec.variables[spec.fine_var][0]
    for run_seed in a.episode_run_seeds:
        run = next((r for r in runs_by_mode.get("state", []) if r["seed"] == run_seed), None)
        if run is None or not eval_npz_paths(run["dir"]):
            continue
        full = dict(np.load(eval_npz_paths(run["dir"])[-1]))
        n_rows = len(names) + 1
        fig, axes = plt.subplots(n_rows, a.episodes, figsize=(3.2 * a.episodes, 1.9 * n_rows), squeeze=False,
                                 sharex="col", sharey="row")
        for k in range(a.episodes):  # episodes 0..k-1 = first stochastic fixed-seed episodes
            m = full["episode_id"] == k
            d = {key: v[m] for key, v in full.items() if not key.startswith("ep_")}
            x = fn(d)
            inside = np.where(x < spec.fine_threshold)[0]
            for i, act in enumerate(names):
                axes[i][k].plot(d["t"], d["std"][:, i], color=MODE_COLORS["state"])
                axes[i][k].set_ylim(bottom=0)
                if k == 0:
                    axes[i][k].set_ylabel(f"σ ({act})")
            axes[-1][k].plot(d["t"], x, color=INK2)
            axes[-1][k].axhline(spec.fine_threshold, color=MUTED, lw=1, ls=":")
            axes[-1][k].set_xlabel("step")
            if k == 0:
                axes[-1][k].set_ylabel(spec.fine_var)
            if len(inside):
                for ax in axes[:, k]:
                    ax.axvline(d["t"][inside[0]], color=MUTED, lw=1, ls="--")
                axes[0][k].annotate("enters fine-control region", (d["t"][inside[0]], axes[0][k].get_ylim()[1]),
                                    xytext=(3, -9), textcoords="offset points", fontsize=7, color=INK2)
            succ = full["ep_success"][k]
            outcome = "" if np.isnan(succ) else (" · success" if succ else " · no success")
            axes[0][k].set_title(f"eval seed {int(full['ep_seed'][k])} · return {full['ep_return'][k]:.1f}{outcome}",
                                 fontsize=8)
        fig.suptitle(f"{env}: state-dependent σ(t) along stochastic eval episodes (training seed {run_seed})", y=1.0)
        fig.tight_layout()
        savefig(fig, os.path.join(a.fig_dir, f"{env}_sigma_episodes_seed{run_seed}.png"))


def main(a: SigmaArgs):
    set_style()
    runs = load_runs(a.runs_root)
    summary, md = {}, []
    for env, by_mode in runs.items():
        if env not in SIGMA_SPECS:
            continue
        data = {}
        for mode in ("global", "state", "state_separate"):
            seeds = {r["seed"]: load_eval_steps(r, a.last_k, a.policy) for r in by_mode.get(mode, [])}
            seeds = {s: d for s, d in seeds.items() if d is not None and len(d["t"]) > 0}
            if seeds:
                data[mode] = seeds
        if not data:
            continue
        stats = {mode: {s: per_seed_stats(env, d) for s, d in seeds.items()} for mode, seeds in data.items()}
        # pooled decomposition of Var[log sigma] into between-seed vs. within-seed (= state) parts
        pooled = {}
        for mode, seeds in data.items():
            pooled[mode] = []
            for i in range(len(ENV_SPECS[env].action_names)):
                logs = [np.log(d["std"][:, i]) for d in seeds.values()]
                allv = np.concatenate(logs)
                within = sum(len(v) * v.var() for v in logs) / len(allv)
                pooled[mode].append({"total_var": float(allv.var()),
                                     "within_seed_state_fraction": float(within / allv.var()) if allv.var() > 1e-12 else 0.0})
        summary[env] = {"per_seed": stats, "pooled_variance": pooled, "policy": a.policy, "last_k": a.last_k}
        plot_binned(env, data, a)
        for mode in data:
            if mode != "global":
                plot_heatmaps(env, data[mode], mode, a)
        plot_episodes(env, by_mode, a)

        spec = SIGMA_SPECS[env]
        md += [f"### {env}", ""]
        for mode in data:
            md += [f"**{MODE_LABELS[mode]}** — per training seed, {a.policy} eval episodes, last {a.last_k} evals", "",
                   "| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near "
                   f"({spec.fine_var} ≷ {spec.fine_threshold}) | "
                   + " | ".join(f"ρ: {v}" for v in spec.variables) + " | "
                   + " | ".join(f"η²: {v}" for v in spec.variables) + " | R² (linear, all vars) |",
                   "|---|---|---|---|---|---|" + "---|" * (2 * len(spec.variables) + 1)]
            for s, st in stats[mode].items():
                for r in st["dims"]:
                    md.append(f"| {s} | {r['action']} | {r['sigma_mean']:.3f} | {r['logsigma_std_across_states']:.3f} | "
                              f"{r['sigma_p95_over_p5']:.2f} | {r['far_over_near']:.2f} | "
                              + " | ".join(f"{r['spearman'][v]['rho']:+.2f}" for v in spec.variables) + " | "
                              + " | ".join(f"{r['eta2'][v]:.2f}" for v in spec.variables)
                              + f" | {r['r2_linear_precision_vars']:.2f} |")
            for i, act in enumerate(ENV_SPECS[env].action_names):
                rows = [st["dims"][i] for st in stats[mode].values()]
                med = lambda f: float(np.nanmedian([f(r) for r in rows]))
                line = (f"- {act}: median std_s(log σ) = {med(lambda r: r['logsigma_std_across_states']):.3f}; "
                        f"median σ far/near = {med(lambda r: r['far_over_near']):.2f}; "
                        f"within-seed (state) share of pooled Var[log σ] = "
                        f"{pooled[mode][i]['within_seed_state_fraction']:.2f}")
                for v in spec.variables:
                    rhos = np.array([r["spearman"][v]["rho"] for r in rows])
                    if np.all(np.isnan(rhos)):
                        continue
                    # seeds are the independent units; per-step p-values are not used (autocorrelated steps)
                    line += (f"; ρ(σ, {v}): median {np.nanmedian(rhos):+.2f}, "
                             f"{int(np.sum(rhos > 0.1))}/{len(rhos)} seeds with ρ > +0.1, "
                             f"{int(np.sum(rhos < -0.1))}/{len(rhos)} with ρ < −0.1")
                md.append(line)
            md.append("")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "sigma_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(a.out_dir, "sigma_summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(tyro.cli(SigmaArgs))
