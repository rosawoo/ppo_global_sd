"""Aggregate the noise-tolerance probes (analysis/probe.py) into the 'why' figures.

    python analysis/probe_report.py --runs-root runs

Per env:
  figures/{env}_probe_cost.png      noise cost L(s, eta), precision requirement c(s) and the
                                    cost of the policy's own noise L_own(s), binned by the
                                    precision variables, for the states of each mode's runs
  figures/{env}_probe_sigma_vs_c.png learned sigma(s) of the state-dependent runs vs. the
                                    measured precision requirement c(s) of the same states
and results/probe_summary.{json,md}.

Definitions (eta in units of the action half-width, so c is comparable across action dims):
  L(s, eta) = V0(s) - E[Q_eta(s)]               (> 0: noise hurts)
  c(s)      = sum_l L_l eta_l^2 / sum_l eta_l^4   (least-squares fit of L ~ c eta^2)
  c_d(s)    = L_dim_d(s) / dim_level^2           (noise on action dim d only)
  heterogeneity index HI = AM(c) / GM(c) over on-policy states: by AM-GM, the factor by which
    an ideal state-dependent sigma (sigma^2 proportional to 1/c, same mean log sigma) would cut
    the quadratic noise cost sum_s c(s) sigma(s)^2 relative to the best global sigma.
    HI = 1: nothing to gain from sigma(s); HI >> 1: the precision requirement is very uneven.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tyro

from common import INK2, MODE_COLORS, MODE_LABELS, MODE_ORDER, MUTED, boot_ci, load_runs, savefig, set_style
from envs.wrappers import ENV_SPECS
from sigma import SIGMA_SPECS


@dataclass
class ReportArgs:
    runs_root: str = "runs"
    out_dir: str = "results"
    fig_dir: str = "figures"
    min_count: int = 3
    c_floor_quantile: float = 0.05
    """c below this on-policy quantile (incl. negative c: noise helped) is floored for the GM"""


def load_probe(run):
    p = os.path.join(run["dir"], "probe.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    eta = d["noise_levels"]
    L = d["V0"][:, None] - d["Q_eta"].mean(axis=2)  # [N, levels]
    d["L"] = L
    d["c"] = (L * eta**2).sum(1) / (eta**4).sum()
    d["L_own"] = d["V0"] - d["Q_own"].mean(1)
    if d["Q_dim"].size:
        d["c_dim"] = (d["V0"][:, None] - d["Q_dim"].mean(2)) / float(d["dim_level"]) ** 2  # [N, act]
    d["sigma_hw"] = d["std"] / d["half_width"]  # sigma in half-width units
    return d


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w)) if len(x) else np.nan


def binned_wmean(x, y, w, edges, min_count):
    idx = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_count:
            out[b] = wmean(y[m], w[m])
    return out


def heterogeneity(c, w, floor_q):
    """AM/GM of c over the on-policy distribution (weights w), c floored at its floor_q quantile."""
    order = np.argsort(c)
    cw = np.cumsum(w[order]) / w.sum()
    floor = c[order][np.searchsorted(cw, floor_q)]
    floor = max(floor, 1e-6 * max(np.abs(c).max(), 1e-12))
    cc = np.maximum(c, floor)
    am = wmean(cc, w)
    gm = float(np.exp(wmean(np.log(cc), w)))
    return am / gm


def _ratio(a, b):
    """a / b, but NaN unless both are positive (a cost ratio is meaningless when noise helps)."""
    return a / b if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else np.nan


def per_run_stats(env, d, floor_q):
    spec = SIGMA_SPECS[env]
    fine_fn = spec.variables[spec.fine_var][0]
    fine = fine_fn(d) < spec.fine_threshold
    w, c = d["weight"], d["c"]
    out = {
        "n_states": int(len(c)), "n_fine": int(fine.sum()),
        "frac_c_negative": wmean((c < 0).astype(float), w),
        "c_fine_mean": wmean(c[fine], w[fine]) if fine.any() else np.nan,
        "c_coarse_mean": wmean(c[~fine], w[~fine]) if (~fine).any() else np.nan,
        "heterogeneity_AM_over_GM": heterogeneity(c, w, floor_q),
        "L_own_fine_mean": wmean(d["L_own"][fine], w[fine]) if fine.any() else np.nan,
        "L_own_coarse_mean": wmean(d["L_own"][~fine], w[~fine]) if (~fine).any() else np.nan,
        "L_own_onpolicy_mean": wmean(d["L_own"], w),
    }
    out["c_fine_over_coarse"] = _ratio(out["c_fine_mean"], out["c_coarse_mean"])
    names = ENV_SPECS[env].action_names
    for i, act in enumerate(names):
        ci = d["c_dim"][:, i] if "c_dim" in d else c
        out[f"c_{act}_fine_mean"] = wmean(ci[fine], w[fine]) if fine.any() else np.nan
        out[f"c_{act}_coarse_mean"] = wmean(ci[~fine], w[~fine]) if (~fine).any() else np.nan
        out[f"c_{act}_fine_over_coarse"] = _ratio(out[f"c_{act}_fine_mean"], out[f"c_{act}_coarse_mean"])
        s = d["sigma_hw"][:, i]
        if s.std() > 1e-9:
            ok = ci > 0
            rho = scipy.stats.spearmanr(np.log(s[ok]), np.log(ci[ok]))[0] if ok.sum() > 5 else np.nan
        else:
            rho = np.nan  # global sigma: constant
        out[f"spearman_logsigma_vs_logc_{act}"] = float(rho)
        out[f"sigma_fine_over_coarse_{act}"] = (wmean(s[fine], w[fine]) / wmean(s[~fine], w[~fine])
                                               if fine.any() and (~fine).any() else np.nan)
    return out


def plot_cost(env, data, a: ReportArgs):
    spec = SIGMA_SPECS[env]
    any_d = next(iter(next(iter(data.values())).values()))
    eta = any_d["noise_levels"]
    rows = [(f"L(s, η={e:g})", lambda d, k=k: d["L"][:, k]) for k, e in enumerate(eta)]
    rows += [("c(s): precision requirement", lambda d: d["c"]), ("L_own(s): cost of own σ", lambda d: d["L_own"])]
    ncol = len(spec.variables)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.6 * ncol, 2.4 * len(rows)), squeeze=False, sharex="col")
    for j, (vname, (fn, edges)) in enumerate(spec.variables.items()):
        centers = 0.5 * (edges[:-1] + edges[1:])
        for i, (rlab, yfn) in enumerate(rows):
            ax = axes[i][j]
            for mode, seeds in data.items():
                per_seed = np.stack([binned_wmean(fn(d), yfn(d), d["weight"], edges, a.min_count) for d in seeds.values()])
                ok = np.sum(~np.isnan(per_seed), axis=0) >= max(1, len(per_seed) // 2)
                with np.errstate(all="ignore"):
                    mean = np.where(ok, np.nanmean(per_seed, axis=0), np.nan)
                lo, hi = boot_ci(per_seed, stat=np.nanmean)
                c = MODE_COLORS[mode]
                ax.fill_between(centers, np.where(ok, lo, np.nan), np.where(ok, hi, np.nan), color=c, alpha=0.18, lw=0)
                ax.plot(centers, mean, "-o", color=c, ms=3.5, label=f"states of {MODE_LABELS[mode]} runs (n={len(per_seed)})")
            ax.axhline(0, color=MUTED, lw=0.8, ls=":")
            if spec.fine_var == vname:
                ax.axvline(spec.fine_threshold, color=MUTED, lw=1, ls=":")
            if j == 0:
                ax.set_ylabel(rlab, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel(vname)
    axes[0][0].legend(loc="best", fontsize=7)
    H, W = int(any_d["horizon"]), int(any_d["window"])
    fig.suptitle(f"{env}: what action noise costs, by state (noise for {H} steps, return over {W} steps; "
                 f"η in action half-widths)", y=1.0)
    fig.tight_layout()
    savefig(fig, os.path.join(a.fig_dir, f"{env}_probe_cost.png"))


def plot_sigma_vs_c(env, data, a: ReportArgs):
    names = ENV_SPECS[env].action_names
    spec = SIGMA_SPECS[env]
    fine_fn = spec.variables[spec.fine_var][0]
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.0), squeeze=False)
    for i, act in enumerate(names):
        ax = axes[0][i]
        allx = []
        for mode, seeds in data.items():
            xs, ys, fs = [], [], []
            for d in seeds.values():
                ci = d["c_dim"][:, i] if "c_dim" in d else d["c"]
                ok = ci > 0
                xs.append(ci[ok])
                ys.append(d["sigma_hw"][ok, i])
                fs.append(fine_fn(d)[ok] < spec.fine_threshold)
            x, y, f = np.concatenate(xs), np.concatenate(ys), np.concatenate(fs)
            allx.append(x)
            c = MODE_COLORS[mode]
            ax.scatter(x[~f], y[~f], s=8, color=c, alpha=0.35, lw=0, label=f"{MODE_LABELS[mode]}: coarse region")
            ax.scatter(x[f], y[f], s=14, facecolor="none", edgecolor=c, alpha=0.8, lw=0.8,
                       label=f"{MODE_LABELS[mode]}: fine region")
        allx = np.concatenate(allx) if allx else np.array([1e-3, 1.0])
        if len(allx) < 2:
            allx = np.array([1e-3, 1.0])
        xx = np.geomspace(np.percentile(allx, 2), np.percentile(allx, 98), 20)
        ref = np.nanmedian(np.concatenate([np.concatenate([d["sigma_hw"][:, i] for d in s.values()]) for s in data.values()]))
        ax.plot(xx, ref * (xx / np.median(xx)) ** -0.5, color=INK2, lw=1, ls="--", label="slope −½ (σ ∝ c^−½)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"precision requirement c ({'noise on ' + act + ' only' if 'c_dim' in d else 'isotropic noise'})")
        ax.set_ylabel(f"policy σ ({act}, half-widths)")
    axes[0][0].legend(loc="best", fontsize=7)
    fig.suptitle(f"{env}: does the learned σ(s) track how much precision each state needs? (states with c > 0)", y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(a.fig_dir, f"{env}_probe_sigma_vs_c.png"))


def main(a: ReportArgs):
    set_style()
    runs = load_runs(a.runs_root)
    summary, md = {}, []
    for env, by_mode in runs.items():
        if env not in SIGMA_SPECS:
            continue
        data = {}
        for mode in MODE_ORDER:
            seeds = {r["seed"]: load_probe(r) for r in by_mode.get(mode, [])}
            seeds = {s: d for s, d in seeds.items() if d is not None and len(d["c"])}
            if seeds:
                data[mode] = seeds
        if not data:
            continue
        stats = {mode: {s: per_run_stats(env, d, a.c_floor_quantile) for s, d in seeds.items()}
                 for mode, seeds in data.items()}
        summary[env] = stats
        plot_cost(env, data, a)
        plot_sigma_vs_c(env, data, a)
        keys = list(next(iter(next(iter(stats.values())).values())))
        md += [f"### {env}", "", "median over seeds [min, max]", "", "| quantity | " + " | ".join(data) + " |",
               "|---|" + "---|" * len(data)]
        for k in keys:
            cells = []
            for mode in data:
                v = np.array([stats[mode][s][k] for s in stats[mode]], dtype=float)
                cells.append("–" if np.all(np.isnan(v)) else
                             f"{np.nanmedian(v):.3g} [{np.nanmin(v):.3g}, {np.nanmax(v):.3g}]")
            md.append(f"| {k} | " + " | ".join(cells) + " |")
        md.append("")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(a.out_dir, "probe_summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(tyro.cli(ReportArgs))
