"""H1 / H2: learning curves, rliable IQM with 95% stratified-bootstrap CIs, per-seed
curves, and the optimization-difficulty diagnostics.

    python analysis/curves.py --runs-root runs --out-dir results

Writes figures/{env}_curves.png, {env}_iqm.png, {env}_diagnostics.png and
{out_dir}/summary.json + summary.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tyro
from rliable import library as rly
from rliable import metrics as rmetrics

from common import MODE_COLORS, MODE_LABELS, MUTED, load_runs, savefig, set_style
from envs.wrappers import ENV_SPECS

RETURN_METRICS = [("stoch_return_mean", "eval return (stochastic policy)"),
                  ("det_return_mean", "eval return (deterministic policy)"),
                  ("stoch_success_rate", "success rate (stochastic policy)"),
                  ("det_success_rate", "success rate (deterministic policy)")]
DIAGNOSTICS = [("logstd_mean", "mean log σ"), ("logstd_min", "min log σ over states"), ("entropy", "policy entropy"),
               ("approx_kl", "approx. KL (mean over minibatches)"), ("clipfrac", "clip fraction"),
               ("explained_variance", "explained variance"), ("grad_norm_sigma", "‖∇‖ σ parameters"),
               ("grad_norm_mean_head", "‖∇‖ mean head"), ("logstd_state_std", "std of log σ across states")]


@dataclass
class CurveArgs:
    runs_root: str = "runs"
    """root of the sweep"""
    out_dir: str = "results"
    """where summary.json / summary.md go"""
    fig_dir: str = "figures"
    """where figures go"""
    reps: int = 20000
    """bootstrap replications"""
    last_k: int = 3
    """'final' = mean of the last k evals"""


def iqm(x, axis=0):
    """Interquartile mean along `axis` (same trimming as rliable.aggregate_iqm)."""
    import scipy.stats

    return scipy.stats.trim_mean(x, proportiontocut=0.25, axis=axis)


def curve_matrix(runs, metric):
    """[n_seeds, n_evals] on the common eval grid. A run that stopped early (NaN) is
    padded with its last recorded value, so it stays in the statistics (DECISIONS D18)."""
    n = max(len(r["evals"]) for r in runs)
    steps = max((r["evals"]["global_step"].to_numpy() for r in runs), key=len)
    mat = np.full((len(runs), n), np.nan)
    for i, r in enumerate(runs):
        v = r["evals"][metric].to_numpy(dtype=float) if len(r["evals"]) else np.array([np.nan])
        mat[i, : len(v)] = v
        mat[i, len(v):] = v[-1]
    return steps, mat


def seed_scores(steps, mat, last_k):
    auc = np.trapezoid(mat, steps, axis=1) / (steps[-1] - steps[0])  # time-averaged eval value
    return {"auc": auc, "final": mat[:, -last_k:].mean(axis=1)}


def rliable_iqm(score_by_mode, reps):
    """IQM and 95% stratified-bootstrap CI per mode (one task => stratified = resample seeds)."""
    scores = {m: np.asarray(v, dtype=float)[:, None] for m, v in score_by_mode.items()}
    point, ci = rly.get_interval_estimates(scores, lambda x: np.array([rmetrics.aggregate_iqm(x)]), reps=reps,
                                           random_state=np.random.RandomState(0))
    return {m: {"iqm": float(point[m][0]), "lo": float(ci[m][0, 0]), "hi": float(ci[m][1, 0])} for m in scores}


def iqm_difference(a, b, reps, seed=0):
    """IQM(b) - IQM(a) with a 95% bootstrap CI; seeds resampled independently per arm."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = iqm(a[rng.integers(0, len(a), (reps, len(a)))], axis=1)
    db = iqm(b[rng.integers(0, len(b), (reps, len(b)))], axis=1)
    d = db - da
    return {"diff": float(iqm(b) - iqm(a)), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_state_better": float((d > 0).mean())}


def plot_curves(env, by_mode, metrics_present, fig_dir):
    fig, axes = plt.subplots(1, len(metrics_present), figsize=(5.2 * len(metrics_present), 3.6), squeeze=False)
    for ax, (metric, label) in zip(axes[0], metrics_present):
        for mode, runs in by_mode.items():
            steps, mat = curve_matrix(runs, metric)
            c = MODE_COLORS[mode]
            for row in mat:
                ax.plot(steps, row, color=c, lw=0.6, alpha=0.3)
            mid = iqm(mat, axis=0)
            if len(mat) >= 4:
                rng = np.random.default_rng(0)
                bs = np.stack([iqm(mat[rng.integers(0, len(mat), len(mat))], axis=0) for _ in range(2000)])
                ax.fill_between(steps, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color=c, alpha=0.18, lw=0)
            ax.plot(steps, mid, color=c, lw=2, label=f"{MODE_LABELS[mode]} (IQM, n={len(mat)})")
        ax.set_xlabel("environment steps")
        ax.set_ylabel(label)
        ax.set_title(f"{env}: {label}")
    axes[0][0].legend(loc="best")
    fig.text(0.01, -0.02, "thin lines: individual seeds; bold: IQM across seeds; band: 95% bootstrap CI", color=MUTED,
             fontsize=8)
    savefig(fig, os.path.join(fig_dir, f"{env}_curves.png"))


def plot_iqm(env, table, fig_dir):
    keys = list(table)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 2.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        modes = list(table[key]["iqm"])
        for y, mode in enumerate(modes):
            e = table[key]["iqm"][mode]
            ax.plot([e["lo"], e["hi"]], [y, y], color=MODE_COLORS[mode], lw=6, alpha=0.35, solid_capstyle="round")
            ax.plot([e["iqm"]], [y], "o", color=MODE_COLORS[mode], ms=8, mec="white", mew=1.5)
            ax.annotate(f"{e['iqm']:.3g}", (e["iqm"], y), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=8)
        ax.set_yticks(range(len(modes)), [MODE_LABELS[m] for m in modes])
        ax.set_ylim(-0.6, len(modes) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(key, fontsize=9)
    fig.suptitle(f"{env}: IQM across seeds with 95% stratified-bootstrap CI", y=1.08, fontsize=10)
    savefig(fig, os.path.join(fig_dir, f"{env}_iqm.png"))


def plot_diagnostics(env, by_mode, fig_dir):
    diags = [d for d in DIAGNOSTICS if all(d[0] in r["updates"] for runs in by_mode.values() for r in runs)]
    ncol = 3
    nrow = int(np.ceil(len(diags) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.9 * nrow), squeeze=False)
    for ax, (key, label) in zip(axes.flat, diags):
        for mode, runs in by_mode.items():
            c = MODE_COLORS[mode]
            n = min(len(r["updates"]) for r in runs)
            for r in runs:
                u = r["updates"]
                ax.plot(u["global_step"], u[key].rolling(10, min_periods=1).mean(), color=c, lw=0.6, alpha=0.3)
            if n > 0:
                med = np.median(np.stack([r["updates"][key].rolling(10, min_periods=1).mean().to_numpy()[:n]
                                          for r in runs]), axis=0)
                ax.plot(runs[0]["updates"]["global_step"][:n], med, color=c, lw=2, label=MODE_LABELS[mode])
        if key.startswith("grad_norm") or key == "approx_kl":
            ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("environment steps")
    for ax in axes.flat[len(diags):]:
        ax.axis("off")
    axes.flat[0].legend(loc="best")
    fig.suptitle(f"{env}: optimization diagnostics (thin: seeds, bold: median; 10-update moving average)", y=1.01)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"{env}_diagnostics.png"))


def h2_table(env, by_mode, args_last_k):
    spec = ENV_SPECS[env]
    out = {}
    for mode, runs in by_mode.items():
        steps, mat = curve_matrix(runs, "stoch_return_mean")
        sc = seed_scores(steps, mat, args_last_k)
        _, learned_mat = curve_matrix(runs, spec.learned_metric)
        never = float(np.mean(np.nanmax(learned_mat, axis=1) < spec.learned_threshold))
        ups = [r["updates"] for r in runs if len(r["updates"])]
        cat = lambda k, f: float(np.mean([f(u[k]) for u in ups])) if ups else float("nan")
        logstd_floor = runs[0]["status"] and json.load(open(os.path.join(runs[0]["dir"], "args.json")))["logstd_min"]
        out[mode] = {
            "n_seeds": len(runs),
            "n_nan_runs": int(sum(r["status"]["status"] == "nan" for r in runs)),
            "final_return_std_across_seeds": float(np.std(sc["final"], ddof=1)) if len(runs) > 1 else float("nan"),
            "final_return_iqr_across_seeds": float(np.subtract(*np.percentile(sc["final"], [75, 25]))),
            "auc_std_across_seeds": float(np.std(sc["auc"], ddof=1)) if len(runs) > 1 else float("nan"),
            f"frac_seeds_never_learned ({spec.learned_metric} >= {spec.learned_threshold})": never,
            "approx_kl_mean": cat("approx_kl", np.mean),
            "approx_kl_max_minibatch_p99": cat("approx_kl_max", lambda s: np.percentile(s, 99)),
            "frac_updates_kl_gt_0.02": cat("approx_kl", lambda s: np.mean(s > 0.02)),
            "clipfrac_mean": cat("clipfrac", np.mean),
            "explained_variance_final": cat("explained_variance", lambda s: s.iloc[-10:].mean()),
            "logstd_mean_final": cat("logstd_mean", lambda s: s.iloc[-1]),
            "logstd_min_final": cat("logstd_min", lambda s: s.iloc[-1]),
            "frac_seeds_sigma_hit_floor": cat("logstd_min", lambda s: float(s.min() <= logstd_floor + 1e-6)),
            "logstd_frac_at_min_final": cat("logstd_frac_at_min", lambda s: s.iloc[-1]),
            "grad_norm_sigma_over_mean_head": cat("grad_norm_sigma", np.mean) / cat("grad_norm_mean_head", np.mean),
            "train_wall_clock_min_mean": float(np.mean([r["status"].get("wall_clock_s", np.nan) for r in runs]) / 60),
        }
    return out


def main(a: CurveArgs):
    set_style()
    runs = load_runs(a.runs_root)
    if not runs:
        raise SystemExit(f"no finished runs under {a.runs_root}")
    summary, md = {}, []
    for env, by_mode in runs.items():
        by_mode = {m: by_mode[m] for m in ("global", "state", "state_separate") if m in by_mode}
        present = [(k, label) for k, label in RETURN_METRICS
                   if all(k in r["evals"] and r["evals"][k].notna().any() for rs in by_mode.values() for r in rs)]
        table = {}
        for metric, _ in present:
            scores = {m: seed_scores(*curve_matrix(rs, metric), a.last_k) for m, rs in by_mode.items()}
            for kind in ("auc", "final"):
                key = f"{metric.replace('_mean', '')} / {kind}"
                by = {m: scores[m][kind] for m in by_mode}
                entry = {"iqm": rliable_iqm(by, a.reps), "per_seed": {m: [float(x) for x in v] for m, v in by.items()}}
                if "global" in by and "state" in by:
                    entry["state_minus_global"] = iqm_difference(by["global"], by["state"], a.reps)
                table[key] = entry
        h2 = h2_table(env, by_mode, a.last_k)
        summary[env] = {"performance": table, "h2": h2}
        plot_curves(env, by_mode, present, a.fig_dir)
        plot_iqm(env, table, a.fig_dir)
        plot_diagnostics(env, by_mode, a.fig_dir)

        md += [f"### {env}", "", "IQM across seeds [95% stratified-bootstrap CI]", "",
               "| metric | " + " | ".join(by_mode) + " | state − global [95% CI] | P(state > global) |",
               "|---|" + "---|" * (len(by_mode) + 2)]
        for key, e in table.items():
            cells = [f"{e['iqm'][m]['iqm']:.3f} [{e['iqm'][m]['lo']:.3f}, {e['iqm'][m]['hi']:.3f}]" for m in by_mode]
            d = e.get("state_minus_global")
            diff = f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['p_state_better']:.2f}" if d else "– | –"
            md.append(f"| {key} | " + " | ".join(cells) + f" | {diff} |")
        md += ["", "Optimization difficulty (H2)", "", "| diagnostic | " + " | ".join(by_mode) + " |",
               "|---|" + "---|" * len(by_mode)]
        for k in next(iter(h2.values())):
            md.append(f"| {k} | " + " | ".join(f"{h2[m][k]:.4g}" for m in by_mode) + " |")
        md.append("")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(tyro.cli(CurveArgs))
