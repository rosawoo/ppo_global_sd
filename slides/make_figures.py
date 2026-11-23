"""Figures for the Simglucose talk (slides/img/).

* Crops single panels out of the study's own figures (figures/*.png) so they can be shown large.
* Redraws the Simglucose learning curves from results/eval_curves.csv with slide-sized text
  (the original figures/…_curves.png is an 8-panel strip that is unreadable on a slide).
  Same statistics as analysis/curves.py: IQM across the 10 seeds, 95% bootstrap band
  (seeds resampled, 2000 draws), thin lines for individual seeds.
* Draws the two small "noise width" illustrations used on the intro slide.

Run from the repository root:  python slides/make_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import trim_mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style  # noqa: E402

ENV = "simglucose-adolescent1-v0"
OUT = "slides/img"
COL = {"global": "#2a78d6", "state": "#eb6834"}  # analysis/common.py MODE_COLORS
LAB = {"global": r"global $\sigma$", "state": r"state-dependent $\sigma(s)$"}

# Panels of the study's figures, as (x0, y0, x1, y1) pixel boxes.
CROPS = {
    "sigma_vs_cgm": ("sigma_binned.png", (0, 75, 792, 489)),
    "executed_noise_vs_cgm": ("effective_noise.png", (0, 70, 792, 489)),
    "min_logsigma": ("diagnostics.png", (770, 105, 1460, 545)),
}


def crop_study_figures():
    for name, (src, box) in CROPS.items():
        im = np.asarray(Image.open(f"figures/{ENV}_{src}").convert("RGB").crop(box)).copy()
        im[(im >= 248).all(axis=2)] = 255  # the figures' off-white surface -> the slide's white
        Image.fromarray(im).save(f"{OUT}/{name}.png", optimize=True)


def iqm(x, axis=0):
    return trim_mean(x, 0.25, axis=axis)


def curve(df, config, arm, metric):
    d = df[(df.config == config) & (df.env == ENV) & (df.arm == arm)]
    mat = d.pivot_table(index="seed", columns="global_step", values=metric)
    return mat.columns.to_numpy() / 1e6, mat.to_numpy() * 100.0


def plot_curves(df, config, path):
    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    metrics = [("stoch_success_rate", "days meeting targets (\\%)" if plt.rcParams["text.usetex"]
                else "days meeting targets (%)"),
               ("stoch_crash_rate", "days with a severe low (\\%)" if plt.rcParams["text.usetex"]
                else "days with a severe low (%)")]
    rng = np.random.default_rng(0)
    handles = []
    for ax, (metric, ylabel), letter in zip(axes, metrics, "ab"):
        for arm in ("global", "state"):
            x, mat = curve(df, config, arm, metric)
            for row in mat:
                ax.plot(x, row, color=COL[arm], lw=0.8, alpha=0.22)
            boot = np.stack([iqm(mat[rng.integers(0, len(mat), len(mat))], axis=0) for _ in range(2000)])
            ax.fill_between(x, np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0), color=COL[arm],
                            alpha=0.18, lw=0)
            (h,) = ax.plot(x, iqm(mat, axis=0), color=COL[arm], lw=2.5, label=LAB[arm])
            if letter == "a":
                handles.append(h)
        plot_style.clean_axes(ax)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-3, 103)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel(ylabel)
        ax.text(0.0, 1.04, rf"$\mathbf{{({letter})}}$", transform=ax.transAxes, ha="left", va="bottom")
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=2.0)
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.08, right=0.985, wspace=0.27)
    plot_style.save_checked(fig, path, dpi=200)
    plt.close(fig)


def plot_noise_widths(path, sigmas, color):
    """Three action distributions (one per state) on a shared action axis; no text."""
    plot_style.apply()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    a = np.linspace(-3.2, 3.2, 600)
    for mu, s, alpha in zip((-1.6, 0.0, 1.6), sigmas, (0.55, 0.8, 1.0)):
        p = np.exp(-0.5 * ((a - mu) / s) ** 2) / s
        p = p / 3.0
        ax.fill_between(a, p, color=color, alpha=0.18 * alpha + 0.05, lw=0)
        ax.plot(a, p, color=color, lw=2.5, alpha=alpha)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 1.45)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)
    plot_style.save_checked(fig, path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    crop_study_figures()
    df = pd.read_csv("results/eval_curves.csv")
    plot_curves(df, "tuned_shared", f"{OUT}/curves_shared.png")
    plot_curves(df, "tuned_per_arm", f"{OUT}/curves_perarm.png")
    plot_noise_widths(f"{OUT}/noise_global.png", (0.55, 0.55, 0.55), COL["global"])
    plot_noise_widths(f"{OUT}/noise_state.png", (1.0, 0.55, 0.25), COL["state"])
    for f in sorted(os.listdir(OUT)):
        print(f, Image.open(f"{OUT}/{f}").size)


if __name__ == "__main__":
    main()
