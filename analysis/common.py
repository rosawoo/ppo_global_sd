"""Run loading and shared plot style for the analysis scripts."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Fixed colour per std_mode (colour follows the entity, never its rank).
MODE_COLORS = {"global": "#2a78d6", "state": "#eb6834", "state_separate": "#1baf7a", "state_detach": "#8a5cd6"}
MODE_LABELS = {"global": "global σ", "state": "state-dependent σ(s)", "state_separate": "σ(s), separate trunk",
               "state_detach": "σ(s), trunk detached"}
MODE_ORDER = ("global", "state", "state_separate", "state_detach")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# single-hue sequential ramp (light -> dark) for magnitude
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 160,
        "font.size": 9,
    })


def load_runs(runs_root: str) -> dict:
    """{env: {mode: [run, ...]}}; run = dict(seed, dir, status, evals(DataFrame), updates(DataFrame))."""
    out: dict = {}
    for status_path in sorted(glob.glob(os.path.join(runs_root, "*", "*", "seed*", "status.json"))):
        d = os.path.dirname(status_path)
        env, mode, seed = d.split(os.sep)[-3:]
        with open(status_path) as f:
            status = json.load(f)
        if status["status"] not in ("done", "nan"):
            continue  # still running / crashed: not part of the results
        run = {"seed": int(seed[4:]), "dir": d, "status": status}
        for name in ("evals", "updates"):
            p = os.path.join(d, f"{name}.csv")
            run[name] = pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 0 else pd.DataFrame()
        out.setdefault(env, {}).setdefault(mode, []).append(run)
    for env in out:
        for mode in out[env]:
            out[env][mode].sort(key=lambda r: r["seed"])
    return out


def eval_npz_paths(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "evals", "step_*.npz")))


def boot_ci(x, stat=np.mean, reps: int = 5000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over the first axis (seeds)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        v = stat(x, axis=0)
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    stats = np.stack([stat(x[i], axis=0) for i in idx])
    return np.nanpercentile(stats, 100 * alpha / 2, axis=0), np.nanpercentile(stats, 100 * (1 - alpha / 2), axis=0)


def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
