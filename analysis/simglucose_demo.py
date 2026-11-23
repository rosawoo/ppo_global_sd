"""Simglucose demo and interpretability check: one global-sigma and one sigma(s) policy.

    python analysis/simglucose_demo.py --global-run RUN_DIR --state-run RUN_DIR --out-dir demos/simglucose

Both policies are run on the same simulated days (same reset seed = same patient start, meal
times and sizes; same action-noise seed). Writes to --out-dir:

* day_<seed>.mp4 (+ .gif): one day side by side. Top: glucose with the 70-180 mg/dL target
  range. Bottom: the insulin the policy means to give (line), its +-1 sigma band after the dose
  limits (shaded), and the dose actually delivered (dots), all in multiples of the basal rate.
* policy_map.png: what each policy does as a function of glucose: the dose it means to give,
  its sigma, and how often noise still delivers insulin.
* interpretability.json: the numbers behind the plain-English statements in demos/README.md.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import torch
import tyro
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "slides")]
import plot_style  # noqa: E402  (house plot style, slides/plot_style.py)
from envs.wrappers import make_env  # noqa: E402
from scripts.eval import load_run  # noqa: E402

ENV = "simglucose-adolescent1-v0"
COL = {"global": "#2a78d6", "state": "#eb6834"}
NAME = {"global": r"Global $\sigma$", "state": r"State-dependent $\sigma(s)$"}
MAX_MULT = 10.0  # linear action map: rate = (a + 1) / 2 * 10 * basal


@dataclass
class Args:
    global_run: str
    state_run: str
    out_dir: str = "demos/simglucose"
    seeds: int = 20
    """days to analyse (reset seeds seed_base .. seed_base + seeds - 1)"""
    seed_base: int = 100_000
    """the study's evaluation days"""
    noise_seed: int = 0
    days_to_render: str = "auto"
    """comma-separated reset seeds to animate, or 'auto' (a typical day and the day where sigma(s) goes lowest)"""
    steps_per_frame: int = 3
    fps: int = 16
    gif: bool = True


def dose(a):
    """action -> insulin rate in multiples of basal (the env clips the action to [-1, 1])."""
    return (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * MAX_MULT


def run_day(agent, normalizer, env, seed, noise_seed):
    """One stochastic day. Decision-time features (cgm, trend) are recorded with the action taken."""
    gen = torch.Generator().manual_seed(noise_seed)
    obs, info = env.reset(seed=seed)
    rec = {k: [] for k in ("cgm", "trend", "mu", "sigma", "action", "delivered", "bg_after", "cgm_after", "meal")}
    done = False
    while not done:
        x = torch.as_tensor(normalizer.obs(obs[None], update=False))
        with torch.no_grad():
            d, _ = agent.dist(x)
        mu, sd = float(d.mean[0, 0]), float(d.stddev[0, 0])
        a = mu + sd * float(torch.randn(1, generator=gen)[0])
        rec["cgm"].append(float(info["goal_rel"][0]))
        rec["trend"].append(float(info["goal_rel"][1]))
        rec["mu"].append(mu)
        rec["sigma"].append(sd)
        rec["action"].append(a)
        obs, _, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
        rec["delivered"].append(float(info["insulin_rate_x_basal"]))
        rec["bg_after"].append(float(info["bg"]))
        rec["cgm_after"].append(float(info["goal_rel"][0]))
        rec["meal"].append(float(info["meal"]))
        done = terminated or truncated
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["stats"] = {"tir": float(info["tir"]), "tbr": float(info["tbr"]), "severe_low": bool(info["crashed"]),
                    "success": bool(info["is_success"]), "min_bg": float(out["bg_after"].min()), "seed": seed}
    return out


# ---- the animation ---------------------------------------------------------------------
def rolling_mean(x, w):
    """Mean over the last w steps (fewer at the start of the day)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def animate(days, seed, path, steps_per_frame, fps, gif, slide=False):
    """Three rows per agent: blood glucose, insulin given (30-min average) and the policy's sigma.

    slide=True: a more compact frame (so the text is larger relative to it), no figure title, GIF only."""
    plot_style.apply()
    n = len(days["global"]["cgm"])
    t = np.arange(1, n + 1) * 3 / 60.0  # hours (one step = 3 min)
    fig, axes = plt.subplots(3, 2, figsize=(12, 6.75) if slide else (16, 9), dpi=160 if slide else 120, sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 0.85, 0.85], "hspace": 0.16, "wspace": 0.16})
    fig.subplots_adjust(left=0.1 if slide else 0.075, right=0.985, top=0.83 if slide else 0.8, bottom=0.1 if slide else 0.085)
    arts = {}
    for j, m in enumerate(("global", "state")):
        d = days[m]
        g, ins, sg = axes[0, j], axes[1, j], axes[2, j]
        for ax in (g, ins, sg):
            plot_style.clean_axes(ax)
            ax.set_xlim(0, 24)
        g.axhspan(70, 180, color="#dcefdc", lw=0, zorder=0)
        g.axhline(54, color="#b03a2e", lw=1.2, ls="--", zorder=1)
        g.set_ylim(20, 320)
        g.set_yticks([70, 180, 300])
        g.set_title(NAME[m], color=COL[m], fontsize=24 if slide else 26, pad=36 if slide else 40)
        meal_t = t[d["meal"] > 0]
        g.plot(meal_t, np.full_like(meal_t, 305), "v", color="#6e6e73", ms=8, zorder=3)
        ins.set_ylim(0, 8)
        ins.set_yticks([0, 4, 8])
        sg.set_ylim(0, 8)
        sg.set_yticks([0, 4, 8])
        sg.set_xticks([0, 6, 12, 18, 24])
        sg.set_xlabel("hours into the day")
        if j == 0:
            g.set_ylabel("glucose\n(mg/dL)")
            ins.set_ylabel("insulin given\n" + r"($\times$ basal)")
            sg.set_ylabel(r"noise level $\sigma$")
            g.text(0.2, 296, "meals", fontsize=15, color="#6e6e73", va="top")
            g.text(23.8, 50, "severe low (below 54)", fontsize=15, color="#b03a2e", ha="right", va="top")
            g.text(23.8, 176, "target range", fontsize=15, color="#3e7c4a", ha="right", va="top")
        (line_g,) = g.plot([], [], color="#222222", lw=2.4, zorder=4)
        (line_i,) = ins.plot([], [], color=COL[m], lw=2.4, zorder=4)
        (line_s,) = sg.plot([], [], color=COL[m], lw=2.0, zorder=4)
        cursors = [ax.axvline(0, color="#999999", lw=1) for ax in (g, ins, sg)]
        arts[m] = dict(line_g=line_g, line_i=line_i, line_s=line_s, cursors=cursors, ax_g=g,
                       ins=rolling_mean(d["delivered"], 10))
    if not slide:
        fig.text(0.075, 0.955, "One simulated day, same meals for both agents", fontsize=26, va="center")
    clock = fig.text(0.985, 0.955, "", fontsize=20 if slide else 24, va="center", ha="right", color="#444444")

    frames = []
    ends = list(range(steps_per_frame, n + 1, steps_per_frame))
    if ends[-1] != n:
        ends.append(n)
    for k in ends:
        for m in ("global", "state"):
            d, a = days[m], arts[m]
            a["line_g"].set_data(t[:k], d["bg_after"][:k])
            a["line_i"].set_data(t[:k], a["ins"][:k])
            a["line_s"].set_data(t[:k], d["sigma"][:k])
            for c in a["cursors"]:
                c.set_xdata([t[k - 1], t[k - 1]])
        hh = t[k - 1]
        clock.set_text(f"hour {int(hh):02d}:{int(round((hh % 1) * 60)) % 60:02d}")
        if k == n:
            for m in ("global", "state"):
                st = days[m]["stats"]
                txt = (f"time in range {100 * st['tir']:.0f}%,   lowest glucose {st['min_bg']:.0f} mg/dL"
                       + (",   severe low" if st["severe_low"] else ""))
                arts[m]["ax_g"].text(0.5, 1.03, txt, transform=arts[m]["ax_g"].transAxes, ha="center", va="bottom",
                                     fontsize=16 if slide else 19, color="#222222")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    frames += [frames[-1]] * int(3 * fps)  # hold the last frame
    if not slide:
        imageio.mimsave(path + ".mp4", frames, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    if gif or slide:
        from PIL import Image
        small = [Image.fromarray(f).resize((1280, 720), Image.LANCZOS).quantize(colors=64) for f in frames[::3]]
        # first frame = the finished day for 20 ms, so viewers that show only frame 1 (PDF export, previews) show
        # the whole day; in the loop it follows the held last frame, so it is invisible
        small = [small[-1]] + small
        small[0].save(path + ".gif", save_all=True, append_images=small[1:], loop=0, optimize=True,
                      duration=[20] + [int(3000 / fps)] * (len(small) - 1))
    if not slide:
        imageio.imwrite(path + "_end.png", frames[-1])
    plt.close(fig)


# ---- the policy map --------------------------------------------------------------------
BINS = np.array([40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 300])


def binned(x, y, fn):
    idx = np.digitize(x, BINS) - 1
    centers, vals = [], []
    for b in range(len(BINS) - 1):
        sel = idx == b
        if sel.sum() >= 20:
            centers.append(0.5 * (BINS[b] + BINS[b + 1]))
            vals.append(fn(y[sel]))
    return np.array(centers), np.array(vals)


def policy_map(pool, path):
    plot_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.8, bottom=0.17, wspace=0.3)
    for m in ("global", "state"):
        p = pool[m]
        c, med = binned(p["cgm"], dose(p["mu"]), np.median)
        _, q1 = binned(p["cgm"], dose(p["mu"]), lambda v: np.percentile(v, 25))
        _, q3 = binned(p["cgm"], dose(p["mu"]), lambda v: np.percentile(v, 75))
        axes[0].fill_between(c, q1, q3, color=COL[m], alpha=0.18, lw=0)
        axes[0].plot(c, med, "o-", color=COL[m], label=NAME[m])
        c, sg = binned(p["cgm"], p["sigma"], np.median)
        axes[1].plot(c, sg, "o-", color=COL[m])
        c, leak = binned(p["cgm"], (p["delivered"] > 1e-6).astype(float), np.mean)
        axes[2].plot(c, 100 * leak, "o-", color=COL[m])
    labels = [r"intended dose ($\times$ basal)", r"policy $\sigma$", r"steps with any insulin (\%)" if plt.rcParams["text.usetex"]
              else "steps with any insulin (%)"]
    for ax, lab, letter in zip(axes, labels, "abc"):
        plot_style.clean_axes(ax)
        ax.set_xlabel("glucose (mg/dL)")
        ax.set_ylabel(lab)
        ax.set_xlim(40, 300)
        ax.axvspan(70, 180, color="#dcefdc", lw=0, zorder=0)
        ax.text(0.0, 1.04, rf"$\mathbf{{({letter})}}$", transform=ax.transAxes, ha="left", va="bottom")
    axes[0].set_ylim(-0.3, 10.3)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.005, 10)
    axes[1].set_yticks([0.01, 0.1, 1, 10])
    axes[1].set_yticklabels(["0.01", "0.1", "1", "10"])
    axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].set_ylim(-3, 103)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    plot_style.save_checked(fig, path, dpi=200)
    plt.close(fig)


def statements(pool):
    out = {}
    for m in ("global", "state"):
        p = pool[m]
        d_int = dose(p["mu"])
        low, mid = p["cgm"] < 80, (p["cgm"] >= 120) & (p["cgm"] <= 180)
        sat_low = p["mu"] <= -1.0
        inside = (p["mu"] > -1.0) & (p["mu"] < 1.0)
        out[m] = {
            "n_steps": int(len(p["cgm"])),
            "spearman_dose_vs_glucose": float(spearmanr(p["cgm"], d_int)[0]),
            "spearman_dose_vs_trend": float(spearmanr(p["trend"], d_int)[0]),
            "intended_dose_median_below80": float(np.median(d_int[low])) if low.any() else None,
            "intended_dose_median_120_180": float(np.median(d_int[mid])),
            "frac_intended_zero_below80": float(np.mean(sat_low[low])) if low.any() else None,
            "frac_any_insulin_delivered_below80": float(np.mean(p["delivered"][low] > 1e-6)) if low.any() else None,
            "mean_delivered_below80": float(np.mean(p["delivered"][low])) if low.any() else None,
            "mean_delivered_120_180": float(np.mean(p["delivered"][mid])),
            "sigma_median_below80": float(np.median(p["sigma"][low])) if low.any() else None,
            "sigma_median_120_180": float(np.median(p["sigma"][mid])),
            "spearman_sigma_vs_glucose": float(spearmanr(p["cgm"], p["sigma"])[0]) if p["sigma"].std() > 0 else None,
            "spearman_sigma_vs_trend": float(spearmanr(p["trend"], p["sigma"])[0]) if p["sigma"].std() > 0 else None,
            "sigma_median_mean_pinned_at_zero": float(np.median(p["sigma"][sat_low])) if sat_low.any() else None,
            "sigma_median_mean_inside_range": float(np.median(p["sigma"][inside])) if inside.any() else None,
            "frac_steps_mean_outside_range": float(np.mean(~inside)),
            "frac_steps_below80": float(np.mean(low)),
        }
    return out


def main():
    a = tyro.cli(Args)
    os.makedirs(a.out_dir, exist_ok=True)
    env = make_env(ENV)
    pols = {m: load_run(r)[:2] for m, r in (("global", a.global_run), ("state", a.state_run))}
    seeds = list(range(a.seed_base, a.seed_base + a.seeds))
    days = {m: {s: run_day(*pols[m], env, s, a.noise_seed + s) for s in seeds} for m in pols}
    per_day = {s: {m: days[m][s]["stats"] for m in pols} for s in seeds}

    pool = {m: {k: np.concatenate([days[m][s][k] for s in seeds]) for k in ("cgm", "trend", "mu", "sigma", "delivered")}
            for m in pols}
    policy_map(pool, os.path.join(a.out_dir, "policy_map.png"))
    summ = {"days": per_day, "statements": statements(pool),
            "per_policy": {m: {"days_meeting_targets": float(np.mean([days[m][s]["stats"]["success"] for s in seeds])),
                               "days_with_severe_low": float(np.mean([days[m][s]["stats"]["severe_low"] for s in seeds])),
                               "mean_time_in_range": float(np.mean([days[m][s]["stats"]["tir"] for s in seeds]))}
                           for m in pols},
            "global_run": a.global_run, "state_run": a.state_run}

    if a.days_to_render == "auto":
        # the day where sigma(s) goes lowest (relative to global sigma), and the most typical day
        gap = {s: days["global"][s]["stats"]["min_bg"] - days["state"][s]["stats"]["min_bg"] for s in seeds}
        worst = max(gap, key=gap.get)
        tir_both = {s: days["global"][s]["stats"]["tir"] + days["state"][s]["stats"]["tir"] for s in seeds if s != worst}
        med = sorted(tir_both, key=tir_both.get)[len(tir_both) // 2]
        render = [med, worst]
    else:
        render = [int(s) for s in a.days_to_render.split(",")]
    summ["rendered_days"] = render
    for s in render:
        animate({m: days[m][s] for m in pols}, s, os.path.join(a.out_dir, f"day_{s}"), a.steps_per_frame, a.fps, a.gif)
    # a compact version of the first (typical) day for the slides
    animate({m: days[m][render[0]] for m in pols}, render[0], os.path.join(a.out_dir, f"day_{render[0]}_slide"),
            a.steps_per_frame, a.fps, True, slide=True)
    with open(os.path.join(a.out_dir, "interpretability.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({"rendered": render, "per_policy": summ["per_policy"], "statements": summ["statements"]}, indent=2))


if __name__ == "__main__":
    main()
