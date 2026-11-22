"""Rendered rollouts with the current sigma (ellipse / bar) and mean action drawn on each frame.

    python analysis/demos.py --runs-root runs --envs parking-v0 --seed 1

For each env / mode writes `episodes` MP4s under demos/{env}/{mode}/. Episodes use the
stochastic policy on consecutive reset seeds starting at --seed-base; if none of the
first `episodes` is a failure, further seeds are scanned (up to --max-scan) and the
first failure found replaces the last clip, so at least one failure case is included
whenever one exists. File names carry the outcome.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw

from common import MODE_COLORS
from envs.wrappers import ENV_SPECS, make_env
from scripts.eval import load_run, rollout

PANEL = 132  # px, square overlay panel


@dataclass
class DemoArgs:
    runs_root: str = "runs"
    out_dir: str = "demos"
    envs: list[str] = field(default_factory=lambda: list(ENV_SPECS))
    modes: list[str] = field(default_factory=lambda: ["global", "state"])
    seed: int = 1
    """training seed of the run to render"""
    episodes: int = 5
    seed_base: int = 200000
    """first env reset seed (disjoint from the eval seed set)"""
    max_scan: int = 60
    """max episodes to scan when looking for a failure case"""
    fps: int = 10
    gif: bool = False
    """also write GIFs"""


def is_failure(env_id, stats):
    if not np.isnan(stats["success"]):
        return stats["success"] < 0.5
    spec = ENV_SPECS[env_id]
    return stats["return"] < spec.learned_threshold


def draw_overlay(frame, mean, std, low, high, action_names, color, caption):
    """Top-left panel: the action box [low, high]; a dot at the mean action (clipped to the box, as the env clips it) and a 1-sigma
    ellipse (2-D actions) or 1-sigma bar (1-D). Sigma is drawn in action units, to scale."""
    img = Image.fromarray(frame).convert("RGB")
    if img.width < 360:  # tiny renders (e.g. Pendulum 500px is fine; keep text legible)
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, s = 8, 8, PANEL
    d.rectangle([x0, y0, x0 + s, y0 + s + 34], fill=(252, 252, 251, 225), outline=(195, 194, 183, 255))
    pad = 14
    bx0, by0, bx1, by1 = x0 + pad, y0 + pad, x0 + s - pad, y0 + s - pad
    to_px = lambda v, lo, hi, a, b: a + (np.clip(v, lo, hi) - lo) / (hi - lo) * (b - a)
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    if len(mean) >= 2:
        d.rectangle([bx0, by0, bx1, by1], outline=(137, 135, 129, 255))
        d.line([(bx0 + bx1) / 2, by0, (bx0 + bx1) / 2, by1], fill=(225, 224, 217, 255))
        d.line([bx0, (by0 + by1) / 2, bx1, (by0 + by1) / 2], fill=(225, 224, 217, 255))
        cx = to_px(mean[1], low[1], high[1], bx0, bx1)  # x: action dim 1, y: action dim 0 (up = positive)
        cy = to_px(mean[0], low[0], high[0], by1, by0)
        rx = std[1] / (high[1] - low[1]) * (bx1 - bx0)
        ry = std[0] / (high[0] - low[0]) * (by1 - by0)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rgb + (70,), outline=rgb + (255,), width=2)
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(11, 11, 11, 255))
        d.text((bx0, y0 + 1), f"y:{action_names[0][:5]} x:{action_names[1][:5]}", fill=(82, 81, 78, 255))
    else:
        mid = (by0 + by1) / 2
        d.line([bx0, mid, bx1, mid], fill=(137, 135, 129, 255), width=1)
        cx = to_px(mean[0], low[0], high[0], bx0, bx1)
        rx = std[0] / (high[0] - low[0]) * (bx1 - bx0)
        left = float(np.clip(cx - rx, x0 + 2, x0 + s - 4))
        right = max(left + 2, float(np.clip(cx + rx, x0 + 2, x0 + s - 2)))
        d.rectangle([left, mid - 9, right, mid + 9], fill=rgb + (90,),
                    outline=rgb + (255,), width=2)
        d.ellipse([cx - 3, mid - 3, cx + 3, mid + 3], fill=(11, 11, 11, 255))
        d.text((bx0, y0 + 1), f"{action_names[0]} (mean +- sigma)", fill=(82, 81, 78, 255))
    sig = " ".join(f"{v:.2f}" for v in std)
    mu = " ".join(f"{v:+.2f}" for v in mean)
    d.text((x0 + 6, y0 + s - 2), f"sigma = {sig}", fill=(11, 11, 11, 255))
    d.text((x0 + 6, y0 + s + 14), f"mean  = {mu}", fill=(82, 81, 78, 255))
    tw = d.textlength(caption)
    d.rectangle([x0 + s + 6, y0, x0 + s + 14 + tw, y0 + 14], fill=(252, 252, 251, 225))
    d.text((x0 + s + 10, y0 + 2), caption, fill=(11, 11, 11, 255))
    return np.asarray(img)


def record(env_id, mode, agent, normalizer, seed, color):
    env = make_env(env_id, render_mode="rgb_array")
    low, high = env.unwrapped.action_space.low, env.unwrapped.action_space.high
    names = ENV_SPECS[env_id].action_names
    frames = []

    def cb(env_, info, steps):
        caption = f"{mode} sigma | seed {seed} | t={len(steps['mean']) - 1} | dist={float(info['goal_dist']):.2f}"
        frames.append(draw_overlay(env_.render(), steps["mean"][-1], steps["std"][-1], low, high, names, color, caption))

    _, stats = rollout(agent, normalizer, env, seed, False, torch.Generator().manual_seed(seed), frame_callback=cb)
    env.close()
    return frames, stats


def main(a: DemoArgs):
    for env_id in a.envs:
        for mode in a.modes:
            run_dir = os.path.join(a.runs_root, env_id, mode, f"seed{a.seed}")
            if not os.path.exists(os.path.join(run_dir, "checkpoint.pt")):
                print("skip (no checkpoint):", run_dir)
                continue
            agent, normalizer, _ = load_run(run_dir)
            color = MODE_COLORS[mode]
            clips = [record(env_id, mode, agent, normalizer, a.seed_base + i, color) + (a.seed_base + i,)
                     for i in range(a.episodes)]
            if not any(is_failure(env_id, st) for _, st, _ in clips):
                for i in range(a.episodes, a.max_scan):  # look for one failure case
                    frames, st = record(env_id, mode, agent, normalizer, a.seed_base + i, color)
                    if is_failure(env_id, st):
                        clips[-1] = (frames, st, a.seed_base + i)
                        break
                else:
                    print(f"{env_id}/{mode}: no failure in {a.max_scan} episodes")
            out_dir = os.path.join(a.out_dir, env_id, mode)
            os.makedirs(out_dir, exist_ok=True)
            for frames, st, seed in clips:
                tag = "fail" if is_failure(env_id, st) else "ok"
                path = os.path.join(out_dir, f"trainseed{a.seed}_ep{seed}_{tag}_ret{st['return']:.0f}.mp4")
                h, w = frames[0].shape[:2]
                frames = [f[: h - h % 2, : w - w % 2] for f in frames]  # even dims for h264
                imageio.mimsave(path, frames, fps=a.fps, macro_block_size=1)
                if a.gif:
                    imageio.mimsave(path[:-4] + ".gif", frames[::2], duration=2 / a.fps)
                print("wrote", path)


if __name__ == "__main__":
    main(tyro.cli(DemoArgs))
