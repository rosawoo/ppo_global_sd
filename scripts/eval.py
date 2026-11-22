"""Periodic evaluation + raw (obs, goal-relative state, mu, sigma, action) dump.

Used in two ways:
* imported by ppo.py, which calls `evaluate(...)` every `--eval-freq` steps;
* as a CLI to re-evaluate a saved checkpoint:
    python scripts/eval.py --run-dir runs/Pendulum-v1/state/seed1 --episodes 20

Evaluation never touches the global torch / numpy RNG streams (it uses its own
torch.Generator and explicit env reset seeds), so training is bit-identical regardless
of how often evaluation runs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# optional per-episode metrics an env may report in its final info (Simglucose: time in
# range etc.); copied into the episode stats and averaged into the eval summary when present
EPISODE_METRIC_KEYS = ("tir", "tbr", "tbr54", "tar", "mean_bg", "mean_risk")


@torch.no_grad()
def rollout(agent, normalizer, env, seed: int, deterministic: bool, generator: torch.Generator, device="cpu",
            frame_callback=None):
    """One episode. Returns per-step arrays and episode-level stats."""
    obs, info = env.reset(seed=seed)
    steps = {k: [] for k in ("obs", "goal_rel", "goal_dist", "mean", "std", "action", "reward")}
    ep_return, length, done = 0.0, 0, False
    while not done:
        x = torch.as_tensor(normalizer.obs(obs[None], update=False), device=device)
        dist, _ = agent.dist(x)
        mean, std = dist.mean[0], dist.stddev[0]
        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn(mean.shape, generator=generator).to(device)
        action_np = action.cpu().numpy()
        steps["obs"].append(obs)
        steps["goal_rel"].append(info["goal_rel"])
        steps["goal_dist"].append(info["goal_dist"])
        steps["mean"].append(mean.cpu().numpy())
        steps["std"].append(std.cpu().numpy())
        steps["action"].append(action_np)  # un-clipped; the env wrapper clips
        if frame_callback is not None:
            frame_callback(env, info, steps)
        obs, reward, terminated, truncated, info = env.step(action_np)
        steps["reward"].append(reward)
        ep_return += float(reward)
        length += 1
        done = terminated or truncated
    stats = {
        "return": ep_return,
        "length": length,
        "success": float(info["is_success"]) if "is_success" in info else np.nan,
        "crashed": float(info["crashed"]) if "crashed" in info else np.nan,
        "terminated": float(terminated),
        "final_goal_dist": float(info["goal_dist"]),
        "seed": seed,
    }
    for k in EPISODE_METRIC_KEYS:
        if k in info:
            stats[k] = float(info[k])
    return {k: np.asarray(v, dtype=np.float32) for k, v in steps.items()}, stats


def evaluate(agent, normalizer, env, n_episodes: int, seed_base: int, noise_seed: int, out_path: str | None = None,
             device="cpu") -> dict:
    """n_episodes with the stochastic policy and n_episodes with the mean policy, on the
    same fixed reset seeds [seed_base, seed_base + n_episodes). Optionally dumps a .npz."""
    was_training = agent.training
    agent.eval()
    generator = torch.Generator().manual_seed(noise_seed)
    step_chunks, ep_rows, summary = [], [], {}
    ep_id = 0
    for deterministic in (False, True):
        tag = "det" if deterministic else "stoch"
        rows = []
        for i in range(n_episodes):
            steps, stats = rollout(agent, normalizer, env, seed_base + i, deterministic, generator, device)
            n = stats["length"]
            steps["episode_id"] = np.full(n, ep_id, dtype=np.int32)
            steps["t"] = np.arange(n, dtype=np.int32)
            steps["deterministic"] = np.full(n, deterministic, dtype=bool)
            step_chunks.append(steps)
            stats["deterministic"] = float(deterministic)
            rows.append(stats)
            ep_id += 1
        ep_rows += rows
        summary[f"{tag}_return_mean"] = float(np.mean([r["return"] for r in rows]))
        summary[f"{tag}_return_std"] = float(np.std([r["return"] for r in rows]))
        summary[f"{tag}_length_mean"] = float(np.mean([r["length"] for r in rows]))
        summary[f"{tag}_success_rate"] = float(np.mean([r["success"] for r in rows]))
        summary[f"{tag}_crash_rate"] = float(np.mean([r["crashed"] for r in rows]))
        summary[f"{tag}_final_goal_dist_mean"] = float(np.mean([r["final_goal_dist"] for r in rows]))
        for k in EPISODE_METRIC_KEYS:
            if k in rows[0]:
                summary[f"{tag}_{k}_mean"] = float(np.mean([r[k] for r in rows]))
    data = {k: np.concatenate([c[k] for c in step_chunks]) for k in step_chunks[0]}
    stoch = ~data["deterministic"]
    summary["stoch_sigma_mean"] = float(data["std"][stoch].mean())
    summary["stoch_logsigma_state_std"] = float(np.log(data["std"][stoch]).std(axis=0).mean())
    if out_path is not None:
        for k in ep_rows[0]:
            data[f"ep_{k}"] = np.asarray([r[k] for r in ep_rows])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **data)
    agent.train(was_training)
    return summary


def load_run(run_dir: str, device="cpu"):
    """Rebuild (agent, normalizer, args-dict) from `run_dir/checkpoint.pt`."""
    from envs.wrappers import VecNormalizer
    from ppo import Agent

    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device, weights_only=False)
    a = ckpt["args"]
    agent = Agent(ckpt["obs_dim"], ckpt["act_dim"], a["std_mode"], a["log_std_init"], a["logstd_min"], a["logstd_max"],
                  a["logstd_head_init_std"], a.get("actor_hidden", 64), a.get("critic_hidden", 256)).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, VecNormalizer.from_state_dict(ckpt["normalizer"]), a


if __name__ == "__main__":
    import json
    from dataclasses import dataclass

    import tyro

    from envs.wrappers import make_env

    @dataclass
    class EvalArgs:
        run_dir: str
        """run directory containing checkpoint.pt"""
        episodes: int = 20
        """episodes per policy type (stochastic and deterministic)"""
        seed_base: int = 100_000
        """first env reset seed of the fixed evaluation seed set"""
        noise_seed: int = 0
        """seed of the action-noise generator for the stochastic episodes"""
        out: str = ""
        """npz output path (default: {run_dir}/evals/final_recheck.npz)"""

    ea = tyro.cli(EvalArgs)
    agent, normalizer, a = load_run(ea.run_dir)
    env = make_env(a["env_id"])
    out = ea.out or os.path.join(ea.run_dir, "evals", "final_recheck.npz")
    print(json.dumps(evaluate(agent, normalizer, env, ea.episodes, ea.seed_base, ea.noise_seed, out), indent=2))
    print("wrote", out)
