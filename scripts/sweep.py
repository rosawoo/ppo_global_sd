"""Launch / resume the (env x std_mode x seed) grid.

One run per cell, written to {runs_root}/{env}/{mode}/seed{N}/ by ppo.py. Resumable:
cells whose status.json says "done" or "nan" are skipped (NaN runs are a result, never
retried); anything else is deleted and re-run from step 0 (DECISIONS.md D13).

    python scripts/sweep.py --smoke                       # 1 seed, 50k steps, all envs, both modes
    python scripts/sweep.py --workers 8                   # full sweep, 10 seeds
    python scripts/sweep.py --envs parking-v0 --runs-root runs_abl/ent0.01 --extra "--ent-coef 0.01"
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import tyro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The only per-env settings (CLAUDE.md §3.2). Everything else is the single shared config in ppo.Args.
ENV_CONFIGS = {
    "parking-v0": {"num_envs": 8, "num_steps": 256, "total_timesteps": 2_000_000},
    "Pendulum-v1": {"num_envs": 4, "num_steps": 512, "total_timesteps": 500_000},
    "LunarLanderContinuous-v3": {"num_envs": 8, "num_steps": 256, "total_timesteps": 1_000_000},
}


@dataclass
class SweepArgs:
    envs: list[str] = field(default_factory=lambda: list(ENV_CONFIGS))
    """environments to run"""
    modes: list[str] = field(default_factory=lambda: ["global", "state"])
    """std modes to run"""
    seeds: int = 10
    """seeds 1..N per (env, mode)"""
    workers: int = 8
    """parallel processes (each is single-threaded)"""
    runs_root: str = "runs"
    """output root"""
    smoke: bool = False
    """smoke test: 1 seed, 50k steps, runs_root=runs_smoke (unless overridden)"""
    total_timesteps: int = 0
    """override total steps for every env (0 = per-env default)"""
    extra: str = ""
    """extra CLI args passed verbatim to ppo.py for BOTH modes (ablations)"""
    dry_run: bool = False
    """print the commands without running"""


def status_of(run_dir: str) -> str:
    try:
        with open(os.path.join(run_dir, "status.json")) as f:
            return json.load(f)["status"]
    except (OSError, ValueError, KeyError):
        return "missing"


def command(a: SweepArgs, env: str, mode: str, seed: int) -> list[str]:
    cfg = dict(ENV_CONFIGS[env])
    if a.total_timesteps:
        cfg["total_timesteps"] = a.total_timesteps
    cmd = [sys.executable, "-u", "ppo.py", "--env-id", env, "--seed", str(seed), "--runs-root", a.runs_root]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    cmd += shlex.split(a.extra)
    # --std-mode is the ONLY difference between the two arms.
    return cmd + ["--std-mode", mode]


def run_one(a: SweepArgs, env: str, mode: str, seed: int) -> tuple[str, str, float]:
    run_dir = os.path.join(ROOT, a.runs_root, env, mode, f"seed{seed}")
    shutil.rmtree(run_dir, ignore_errors=True)  # incomplete run -> restart from scratch
    os.makedirs(run_dir)
    t0 = time.time()
    with open(os.path.join(run_dir, "stdout.log"), "w") as log:
        proc = subprocess.run(command(a, env, mode, seed), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status = status_of(run_dir)
    if proc.returncode != 0 and status not in ("done", "nan"):
        status = f"crashed(rc={proc.returncode})"
    return f"{env}/{mode}/seed{seed}", status, time.time() - t0


if __name__ == "__main__":
    a = tyro.cli(SweepArgs)
    if a.smoke:
        a.seeds = 1
        a.total_timesteps = a.total_timesteps or 50_000
        if a.runs_root == "runs":
            a.runs_root = "runs_smoke"
    for env in a.envs:
        assert env in ENV_CONFIGS, f"unknown env {env}; known: {list(ENV_CONFIGS)}"

    # longest jobs first; seeds outermost so partial sweeps stay balanced across modes
    order = sorted(a.envs, key=lambda e: -(a.total_timesteps or ENV_CONFIGS[e]["total_timesteps"]))
    cells = [(env, mode, seed) for env in order for seed in range(1, a.seeds + 1) for mode in a.modes]
    todo, skipped = [], []
    for cell in cells:
        st = status_of(os.path.join(ROOT, a.runs_root, cell[0], cell[1], f"seed{cell[2]}"))
        (skipped if st in ("done", "nan") else todo).append(cell)
    print(f"{len(cells)} cells: {len(skipped)} finished (skipped), {len(todo)} to run, {a.workers} workers")
    if a.dry_run:
        for cell in todo:
            print(" ".join(shlex.quote(c) for c in command(a, *cell)))
        sys.exit(0)

    t0, results = time.time(), []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_one, a, *cell) for cell in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            name, status, secs = fut.result()
            results.append((name, status, secs))
            print(f"[{i}/{len(todo)}] {name}: {status} in {secs / 60:.1f} min "
                  f"(sweep elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    bad = [r for r in results if r[1] != "done"]
    print(f"sweep finished in {(time.time() - t0) / 60:.1f} min; {len(bad)} not 'done': {[(n, s) for n, s, _ in bad]}")
