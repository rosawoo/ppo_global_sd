"""Launch / resume the (env x std_mode x seed) grid.

One run per cell, written to {runs_root}/{env}/{mode}/seed{N}/ by ppo.py. Resumable:
cells whose status.json says "done" or "nan" are skipped (NaN runs are a result, never
retried); anything else is deleted and re-run from step 0 (DECISIONS.md D13).

    python scripts/sweep.py --smoke                       # 1 seed, 50k steps, core envs, both modes
    python scripts/sweep.py --workers 8                   # full sweep, 10 seeds, core envs
    python scripts/sweep.py --envs parking-v0 --runs-root runs_abl/ent0.01 --extra "--ent-coef 0.01"
    python scripts/sweep.py --emit-tasks cluster/tasks/main.txt   # one command per line, for Slurm arrays
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
    "simglucose-adolescent1-v0": {"num_envs": 8, "num_steps": 256, "total_timesteps": 1_000_000},
    # optional: action-parameterisation ablation of the third domain, and the former third env
    "simglucose-adolescent1-logact-v0": {"num_envs": 8, "num_steps": 256, "total_timesteps": 1_000_000},
    "LunarLanderContinuous-v3": {"num_envs": 8, "num_steps": 256, "total_timesteps": 1_000_000},
    "parking-nowalls-v0": {"num_envs": 8, "num_steps": 256, "total_timesteps": 2_000_000},
}
# the three domains of the study (project goal); the others run only when asked for
CORE_ENVS = ["parking-v0", "Pendulum-v1", "simglucose-adolescent1-v0"]


@dataclass
class SweepArgs:
    envs: list[str] = field(default_factory=lambda: list(CORE_ENVS))
    """environments to run (default: the three core domains)"""
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
    emit_tasks: str = ""
    """write the runs still to do to this file (one JSON task per line, for a Slurm job array:
    cluster/array.sbatch runs line $SLURM_ARRAY_TASK_ID via scripts/run_task.py) and exit"""
    seed_offset: int = 0
    """run seeds seed_offset+1 .. seed_offset+seeds (to add seeds to an existing sweep)"""
    config: str = ""
    """JSON written by scripts/tune.py export: tuned per-env hyper-parameters added to every run"""
    config_key: str = "shared"
    """with --config: 'shared' (one config per env, both modes identical except --std-mode, as the
    spec requires) or 'per_mode' (each mode at its own best tuned config)"""


def status_of(run_dir: str) -> str:
    try:
        with open(os.path.join(run_dir, "status.json")) as f:
            return json.load(f)["status"]
    except (OSError, ValueError, KeyError):
        return "missing"


# variants that reuse the tuned config of the env they are derived from
TUNE_ALIASES = {"parking-nowalls-v0": "parking-v0", "simglucose-adolescent1-logact-v0": "simglucose-adolescent1-v0"}


def tuned_overrides(a: SweepArgs, env: str, mode: str) -> dict:
    if not a.config:
        return {}
    with open(a.config if os.path.isabs(a.config) else os.path.join(ROOT, a.config)) as f:
        conf = json.load(f)
    env = env if env in conf else TUNE_ALIASES.get(env, env)
    if env not in conf:
        print(f"warning: {env} not in {a.config}; using the default config", file=sys.stderr)
        return {}
    return dict(conf[env]["shared"] if a.config_key == "shared" else conf[env]["per_mode"][mode])


def command(a: SweepArgs, env: str, mode: str, seed: int) -> list[str]:
    cfg = dict(ENV_CONFIGS[env])
    cfg.update(tuned_overrides(a, env, mode))
    if a.total_timesteps:
        cfg["total_timesteps"] = a.total_timesteps
    cmd = [sys.executable, "-u", "ppo.py", "--env-id", env, "--seed", str(seed), "--runs-root", a.runs_root]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}={v!r}" if isinstance(v, float) else f"--{k.replace('_', '-')}={v}"]
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
    assert a.config_key in ("shared", "per_mode"), a.config_key

    # longest jobs first; seeds outermost so partial sweeps stay balanced across modes
    order = sorted(a.envs, key=lambda e: -(a.total_timesteps or ENV_CONFIGS[e]["total_timesteps"]))
    seeds = range(a.seed_offset + 1, a.seed_offset + a.seeds + 1)
    cells = [(env, mode, seed) for env in order for seed in seeds for mode in a.modes]
    todo, skipped = [], []
    for cell in cells:
        st = status_of(os.path.join(ROOT, a.runs_root, cell[0], cell[1], f"seed{cell[2]}"))
        (skipped if st in ("done", "nan") else todo).append(cell)
    print(f"{len(cells)} cells: {len(skipped)} finished (skipped), {len(todo)} to run, {a.workers} workers")
    if a.dry_run:
        for cell in todo:
            print(" ".join(shlex.quote(c) for c in command(a, *cell)))
        sys.exit(0)
    if a.emit_tasks:
        os.makedirs(os.path.dirname(os.path.abspath(a.emit_tasks)), exist_ok=True)
        with open(a.emit_tasks, "w") as f:
            for env, mode, seed in todo:
                f.write(json.dumps({"kind": "train", "env": env, "mode": mode, "seed": seed, "runs_root": a.runs_root,
                                    "extra": a.extra, "total_timesteps": a.total_timesteps, "config": a.config,
                                    "config_key": a.config_key}) + "\n")
        print(f"wrote {len(todo)} tasks to {a.emit_tasks}")
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
