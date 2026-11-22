"""Run one line of a task file (one JSON object per line). Used by cluster/array.sbatch:

    python scripts/run_task.py cluster/tasks/train_parking-v0.jsonl $SLURM_ARRAY_TASK_ID

kind "train": one (env, std_mode, seed) PPO run via scripts/sweep.run_one. A run whose
  status.json says "done" or "nan" is skipped; anything else (never started, killed,
  pre-empted and requeued) is deleted and restarted from step 0 (DECISIONS.md D13), so a
  task can be re-submitted or requeued safely.
kind "probe": the noise-tolerance probe of one finished run (analysis/probe.py --run-dir).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(path: str, index: int) -> int:
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    if not 1 <= index <= len(lines):
        print(f"task index {index} out of range 1..{len(lines)} in {path}")
        return 2
    t = json.loads(lines[index - 1])
    print(f"task {index}/{len(lines)} from {path}: {t}", flush=True)
    if t["kind"] == "train":
        from scripts.sweep import SweepArgs, run_one, status_of

        a = SweepArgs(runs_root=t["runs_root"], extra=t.get("extra", ""), total_timesteps=t.get("total_timesteps", 0),
                      config=t.get("config", ""), config_key=t.get("config_key", "shared"))
        run_dir = os.path.join(ROOT, a.runs_root, t["env"], t["mode"], f"seed{t['seed']}")
        st = status_of(run_dir)
        if st in ("done", "nan"):
            print(f"{run_dir}: already '{st}', nothing to do")
            return 0
        name, status, secs = run_one(a, t["env"], t["mode"], t["seed"])
        print(f"{name}: {status} in {secs / 60:.1f} min", flush=True)
        return 0 if status in ("done", "nan") else 1
    if t["kind"] == "probe":
        run_dir = t["run_dir"]
        if not os.path.exists(os.path.join(ROOT, run_dir, "checkpoint.pt")):
            print(f"{run_dir}: no checkpoint.pt (training not finished?)")
            return 1
        cmd = [sys.executable, "analysis/probe.py", "--run-dir", run_dir] + shlex.split(t.get("extra", ""))
        return subprocess.call(cmd, cwd=ROOT)
    print(f"unknown task kind {t['kind']!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
