"""Hyperparameter tuning: one *paired* Optuna study per environment.

Every trial trains BOTH σ parameterisations (global and state) with the SAME sampled
hyper-parameters on the same tuning seeds, so each trial is a paired comparison. A study yields
  * the best shared config: both arms identical except --std-mode, as the spec requires;
  * each arm's own best config among the configs tried;
  * the global-vs-state difference across the whole explored region, which tests whether one
    parameterisation is *consistently* better rather than better at one hand-picked config.

Tuning never touches the final protocol: tuning seeds (101, 102) and the eval reset seeds
(500000+) are disjoint from the final runs (seeds 1-10, eval seeds 100000+), and the final
comparison is re-run from scratch at the exported configs (DECISIONS.md D39).

    python scripts/tune.py init   --env parking-v0            # create the study, enqueue known-good configs
    python scripts/tune.py worker --env parking-v0 --n-trials 5 --jobs 4   # one worker (= one Slurm array task)
    python scripts/tune.py export                             # -> configs/tuned.json
    python scripts/tune.py report                             # -> results/tuning_summary.md, figures/tuning/

Search space (TPE, multivariate, constant-liar for parallel workers; median pruning on a
smoothed intermediate score after 30% of training): learning rate, discount, GAE lambda,
clip range, entropy bonus, epochs, minibatch size, rollout length, initial sigma. Ranges follow
CleanRL / SB3 defaults, the RL-Zoo's tuned PPO settings and Andrychowicz et al. (2021); the
discount and rollout-length ranges depend on the env's time scale.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import tyro

from scripts.sweep import ENV_CONFIGS

MODES = ("global", "state")
TUNE_SEEDS = (101, 102)
TUNE_EVAL_SEED_BASE = 500000

# per-env pieces of the search space (the time scale of each task differs)
GAMMAS = {
    "Pendulum-v1": [0.9, 0.95, 0.98, 0.99],  # dense, short-horizon cost; the RL-Zoo's tuned PPO uses 0.9
    "parking-v0": [0.95, 0.98, 0.99, 0.995],  # 5 Hz; parking takes ~5-30 s = 25-150 steps
    "parking-nowalls-v0": [0.95, 0.98, 0.99, 0.995],
    "simglucose-adolescent1-v0": [0.98, 0.99, 0.995, 0.999],  # 3-min steps; insulin acts over 2-6 h = 40-120 steps
    "simglucose-adolescent1-logact-v0": [0.98, 0.99, 0.995, 0.999],
    "LunarLanderContinuous-v3": [0.98, 0.99, 0.995, 0.999],
}
NUM_STEPS = {
    "Pendulum-v1": [256, 512, 1024, 2048],  # x 4 envs
    "parking-v0": [128, 256, 512],  # x 8 envs
    "parking-nowalls-v0": [128, 256, 512],
    "simglucose-adolescent1-v0": [128, 256, 512],
    "simglucose-adolescent1-logact-v0": [128, 256, 512],
    "LunarLanderContinuous-v3": [128, 256, 512],
}
# score of one run = mean of this eval metric over the last 25% of evaluations
OBJECTIVE = {
    "parking-v0": "parking", "parking-nowalls-v0": "parking",  # success rate, ties broken by final distance
    "Pendulum-v1": "stoch_return_mean", "simglucose-adolescent1-v0": "stoch_return_mean",
    "simglucose-adolescent1-logact-v0": "stoch_return_mean", "LunarLanderContinuous-v3": "stoch_return_mean",
}
SIGMA_INITS = [0.25, 0.5, 1.0]


def sample(trial, env):
    """Hyper-parameters in ppo.py Args names (+ 'minibatch_size', converted in to_args)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True),
        "gamma": trial.suggest_categorical("gamma", GAMMAS[env]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.95, 0.98, 0.99]),
        "clip_coef": trial.suggest_categorical("clip_coef", [0.1, 0.2, 0.25, 0.3]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 1e-4, 1e-3, 1e-2]),
        "update_epochs": trial.suggest_categorical("update_epochs", [3, 5, 10, 20]),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256, 512]),
        "num_steps": trial.suggest_categorical("num_steps", NUM_STEPS[env]),
        "sigma_init": trial.suggest_categorical("sigma_init", SIGMA_INITS),
    }


def to_args(env, params):
    """Sampled params -> ppo.py Args overrides (minibatch_size -> num_minibatches, sigma -> log sigma)."""
    p = dict(params)
    batch = ENV_CONFIGS[env]["num_envs"] * int(p["num_steps"])
    p["num_minibatches"] = max(1, batch // int(p.pop("minibatch_size")))
    p["log_std_init"] = math.log(float(p.pop("sigma_init")))
    return p


def known_good(env):
    """Configs enqueued as the first trials: the spec's shared default and published tuned ones."""
    default_steps = ENV_CONFIGS[env]["num_steps"]
    batch = ENV_CONFIGS[env]["num_envs"] * default_steps
    spec = {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "ent_coef": 0.0,
            "update_epochs": 10, "minibatch_size": batch // 32, "num_steps": default_steps, "sigma_init": 0.5}
    out = [spec]
    # Andrychowicz et al. 2021 recommendations: lambda 0.9, clip 0.25
    out.append({**spec, "gae_lambda": 0.9, "clip_coef": 0.25})
    if env == "Pendulum-v1":  # RL-Zoo tuned PPO (without gSDE)
        out.append({**spec, "learning_rate": 1e-3, "gamma": 0.9, "num_steps": 1024, "minibatch_size": 64})
    return out


@dataclass
class TuneArgs:
    command: tyro.conf.Positional[str]
    """init | worker | export | report"""
    env: str = ""
    """environment (init / worker)"""
    study_dir: str = "tuning"
    """journal files and trial runs live here"""
    n_trials: int = 4
    """worker: trials this worker runs before exiting"""
    jobs: int = 4
    """worker: concurrent training processes (2 modes x 2 seeds = 4 per trial)"""
    seeds: list[int] = field(default_factory=lambda: list(TUNE_SEEDS))
    eval_episodes: int = 10
    """eval episodes per policy type during tuning (final runs use 20)"""
    total_timesteps: int = 0
    """override the env's training budget (0 = the same budget as the final runs; for tests)"""
    sampler_seed: int = -1
    """TPE seed (-1: derived from SLURM_ARRAY_TASK_ID / pid so parallel workers differ)"""
    poll_seconds: float = 20.0
    out: str = "configs/tuned.json"
    """export: output file"""
    out_dir: str = "results"
    fig_dir: str = "figures/tuning"
    startup_trials: int = 10
    """TPE random start-up trials (also used for the 'random configs' paired comparison)"""
    require: list[str] = field(default_factory=list)
    """export: fail (exit 1, nothing written) unless every one of these envs has completed trials"""


# ---------------------------------------------------------------------------------------
def storage(a: TuneArgs, env: str):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    os.makedirs(a.study_dir, exist_ok=True)
    path = os.path.join(a.study_dir, f"{env}.journal")
    return optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def load_study(a: TuneArgs, env: str, sampler=None, pruner=None):
    import optuna

    return optuna.create_study(study_name=env, storage=storage(a, env), direction="maximize",
                               sampler=sampler, pruner=pruner, load_if_exists=True)


def run_score(evals, env, tail=0.25):
    """Score of one run from its evals.csv DataFrame (and the smoothed value at each eval)."""
    if OBJECTIVE[env] == "parking":
        v = evals["stoch_success_rate"].to_numpy(float) - 0.002 * np.minimum(
            evals["stoch_final_goal_dist_mean"].to_numpy(float), 30.0)
    else:
        v = evals[OBJECTIVE[env]].to_numpy(float)
    return v


def final_score(values, n_expected):
    """Mean over the last 25% of the expected evals (a run that stopped early is padded with its last value)."""
    if len(values) == 0:
        return float("nan")
    v = np.concatenate([values, np.full(max(0, n_expected - len(values)), values[-1])])
    k = max(1, int(round(0.25 * len(v))))
    return float(np.mean(v[-k:]))


def objective_factory(a: TuneArgs, env: str):
    import optuna
    import pandas as pd

    budget = a.total_timesteps or ENV_CONFIGS[env]["total_timesteps"]
    n_evals = budget // 25000 + 2  # step 0, every 25k, final

    def objective(trial):
        params = sample(trial, env)
        hp = to_args(env, params)
        trial_dir = os.path.abspath(os.path.join(a.study_dir, env, f"trial{trial.number:04d}"))
        runs = [(m, s) for m in MODES for s in a.seeds]
        procs = {}
        for mode, seed in runs:
            cmd = [sys.executable, "-u", "ppo.py", "--env-id", env, "--seed", str(seed), "--std-mode", mode,
                   "--runs-root", trial_dir, "--num-envs", str(ENV_CONFIGS[env]["num_envs"]),
                   "--total-timesteps", str(budget), "--eval-episodes", str(a.eval_episodes),
                   "--eval-seed-base", str(TUNE_EVAL_SEED_BASE), "--no-save-eval-dumps", "--no-grad-diagnostics"]
            for k, v in hp.items():
                cmd += [f"--{k.replace('_', '-')}={v!r}" if isinstance(v, float) else f"--{k.replace('_', '-')}={v}"]
            run_dir = os.path.join(trial_dir, env, mode, f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            log = open(os.path.join(run_dir, "stdout.log"), "w")
            procs[(mode, seed)] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log, run_dir)
        trial.set_user_attr("ppo_args", hp)
        t0, reported = time.time(), -1
        try:
            while True:
                alive = [p.poll() is None for p, _, _ in procs.values()]
                curves = []
                for p, _, run_dir in procs.values():
                    f = os.path.join(run_dir, "evals.csv")
                    try:
                        curves.append(run_score(pd.read_csv(f), env).copy() if os.path.exists(f)
                                      and os.path.getsize(f) > 0 else np.array([]))
                    except Exception:  # file being written right now: try again next poll
                        curves.append(np.array([]))
                    if len(curves[-1]) and not np.isfinite(curves[-1][-1]):
                        curves[-1] = curves[-1][:-1]  # partially written last row
                common = min(len(c) for c in curves)
                for k in range(reported + 1, common):  # report every eval all runs have reached
                    smoothed = float(np.mean([np.mean(c[max(0, k - 3):k + 1]) for c in curves]))
                    trial.report(smoothed, k)
                    reported = k
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"pruned at eval {k}")
                if not any(alive):
                    break
                time.sleep(a.poll_seconds)
        finally:
            for p, log, _ in procs.values():
                if p.poll() is None:
                    p.kill()
                    p.wait()
                log.close()
        scores = {}
        for (mode, seed), (p, _, run_dir) in procs.items():
            f = os.path.join(run_dir, "evals.csv")
            st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(
                os.path.join(run_dir, "status.json")) else {"status": "missing"}
            if st["status"] not in ("done", "nan"):
                raise RuntimeError(f"run {run_dir} ended with status {st['status']} (see its stdout.log)")
            scores[(mode, seed)] = final_score(run_score(pd.read_csv(f), env), n_evals)
        per_mode = {m: float(np.mean([scores[(m, s)] for s in a.seeds])) for m in MODES}
        for m in MODES:
            trial.set_user_attr(f"score_{m}", per_mode[m])
            trial.set_user_attr(f"seed_scores_{m}", [scores[(m, s)] for s in a.seeds])
        trial.set_user_attr("wall_clock_min", (time.time() - t0) / 60)
        return float(np.mean(list(per_mode.values())))

    return objective


# ---------------------------------------------------------------------------------------
def cmd_init(a: TuneArgs):
    study = load_study(a, a.env)
    if len(study.trials) == 0:
        for p in known_good(a.env):
            study.enqueue_trial(p, skip_if_exists=True)
        print(f"created study {a.env} with {len(known_good(a.env))} enqueued known-good configs")
    else:
        print(f"study {a.env} exists with {len(study.trials)} trials")


def cmd_worker(a: TuneArgs):
    import optuna

    seed = a.sampler_seed if a.sampler_seed >= 0 else int(os.environ.get("SLURM_ARRAY_TASK_ID", os.getpid()))
    budget = a.total_timesteps or ENV_CONFIGS[a.env]["total_timesteps"]
    warmup = int(0.3 * (budget // 25000 + 1))
    sampler = optuna.samplers.TPESampler(n_startup_trials=a.startup_trials, multivariate=True, constant_liar=True,
                                         seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=warmup, interval_steps=1)
    study = load_study(a, a.env, sampler, pruner)
    study.optimize(objective_factory(a, a.env), n_trials=a.n_trials, catch=(RuntimeError,), gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"{a.env}: {len(done)} complete trials; best so far {study.best_value if done else float('nan'):.4g}")


def completed(study):
    import optuna

    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and all(f"score_{m}" in t.user_attrs for m in MODES)]


def studies(a: TuneArgs):
    out = {}
    for f in sorted(os.listdir(a.study_dir)) if os.path.isdir(a.study_dir) else []:
        if f.endswith(".journal"):
            env = f[: -len(".journal")]
            out[env] = load_study(a, env)
    return out


def cmd_export(a: TuneArgs):
    conf = {}
    for env, study in studies(a).items():
        done = completed(study)
        if not done:
            print(f"{env}: no completed trials, skipped")
            continue
        best = max(done, key=lambda t: t.value)
        per = {m: max(done, key=lambda t: t.user_attrs[f"score_{m}"]) for m in MODES}
        conf[env] = {
            "shared": best.user_attrs["ppo_args"],
            "per_mode": {m: per[m].user_attrs["ppo_args"] for m in MODES},
            "info": {"n_complete_trials": len(done), "n_trials": len(study.trials), "shared_trial": best.number,
                     "shared_objective": best.value,
                     "shared_scores": {m: best.user_attrs[f"score_{m}"] for m in MODES},
                     "per_mode_trial": {m: per[m].number for m in MODES},
                     "per_mode_score": {m: per[m].user_attrs[f"score_{m}"] for m in MODES},
                     "note": "tuning scores use 2 seeds each and are optimistically biased (winner's curse); "
                             "the final runs re-estimate them with 10 fresh seeds"},
        }
        print(f"{env}: shared best = trial {best.number} (objective {best.value:.4g}); per-arm best = "
              + ", ".join(f"{m}: trial {per[m].number} ({per[m].user_attrs[f'score_{m}']:.4g})" for m in MODES))
    missing = [e for e in a.require if e not in conf]
    if missing or not conf:
        print(f"ERROR: no completed tuning trials for {missing or 'any env'}; {a.out} NOT written. "
              f"Wait for the tuning workers (bash cluster/submit.sh status) or run more of them.")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(conf, f, indent=2)
    print("wrote", a.out)


def cmd_report(a: TuneArgs):
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna
    import scipy.stats
    from common import MODE_COLORS, MODE_LABELS, MUTED, savefig, set_style

    set_style()
    md = ["# Hyper-parameter tuning (paired: both σ arms per trial)", "",
          "Scores: mean over the last 25% of evaluations, 2 tuning seeds per arm, held-out eval seeds. "
          "Difference = score(state) − score(global) at the same hyper-parameters. 'random configs' = the TPE "
          "start-up trials (uniform samples of the search space); 'all' also includes TPE's model-guided "
          "trials, which concentrate where the mean of the two arms is high.", ""]
    for env, study in studies(a).items():
        done = completed(study)
        states = [t.state.name for t in study.trials]
        md += [f"## {env}", "", f"trials: {len(study.trials)} ({states.count('COMPLETE')} complete, "
               f"{states.count('PRUNED')} pruned, {states.count('FAIL')} failed, {states.count('RUNNING')} running)", ""]
        if not done:
            continue
        g = np.array([t.user_attrs["score_global"] for t in done])
        s = np.array([t.user_attrs["score_state"] for t in done])
        rnd = np.array([t.number < a.startup_trials and "fixed_params" not in t.system_attrs for t in done])
        md += ["| configs | n | P(state > global) | mean diff | median diff | Wilcoxon p |", "|---|---|---|---|---|---|"]
        for lab, m in (("all complete", np.ones(len(done), bool)), ("random configs", rnd),
                       ("top quartile by shared objective", np.array([t.value for t in done])
                        >= np.percentile([t.value for t in done], 75))):
            d = (s - g)[m]
            if len(d) == 0:
                continue
            p = scipy.stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
            md.append(f"| {lab} | {len(d)} | {np.mean(d > 0):.2f} | {np.mean(d):+.4g} | {np.median(d):+.4g} | {p:.3g} |")
        best = max(done, key=lambda t: t.value)
        md += ["", "| config | trial | objective | score global | score state | " +
               " | ".join(best.user_attrs["ppo_args"]) + " |", "|---|---|---|---|---|" + "---|" * len(best.user_attrs["ppo_args"])]
        rows = [("shared best", best)] + [(f"best for {m}", max(done, key=lambda t, m=m: t.user_attrs[f"score_{m}"]))
                                          for m in MODES]
        for lab, t in rows:
            md.append(f"| {lab} | {t.number} | {t.value:.4g} | {t.user_attrs['score_global']:.4g} | "
                      f"{t.user_attrs['score_state']:.4g} | "
                      + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in t.user_attrs["ppo_args"].values())
                      + " |")
        try:
            imp = {m: optuna.importance.get_param_importances(study, target=lambda t, m=m: t.user_attrs.get(f"score_{m}", np.nan))
                   for m in MODES}
            md += ["", "Hyper-parameter importance (fANOVA) for each arm's score:", "",
                   "| param | " + " | ".join(MODES) + " |", "|---|" + "---|" * len(MODES)]
            for k in imp["global"]:
                md.append(f"| {k} | " + " | ".join(f"{imp[m].get(k, float('nan')):.2f}" for m in MODES) + " |")
        except Exception as e:  # too few trials, etc.
            md += ["", f"(parameter importance not computed: {type(e).__name__})"]
        md.append("")

        # figures: paired scatter + per-hyper-parameter slices for both arms
        params = list(done[0].params)
        fig, axes = plt.subplots(2, (len(params) + 2) // 2, figsize=(3.3 * ((len(params) + 2) // 2), 6.4), squeeze=False)
        ax = axes.flat[0]
        lo, hi = min(g.min(), s.min()), max(g.max(), s.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":")
        ax.scatter(g, s, s=14, color="#52514e")
        ax.set_xlabel("score, global σ")
        ax.set_ylabel("score, σ(s)")
        ax.set_title(f"same hyper-params (n={len(done)})\nabove the line: σ(s) better", fontsize=8)
        for ax, p in zip(list(axes.flat)[1:], params):
            x = [t.params[p] for t in done]
            cat = not all(isinstance(v, (int, float)) for v in x) or len(set(x)) <= 6
            xs = np.array([sorted(set(x)).index(v) for v in x], float) if cat else np.array(x, float)
            for m, off in zip(MODES, (-0.12, 0.12)):
                y = [t.user_attrs[f"score_{m}"] for t in done]
                ax.scatter(xs + (off if cat else 0), y, s=10, color=MODE_COLORS[m], alpha=0.7, label=MODE_LABELS[m])
            if cat:
                ax.set_xticks(range(len(set(x))), [f"{v:g}" if isinstance(v, float) else str(v) for v in sorted(set(x))],
                              fontsize=7)
            elif p == "learning_rate":
                ax.set_xscale("log")
            ax.set_title(p, fontsize=9)
        for ax in list(axes.flat)[len(params) + 1:]:
            ax.axis("off")
        axes.flat[1].legend(loc="best", fontsize=7)
        fig.suptitle(f"{env}: tuning trials, each training both σ arms at the same hyper-parameters", y=1.0)
        fig.tight_layout()
        savefig(fig, os.path.join(a.fig_dir, f"{env}_tuning.png"))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "tuning_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    args = tyro.cli(TuneArgs)
    {"init": cmd_init, "worker": cmd_worker, "export": cmd_export, "report": cmd_report}[args.command](args)
