#!/bin/bash
# Submit the study to Slurm (FASRC Cannon). Run from the repo root.
#
#   bash cluster/submit.sh smoke      # 6 short runs (1 seed, 50k steps, 3 envs x 2 modes) on the test partition
#   bash cluster/submit.sh all        # tune -> final comparison -> probes -> analysis, chained (the whole study)
#
# The stages of `all`, if you want to run them one by one:
#   bash cluster/submit.sh tune       # paired Optuna studies, one per env (scripts/tune.py)
#   bash cluster/submit.sh final      # export tuned configs; 10 seeds x 2 modes x 3 envs at the shared tuned
#                                     #   config ($RUNS_ROOT) and at each arm's own best config ($PERARM_ROOT)
#   bash cluster/submit.sh probe      # noise-tolerance probe of every finished run under $RUNS_ROOT
#   bash cluster/submit.sh analysis   # all figures and tables (+ demo videos)
# Optional:
#   bash cluster/submit.sh main       # the spec's untuned shared default config -> $DEFAULT_ROOT (robustness check)
#   bash cluster/submit.sh extra      # parking without walls, Simglucose log-action map, LunarLander (tuned config if available)
#   bash cluster/submit.sh ablations  # state_detach / state_separate modes, entropy 0.01, GAE lambda 0.9 (parking), clip 0.25
#   bash cluster/submit.sh status     # done / running / missing counts
#
# Re-running a stage only submits what is not finished yet (runs whose status.json says "done"
# or "nan" are skipped; tuning workers just add trials to the existing study), so after a
# time-out or failure run the same command again. Options via environment variables (see
# cluster/config.sh): PARTITION, ACCOUNT, RUNS_ROOT, SEEDS (default 10), MODES, AFTER (job ids
# to wait for, colon-separated).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
mkdir -p cluster/logs cluster/tasks
STAGE=${1:-help}
ACC=()
[ -n "$ACCOUNT" ] && ACC=(--account="$ACCOUNT")
CORE_ENVS=(parking-v0 Pendulum-v1 simglucose-adolescent1-v0)
EXTRA_ENVS=(parking-nowalls-v0 simglucose-adolescent1-logact-v0 LunarLanderContinuous-v3)

dep_args() { if [ -n "${1:-}" ]; then echo "--dependency=afterany:$1"; fi; }
join_ids() { local IFS=:; echo "$*"; }

submit_array() {  # name taskfile time mem partition [dependency] -> prints job id
  local name=$1 tasks=$2 time=$3 mem=$4 part=$5 dep=${6:-} n jid
  n=$(grep -c . "$tasks" || true)
  if [ "$n" -eq 0 ]; then echo "  $name: nothing to submit" >&2; return 0; fi
  jid=$(sbatch --parsable "${ACC[@]}" $(dep_args "$dep") -J "$name" -p "$part" -t "$time" --mem="$mem" \
        --array=1-"$n" cluster/array.sbatch "$tasks")
  echo "  $name: $n tasks -> job $jid  (partition $part, limit $time each${dep:+, after $dep})" >&2
  echo "$jid"
}

train_stage() {  # runs_root "ppo extra args" env...   (env vars: MODES, SEEDS, CONFIG, CONFIG_KEY, AFTER)
  local root=$1 extra=$2 ids=() env tf tag jid
  shift 2
  for env in "$@"; do
    tag=$(echo "$root" | tr '/.' '__')_$(echo "${MODES:-global state}" | tr ' ' '-')_${CONFIG_KEY:-shared}
    tf=cluster/tasks/train_${tag}_${env}.jsonl
    "$PY" scripts/sweep.py --envs "$env" --modes ${MODES:-global state} --seeds "${SEEDS:-10}" --runs-root "$root" \
        --extra="$extra" --config="${CONFIG:-}" --config-key "${CONFIG_KEY:-shared}" --emit-tasks "$tf" >&2
    jid=$(submit_array "ppo_${env%%-*}" "$tf" "${TIME_LIMIT[$env]}" "$MEM_TRAIN" "$PARTITION" "${AFTER:-}")
    if [ -n "$jid" ]; then ids+=("$jid"); fi
  done
  join_ids "${ids[@]}"
}

tune_stage() {  # -> prints job ids   (envs: $TUNE_ENVS, see cluster/config.sh)
  local ids=() env jid
  for env in $TUNE_ENVS; do
    "$PY" scripts/tune.py init --env "$env" --study-dir "$TUNE_DIR" >&2
    jid=$(sbatch --parsable "${ACC[@]}" -J "tune_${env%%-*}" -p "$PARTITION" -t "${TUNE_TIME[$env]}" --mem="$MEM_TUNE" \
          --array=1-"${TUNE_WORKERS[$env]}" cluster/tune.sbatch "$env" "${TUNE_TRIALS[$env]}")
    echo "  tune $env: ${TUNE_WORKERS[$env]} workers x ${TUNE_TRIALS[$env]} trials -> job $jid (limit ${TUNE_TIME[$env]})" >&2
    ids+=("$jid")
  done
  join_ids "${ids[@]}"
}

final_stage() {  # -> prints "shared_ids|perarm_ids"
  "$PY" scripts/tune.py export --study-dir "$TUNE_DIR" --out "$TUNED_CONFIG" --require "${CORE_ENVS[@]}" >&2 \
    || { echo "tuning is not finished for every env: final comparison NOT submitted" >&2; exit 1; }
  local shared perarm
  echo "final comparison at the shared tuned config -> $RUNS_ROOT" >&2
  shared=$(CONFIG="$TUNED_CONFIG" CONFIG_KEY=shared train_stage "$RUNS_ROOT" "" "${CORE_ENVS[@]}")
  echo "final comparison, each arm at its own best config -> $PERARM_ROOT" >&2
  perarm=$(CONFIG="$TUNED_CONFIG" CONFIG_KEY=per_mode train_stage "$PERARM_ROOT" "" "${CORE_ENVS[@]}")
  echo "$shared|$perarm"
}

probe_stage() {  # dependency ("" = probe whatever has finished) -> prints job id
  local dep=${1:-} only=() envs modes
  if [ -z "$dep" ]; then   # stand-alone: every finished run of any env / mode
    only=(--only-finished)
    envs=("${CORE_ENVS[@]}" parking-nowalls-v0 simglucose-adolescent1-logact-v0)
    modes=(global state state_detach state_separate)
  else                     # chained after the final comparison: exactly its runs
    envs=("${CORE_ENVS[@]}")
    modes=(global state)
  fi
  "$PY" analysis/probe.py --runs-root "$RUNS_ROOT" --envs "${envs[@]}" --modes "${modes[@]}" \
      --seeds $(seq 1 "${SEEDS:-10}") --emit-tasks cluster/tasks/probe.jsonl "${only[@]}" >&2
  submit_array probe cluster/tasks/probe.jsonl "$TIME_PROBE" "$MEM_PROBE" "$PARTITION" "$dep"
}

analysis_stage() {  # [dependency] -> prints job id
  local dep=${1:-} jid
  jid=$(sbatch --parsable "${ACC[@]}" $(dep_args "$dep") -J analysis -p "$PARTITION" -t "$TIME_ANALYSIS" \
        --mem="$MEM_ANALYSIS" cluster/analysis.sbatch)
  echo "  analysis -> job $jid${dep:+ (after $dep)}" >&2
  echo "$jid"
}

tuned_config_or_default() {
  if [ -f "$TUNED_CONFIG" ]; then echo "$TUNED_CONFIG"; else
    echo "  note: $TUNED_CONFIG not found (tuning not exported yet) -> the spec's default config is used" >&2; echo ""; fi
}

case "$STAGE" in
  smoke)
    "$PY" scripts/sweep.py --smoke --envs "${CORE_ENVS[@]}" --emit-tasks cluster/tasks/smoke.jsonl
    submit_array smoke cluster/tasks/smoke.jsonl 0-01:00 "$MEM_TRAIN" "$SMOKE_PARTITION" >/dev/null
    echo "check with: squeue --me ; then: $PY scripts/sweep.py --smoke --dry-run (shows what is left)" ;;
  tune)
    tune_stage >/dev/null ;;
  final)
    final_stage >/dev/null ;;
  after-tuning)   # used by cluster/chain.sbatch
    ids=$(final_stage)
    shared=${ids%%|*}; perarm=${ids##*|}
    probe_id=$(probe_stage "$shared")
    analysis_stage "$(join_ids $(echo "${probe_id:-$shared}:$perarm" | tr ':' ' '))" >/dev/null ;;
  all)
    echo "1) tuning (paired Optuna study per env)"
    tune_ids=$(tune_stage)
    echo "2) when tuning is done: export -> final comparison -> probes -> analysis (chained job)"
    jid=$(sbatch --parsable "${ACC[@]}" --dependency=afterany:"$tune_ids" -J chain -p "$PARTITION" cluster/chain.sbatch)
    echo "  chain -> job $jid (after $tune_ids)" ;;
  main)
    echo "spec default config (untuned) -> $DEFAULT_ROOT"
    train_stage "$DEFAULT_ROOT" "" "${CORE_ENVS[@]}" >/dev/null ;;
  extra)
    cfg=$(tuned_config_or_default)
    echo "extra envs -> $RUNS_ROOT"
    CONFIG="$cfg" train_stage "$RUNS_ROOT" "" "${EXTRA_ENVS[@]}" >/dev/null ;;
  ablations)
    cfg=$(tuned_config_or_default)
    echo "ablation modes (state_detach, state_separate) -> $RUNS_ROOT"
    CONFIG="$cfg" MODES="state_detach state_separate" train_stage "$RUNS_ROOT" "" "${CORE_ENVS[@]}" >/dev/null
    echo "entropy bonus 0.01 (both modes) -> $ABL_ROOT/ent0.01"
    CONFIG="$cfg" train_stage "$ABL_ROOT/ent0.01" "--ent-coef 0.01" "${CORE_ENVS[@]}" >/dev/null
    echo "GAE lambda 0.9 on parking (both modes) -> $ABL_ROOT/lam0.9"
    CONFIG="$cfg" train_stage "$ABL_ROOT/lam0.9" "--gae-lambda 0.9" parking-v0 >/dev/null
    echo "PPO clip 0.25 (each mean step is bounded by ~clip x sigma) -> $ABL_ROOT/clip0.25"
    CONFIG="$cfg" train_stage "$ABL_ROOT/clip0.25" "--clip-coef 0.25" "${CORE_ENVS[@]}" >/dev/null ;;
  probe)
    probe_stage "${AFTER:-}" >/dev/null ;;
  analysis)
    analysis_stage "${AFTER:-}" >/dev/null ;;
  status)
    "$PY" - "$RUNS_ROOT" "$PERARM_ROOT" "$DEFAULT_ROOT" "$ABL_ROOT" <<'PYEOF'
import glob, json, os, sys
from collections import Counter
for root in sys.argv[1:]:
    rows = Counter()
    for st_path in glob.glob(os.path.join(root, "**", "seed*", "status.json"), recursive=True):
        group = os.path.relpath(os.path.dirname(os.path.dirname(st_path)), root)  # [abl/]env/mode
        try:
            st = json.load(open(st_path))["status"]
        except Exception:
            st = "unreadable"
        rows[(group, st)] += 1
    if rows:
        print(f"\n{root}:  (runs per status)")
        for (group, st), n in sorted(rows.items()):
            print(f"  {group:60s} {st:10s} {n}")
PYEOF
    for j in "$TUNE_DIR"/*.journal; do
      [ -f "$j" ] || continue
      "$PY" -c "
import optuna, sys
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
optuna.logging.set_verbosity(optuna.logging.WARNING)
p = sys.argv[1]; st = optuna.storages.JournalStorage(JournalFileBackend(p, lock_obj=JournalFileOpenLock(p)))
for s in optuna.get_all_study_summaries(st):
    tr = optuna.load_study(study_name=s.study_name, storage=st).trials
    c = lambda k: sum(t.state.name == k for t in tr)
    print(f'tuning {s.study_name:30s} complete {c(\"COMPLETE\")}, pruned {c(\"PRUNED\")}, running {c(\"RUNNING\")}, failed {c(\"FAIL\")}')
" "$j"
    done
    echo; squeue --me 2>/dev/null | head -25 || true ;;
  *)
    sed -n '2,24p' "$0" ;;
esac
