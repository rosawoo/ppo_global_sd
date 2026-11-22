# Settings shared by the cluster scripts. Edit here, or override any of them per command:
#   PARTITION=serial_requeue bash cluster/submit.sh main
# Find the partitions your account can use with:  spart

# Python environment created by cluster/setup_env.sh
ENV_PREFIX=${ENV_PREFIX:-$HOME/envs/ppo_sd}
PY=${PY:-$ENV_PREFIX/bin/python}

# CPU partitions (comma-separated: Slurm starts each task on whichever frees up first).
# shared = 48-core Cascade Lake nodes, sapphire = 112-core Sapphire Rapids nodes; both 3-day limit.
# Adding serial_requeue gives much more throughput, but a task can be pre-empted; it is then
# requeued and restarts from step 0 (safe: see scripts/run_task.py), so it only costs time.
PARTITION=${PARTITION:-shared,sapphire}
# Partition for the quick smoke test (test: 12 h limit, at most 5 jobs per user, no fairshare cost)
SMOKE_PARTITION=${SMOKE_PARTITION:-test}

# Optional: a Slurm account (e.g. your lab's) for fairshare; empty = your default account
ACCOUNT=${ACCOUNT:-}

# Where runs are written (relative to the repo or absolute, e.g. /n/netscratch/<lab>/Lab/$USER/ppo_sd/...)
RUNS_ROOT=${RUNS_ROOT:-runs_final}            # final comparison at the tuned SHARED config (primary result)
PERARM_ROOT=${PERARM_ROOT:-runs_final_perarm} # final comparison, each arm at its OWN best tuned config
DEFAULT_ROOT=${DEFAULT_ROOT:-runs}            # the spec's untuned shared default config (robustness check)
ABL_ROOT=${ABL_ROOT:-runs_abl}                # hyper-parameter ablations (on top of the tuned config)
TUNE_DIR=${TUNE_DIR:-tuning}                  # Optuna journals + tuning runs
TUNED_CONFIG=${TUNED_CONFIG:-configs/tuned.json}

# Tuning budget per env: workers (parallel array tasks, 4 cores each) x trials per worker.
# Each trial = 2 arms x 2 seeds at the full training length; median pruning stops weak trials early.
# Which envs get their own study: the three core envs by default. Add parking-nowalls-v0 if you
# also run that variant (otherwise it inherits parking-v0's tuned config), e.g.
#   TUNE_ENVS="parking-v0 Pendulum-v1 simglucose-adolescent1-v0 parking-nowalls-v0" bash cluster/submit.sh all
TUNE_ENVS=${TUNE_ENVS:-"parking-v0 Pendulum-v1 simglucose-adolescent1-v0"}
declare -A TUNE_WORKERS=( [parking-v0]=8 [Pendulum-v1]=8 [simglucose-adolescent1-v0]=8 [parking-nowalls-v0]=8
                          [simglucose-adolescent1-logact-v0]=8 [LunarLanderContinuous-v3]=8 )
declare -A TUNE_TRIALS=( [parking-v0]=4 [Pendulum-v1]=8 [simglucose-adolescent1-v0]=5 [parking-nowalls-v0]=4
                         [simglucose-adolescent1-logact-v0]=5 [LunarLanderContinuous-v3]=5 )
declare -A TUNE_TIME=( [parking-v0]=1-12:00 [Pendulum-v1]=0-06:00 [simglucose-adolescent1-v0]=0-16:00
                       [parking-nowalls-v0]=1-12:00 [simglucose-adolescent1-logact-v0]=0-16:00
                       [LunarLanderContinuous-v3]=0-12:00 )
MEM_TUNE=${MEM_TUNE:-12G}

# Per-task resources. Each task is ONE single-threaded PPO run (the networks are tiny and the
# simulators are CPU-bound, so GPUs would sit idle; see cluster/README.md).
MEM_TRAIN=${MEM_TRAIN:-3G}
MEM_PROBE=${MEM_PROBE:-4G}
MEM_ANALYSIS=${MEM_ANALYSIS:-16G}

# Wall-clock limits per training run (generous upper bounds; a run ends as soon as it is done)
declare -A TIME_LIMIT=(
  [parking-v0]=0-08:00
  [parking-nowalls-v0]=0-10:00
  [Pendulum-v1]=0-01:00
  [simglucose-adolescent1-v0]=0-04:00
  [simglucose-adolescent1-logact-v0]=0-04:00
  [LunarLanderContinuous-v3]=0-02:00
)
TIME_PROBE=${TIME_PROBE:-0-03:00}
TIME_ANALYSIS=${TIME_ANALYSIS:-0-06:00}
