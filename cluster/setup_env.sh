#!/bin/bash
# One-time setup of the Python environment on FASRC Cannon. Run it from the repo root,
# ideally inside a short interactive job (installing on a login node is slow and frowned upon):
#
#   salloc -p test -c 4 --mem=8G -t 0-01:00
#   bash cluster/setup_env.sh
#
# Creates a Python 3.11 environment at $ENV_PREFIX (default ~/envs/ppo_sd), installs the
# pinned requirements (CPU-only PyTorch), installs simglucose without its stale pins, and
# runs the unit tests.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

# `module` is a shell function; make sure it exists in this non-interactive shell (Lmod init).
# Lmod's scripts may read unset variables, so relax `set -u` around them.
set +u
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    [ -f "$f" ] && source "$f" && break
  done
fi
module load python   # FASRC Miniforge (mamba + conda-forge)
set -u
if [ ! -x "$PY" ]; then
  echo ">>> creating Python 3.11 environment at $ENV_PREFIX"
  if type mamba >/dev/null 2>&1; then CONDA=mamba; else CONDA=conda; fi
  "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip
fi
echo ">>> python: $("$PY" --version) at $PY"
# FASRC's 2026 certificate update needs a recent certifi in every environment
"$PY" -m pip install --upgrade pip "certifi>=2026.10.19" || "$PY" -m pip install --upgrade pip certifi
# CPU-only PyTorch: ~200 MB instead of ~3 GB of CUDA libraries we would never use
"$PY" -m pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu \
  || { echo ">>> CPU wheel index unavailable, installing torch from PyPI instead"; "$PY" -m pip install "torch==2.15.0"; }
"$PY" -m pip install -r requirements.txt
# simglucose pins gymnasium 0.29 / gym 0.9.4 for wrappers we do not use -> install without deps
"$PY" -m pip install --no-deps -r requirements-simglucose.txt
"$PY" -c "import gymnasium, torch, highway_env, envs.simglucose_env; print('gymnasium', gymnasium.__version__, '| torch', torch.__version__)"
echo ">>> running unit tests"
SDL_VIDEODRIVER=offscreen MPLBACKEND=Agg "$PY" -m pytest -q
echo ">>> environment ready: $ENV_PREFIX"
