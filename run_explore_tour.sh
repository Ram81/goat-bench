#!/usr/bin/env bash
# Collect frontier-exploration coverage tours in GOAT-Bench scenes, streaming
# rgb/depth/pose to disk.
#
#   ./run_explore_tour.sh                          # defaults below
#   EPISODES=20 STEPS=750 ./run_explore_tour.sh    # override via environment
#   ./run_explore_tour.sh habitat.dataset.split=train   # habitat overrides pass through
#
# Note on quoting: habitat data_path templates contain braces, which hydra's
# override grammar rejects unless the value is quoted *inside* the argument:
#   ./run_explore_tour.sh "habitat.dataset.data_path='data/foo/{split}/{split}.json.gz'"
set -euo pipefail

# ROS distributions put their own python packages on PYTHONPATH, which shadow
# the venv's numpy/opencv and produce confusing import errors.
unset PYTHONPATH

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# The per-step map update is small; BLAS thread pools cost more than they save.
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

STEPS="${STEPS:-500}"
EPISODES="${EPISODES:-5}"
OUT="${OUT:-$REPO_ROOT/data/tours}"
DEVICE="${DEVICE:-0}"

# Use the goat conda env that setup_explore_env.sh builds, unless PYTHON says
# otherwise or that env does not exist -- in which case fall back to whatever
# python is active, which is what you want inside an already-activated env.
CONDA="${CONDA:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-goat}"
if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "$CONDA/envs/$ENV_NAME/bin/python" ]; then
  PYTHON="$CONDA/envs/$ENV_NAME/bin/python"
else
  PYTHON="python"
fi

exec "$PYTHON" scripts/exploration/collect_tour.py \
  --out "$OUT" \
  --num_episodes "$EPISODES" \
  --max_steps "$STEPS" \
  --device_id "$DEVICE" \
  "$@"
