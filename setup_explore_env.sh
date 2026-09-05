#!/usr/bin/env bash
# Build the environment needed to collect frontier-exploration coverage tours.
#
# This is deliberately NOT goat-bench's setup.sh. That script targets python 3.7
# + habitat-sim 0.2.3 + habitat-lab v0.2.3 and installs the full goal-conditioned
# stack (CLIP, salesforce-lavis, open3d). Exploration needs none of that, and
# python 3.7 is long past end of life. What it needs is habitat-sim, habitat-lab,
# torch, and a handful of numerical libraries.
#
# The result is a python 3.9 / habitat-sim 0.2.5 environment, matching the one
# home-robot's exploration stack was validated in.
#
# Caveat worth knowing: goat-bench's *RL* code does not run here.
# `goat_bench.task.actions` and `goat_bench.task.environments` use habitat-lab
# 0.2.3 APIs removed in 0.2.5, and `goat_bench.task.sensors` needs lavis. The
# package guards those imports (see goat_bench/__init__.py), so exploration works
# and GOAT training does not. For GOAT training, use the upstream setup.sh.
set -euo pipefail
unset PYTHONPATH                       # ROS on PYTHONPATH shadows the venv
cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA="${CONDA:-$HOME/miniforge3}"
# Same headless build home-robot uses: EGL rendering, no DISPLAY required.
HSIM_BUILD=py3.9_headless_bullet_linux_c8887c8cf421b6df2d77489d42f4e20488e184eb

# habitat-lab is not reliably pip-installable, so it is used from a checkout.
# Point this at your own clone (v0.2.5) if you have one; the default is the
# copy home-robot already vendors as a submodule.
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-$HOME/home-robot/src/third_party/habitat-lab}"

# 1. habitat-sim only. It is not pip-installable (PyPI carries a single
#    0.2.4.dev macOS arm64 wheel), so conda supplies exactly this one package.
if [ ! -x "$CONDA/envs/hsim/bin/python" ]; then
  "$CONDA/bin/conda" create -n hsim -y --solver=libmamba \
    -c aihabitat -c conda-forge python=3.9 "habitat-sim=0.2.5=$HSIM_BUILD"
fi

# 2. A uv venv on that interpreter, so habitat_sim's compiled extensions stay
#    ABI-compatible with the python running them.
if [ ! -d .venv ]; then
  uv venv --python "$CONDA/envs/hsim/bin/python" --system-site-packages
fi
export VIRTUAL_ENV="$PWD/.venv"

# 3. Dependencies of the exploration stack. scikit-fmm is the fast-marching
#    solver behind the planner; numpy is pinned below 2.0 because habitat-sim
#    0.2.5 and gym are both built against the 1.x ABI.
uv pip install \
  "numpy<2" torch torchvision \
  scikit-fmm scikit-image scikit-learn opencv-python \
  imageio "imageio-ffmpeg" matplotlib trimesh numpy-quaternion

# 4. habitat-lab and habitat-baselines from the checkout, then goat_bench itself.
#    --no-deps throughout: goat-bench's setup.py and requirements.txt pull the
#    goal-conditioned model stack, which step 3 deliberately leaves out.
if [ ! -d "$HABITAT_LAB_DIR/habitat-lab" ]; then
  echo "habitat-lab not found at $HABITAT_LAB_DIR" >&2
  echo "Set HABITAT_LAB_DIR to a habitat-lab v0.2.5 checkout and re-run." >&2
  exit 1
fi
uv pip install --no-deps \
  -e "$HABITAT_LAB_DIR/habitat-lab" \
  -e "$HABITAT_LAB_DIR/habitat-baselines" \
  -e .

echo
echo "OK. Collect tours with:  ./run_explore_tour.sh"
echo "Visualise them with:     .venv/bin/python scripts/exploration/view_tour.py data/tours"
