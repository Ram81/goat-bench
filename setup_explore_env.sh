#!/usr/bin/env bash
# Build the environment for frontier-exploration coverage tours.
#
# This is goat-bench's own environment -- habitat-sim 0.2.3, habitat-lab v0.2.3,
# goat_bench -- plus the handful of numerical packages the exploration stack
# needs (scikit-fmm for the planner, sophus for the pose algebra, imageio for
# writing frames). It is the same stack README's install section describes, so
# GOAT training and exploration share one environment.
#
# It has to be the 0.2.3 stack: exploration runs inside GOATSim-v0, whose
# navmesh recomputation calls `recompute_navmesh(..., include_static_objects=
# False)` -- a keyword habitat-sim removed in 0.2.4.
#
# Two deviations from the README recipe, both forced rather than chosen:
#
#   * python 3.8, not 3.7. Nothing in goat-bench needs 3.7; what needs 3.8 is
#     torch. Ada GPUs (RTX 40-series) want a CUDA 11.8 build, torch first
#     shipped one in 2.0, and torch 2.x dropped python 3.7. habitat-sim 0.2.3
#     publishes py3.8 builds, so this costs nothing. On pre-Ada hardware
#     python=3.7 with the README's cudatoolkit=11.3 works as written.
#
#   * a few version pins the 2024-era requirements no longer resolve to on
#     their own. Each is commented where it appears; without them the install
#     "succeeds" and then fails at import.
#
# The full GOAT model stack (CLIP, LAVIS, VC-1) is NOT installed here -- it is
# several GB and exploration touches none of it. goat_bench/__init__.py guards
# those imports, so exploration works without them. See the end of this script.
set -euo pipefail
unset PYTHONPATH                       # ROS on PYTHONPATH shadows the env
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$PWD"

CONDA="${CONDA:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-goat}"
# Headless build: EGL rendering, no DISPLAY required. Bullet is included
# because habitat-sim's conda builds pair the two.
HSIM_BUILD=py3.8_headless_bullet_linux_fffa54376766602c4f12e30d0ee6100d56dd7a96

# habitat-lab is not pip-installable; it is used from a source checkout.
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-$HOME/habitat-lab-v0.2.3}"

# 1. habitat-sim. Conda supplies exactly this one package: it is not on PyPI.
if [ ! -x "$CONDA/envs/$ENV_NAME/bin/python" ]; then
  "$CONDA/bin/conda" create -n "$ENV_NAME" -y --solver=libmamba \
    -c aihabitat -c conda-forge python=3.8 "habitat-sim=0.2.3=$HSIM_BUILD"
fi
PY="$CONDA/envs/$ENV_NAME/bin/python"

# 2. habitat-lab v0.2.3 source, if not already present.
if [ ! -d "$HABITAT_LAB_DIR/habitat-lab" ]; then
  echo "Fetching habitat-lab v0.2.3 into $HABITAT_LAB_DIR"
  tmp="$(mktemp -d)"
  curl -sSL -o "$tmp/hl.tar.gz" \
    "https://codeload.github.com/facebookresearch/habitat-lab/tar.gz/refs/tags/v0.2.3"
  tar xzf "$tmp/hl.tar.gz" -C "$tmp"
  mv "$tmp/habitat-lab-0.2.3" "$HABITAT_LAB_DIR"
  rm -rf "$tmp"
fi

# 3. torch. cu118 is the oldest CUDA build with Ada support; 2.4.1 is the last
#    torch release for python 3.8. numpy is pinned below 2.0 because
#    habitat-sim 0.2.3 and gym are both built against the 1.x ABI.
"$PY" -m pip install "numpy<2" torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple

# 4. The exploration stack's own dependencies.
#
#    opencv is pinned: habitat-lab 0.2.3's visualizations/maps.py does
#    `cv2.applyColorMap(...).squeeze(1)`, and OpenCV >= 4.9 changed that call's
#    output shape, so `import habitat` itself raises ValueError.
#
#    sophuspy is pinned to 0.0.8 because that is the last release exposing the
#    module as `sophus`, which is how the vendored pose helpers import it.
"$PY" -m pip install \
  scikit-fmm scikit-image scikit-learn "opencv-python==4.8.1.78" \
  imageio imageio-ffmpeg matplotlib trimesh numpy-quaternion \
  "sophuspy==0.0.8"

# 5. habitat-baselines' pinned dependencies. goat_bench.config imports
#    habitat_baselines, so these are needed even though exploration runs no RL.
#
#    faster_fifo matters more than it looks: without it habitat-baselines falls
#    back to `class BatchedQueue(torch.multiprocessing.Queue)`, and that
#    fallback is broken (multiprocessing.Queue is a bound method, not a class),
#    so importing habitat_baselines raises TypeError.
#
#    lmdb is pinned because recent wheels are built against a newer CPython ABI
#    and fail on 3.8 with `undefined symbol: Py_SET_REFCNT`.
"$PY" -m pip install \
  "protobuf==3.20.1" "tensorboard==2.8.0" ifcfg moviepy threadpoolctl \
  "webdataset==0.1.40" "lmdb==1.4.1" faster_fifo

# 6. habitat-lab, habitat-baselines and goat_bench itself.
#    --no-deps on the latter two: their requirements pull the goal-conditioned
#    model stack, which step 4 deliberately leaves out.
"$PY" -m pip install -e "$HABITAT_LAB_DIR/habitat-lab"
"$PY" -m pip install --no-deps \
  -e "$HABITAT_LAB_DIR/habitat-baselines" \
  -e "$REPO_ROOT"

echo
echo "OK -- environment '$ENV_NAME'."
echo "Collect tours with:  ./run_explore_tour.sh"
echo
echo "For GOAT training as well, add the goal-conditioned stack to this same"
echo "environment:  $PY -m pip install -r requirements.txt \\"
echo "                  git+https://github.com/openai/CLIP.git"
