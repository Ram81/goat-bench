#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/goat-ver-%j.out
#SBATCH --error=slurm_logs/goat-ver-%j.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --constraint=a40
#SBATCH --exclude=gundam,xaea-12,baymax
#SBATCH --partition=short
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate goat

export PYTHONPATH=/srv/flash1/rramrakhya3/fall_2023/habitat-sim/src_python/

TENSORBOARD_DIR="tb/goat/release/ver/monlithic_clip/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/goat/release/ver/monolithic_clip/seed_1/"
DATA_PATH="data/datasets/goat_bench/v1/"

srun python -um goat_bench.run \
  --run-type train \
  --exp-config config/experiments/ver_goat_monolithic.yaml \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  habitat_baselines.rl.ddppo.distrib_backend="GLOO"
