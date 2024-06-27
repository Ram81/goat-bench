#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/goat-croco-ver-%j.out
#SBATCH --error=slurm_logs/goat-croco-ver-%j.err
#SBATCH --gpus a40:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --exclude=xaea-12,nestor,shakey,dave,voltron,deebot
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=short

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

TENSORBOARD_DIR="tb/iin/ver/resnetclip_rgb_croco_image/seed_1"
CHECKPOINT_DIR="data/new_checkpoints/iin/ver/resnetclip_rgb_croco_image/seed_1"
DATA_PATH="data/datasets/iin/hm3d/v2"

srun python -um goat_bench.run \
  --run-type train \
  --exp-config config/experiments/ver_instance_imagenav.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.num_environments=16 \
  habitat_baselines.rl.policy.name=GOATPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.simulator.type="GOATSim-v0" \
  habitat_baselines.rl.policy.add_instance_linear_projection=True \
  habitat_baselines.rl.policy.croco_adapter=True \
  habitat_baselines.rl.policy.use_croco=True 
