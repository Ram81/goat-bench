#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/goat-ver-%j.out
#SBATCH --error=slurm_logs/goat-ver-%j.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --constraint=a40
#SBATCH --exclude=megabot,gundam,kitt,cheetah
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
export HOME=/srv/flash1/rramrakhya3/summer_2023

TENSORBOARD_DIR="tb/iin/ver/resnetclip_rgb_clip_image/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/iin/ver/resnetclip_rgb_clip_image/seed_1/"
DATA_PATH="data/datasets/iin/hm3d/v2/"

srun python -um goat.run \
  --run-type train \
  --exp-config config/experiments/ver_instance_imagenav.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.num_environments=32 \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.cache_instance_imagegoal_sensor=cache_instance_imagegoal_sensor \
  ~habitat.task.lab_sensors.instance_imagegoal_sensor \
  habitat.task.lab_sensors.cache_instance_imagegoal_sensor.cache=data/datasets/iin/hm3d/v2/train_goal_embeddings/ \
  habitat.task.lab_sensors.cache_instance_imagegoal_sensor.image_cache_encoder="CLIP" \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.simulator.type="GOATSim-v0" \
  habitat_baselines.rl.ddppo.distrib_backend="GLOO"
