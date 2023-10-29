#!/bin/bash
#SBATCH --job-name=goat-croco-4-gpus
#SBATCH --output=slurm_logs/goat-croco-ver-%j.out
#SBATCH --error=slurm_logs/goat-croco-ver-%j.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --constraint=a40
#SBATCH --exclude=xaea-12,cheetah,omgwth,conroy,ig-88,sonny,deebot,chappie,baymax,heistotron,uniblab,chitti,gundam,megabot,optimistprime
#SBATCH --partition=short
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/gchhablani3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate goat

TENSORBOARD_DIR="tb/iin/ver/resnetclip_rgb_croco_image/4_gpus/"
CHECKPOINT_DIR="data/new_checkpoints/iin/ver/resnetclip_rgb_croco_image/4_gpus/"
DATA_PATH="data/datasets/iin/hm3d/v2"

srun python -um goat.run \
  --run-type train \
  --exp-config config/experiments/ver_instance_imagenav.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.num_environments=16 \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.simulator.type="OVONSim-v0" \
  habitat_baselines.rl.policy.add_instance_linear_projection=True \
  habitat_baselines.rl.policy.croco_adapter=True \
  habitat_baselines.rl.policy.use_croco=True 
