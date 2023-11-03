#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/eval/goat-%j.out
#SBATCH --error=slurm_logs/eval/goat-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --constraint="a40"
#SBATCH --partition=short
#SBATCH --exclude=xaea-12
#SBATCH --signal=USR1@100

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate goat

export PYTHONPATH=/srv/flash1/rramrakhya3/fall_2023/habitat-sim/src_python/

DATA_PATH="data/datasets/goat/v0.1.3"
# eval_ckpt_path_dir="data/new_checkpoints/goat/ver/resnetclip_rgb_multimodal/seed_1/"
# tensorboard_dir="tb/goat/ver/resnetclip_rgb_multimodal/seed_1/val_seen/"
# split="val_seen"

srun python -um goat.run \
  --run-type eval \
  --exp-config config/experiments/ver_goat.yaml \
  habitat_baselines.num_environments=20 \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.goat_goal_sensor=goat_goal_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.lab_sensors.goat_goal_sensor.object_cache=data/clip_embeddings/ovon_stretch_final_cache.pkl \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache=data/datasets/iin/hm3d/v2/${split}_embeddings/ \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache_encoder="CLIP_goat" \
  habitat.task.lab_sensors.goat_goal_sensor.language_cache="data/datasets/languagenav/hm3d/v5_final/embeddings/${split}_clip_embedding.pkl" \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="Goat-v1" \
  habitat.task.measurements.distance_to_goal.type=GoatDistanceToGoal \
  habitat.task.measurements.success.type=GoatSuccess \
  habitat.task.measurements.spl.type=GoatSPL \
  habitat.task.measurements.soft_spl.type=GoatSoftSPL \
  +habitat/task/measurements@habitat.task.measurements.goat_distance_to_goal_reward=goat_distance_to_goal_reward \
  ~habitat.task.measurements.distance_to_goal_reward \
  habitat.simulator.type="OVONSim-v0" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split \
  habitat_baselines.should_load_agent_state=False

touch $checkpoint_counter
