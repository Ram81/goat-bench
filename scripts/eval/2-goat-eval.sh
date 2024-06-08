#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/eval/goat-%j.out
#SBATCH --error=slurm_logs/eval/goat-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=short
#SBATCH --exclude=xaea-12
#SBATCH --signal=USR1@100

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

DATA_PATH="data/datasets/goat_bench/hm3d/v1/"
eval_ckpt_path_dir="data/goat-assets/checkpoints/sense_act_nn_monolithic/"
tensorboard_dir="tb/goat/sense_act_nn_monolithic/val_seen/"
split="val_seen"

srun python -um goat_bench.run \
  --run-type eval \
  --exp-config config/experiments/ver_goat_monolithic.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.trainer_name="goat_ppo" \
  habitat_baselines.video_dir="${tensorboard_dir}/videos" \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache=data/goat-assets/goal_cache/iin/${split}_embeddings/ \
  habitat.task.lab_sensors.goat_goal_sensor.language_cache=data/goat-assets/goal_cache/language_nav/${split}_instruction_clip_embeddings.pkl

touch $checkpoint_counter
