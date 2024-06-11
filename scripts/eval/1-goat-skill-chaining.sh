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

DATA_PATH="data/datasets/goat_bench/hm3d/v1/"
eval_ckpt_path_dir="data/new_checkpoints/goat_bench/ver/skill_chain/"
tensorboard_dir="tb/goat_bench/ver/skill_chain/val_seen/"
split="val_seen"

srun python -um goat_bench.run \
  --run-type eval \
  --exp-config config/experiments/ver_goat_skill_chain.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.trainer_name="goat_ppo" \
  habitat_baselines.rl.policy.name=GoatHighLevelPolicy \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat_baselines.checkpoint_folder=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.clip_objectgoal_sensor=clip_objectgoal_sensor \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.language_goal_sensor=language_goal_sensor \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.cache_instance_imagegoal_sensor=cache_instance_imagegoal_sensor \
  ~habitat.task.lab_sensors.goat_goal_sensor \
  habitat.task.lab_sensors.cache_instance_imagegoal_sensor.cache=data/goat-assets/goal_cache/iin/${split}_embeddings/ \
  habitat.task.lab_sensors.language_goal_sensor.cache=data/goat-assets/goal_cache/language_nav/${split}_bert_embedding.pkl \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split \
  habitat_baselines.eval.should_load_ckpt=False \
  habitat_baselines.should_load_agent_state=False

touch $checkpoint_counter
