#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/eval/goat-%j.out
#SBATCH --error=slurm_logs/eval/goat-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=xaea-12
#SBATCH --signal=USR1@100

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

DATA_PATH="data/datasets/goat_bench/hm3d/v1/"
eval_ckpt_path_dir="data/new_checkpoints/goat_bench/ver/skill_chain_croco/"
tensorboard_dir="tb/goat_bench/ver/skill_chain_croco/val_seen/"
split="val_seen"

srun python -um goat.run \
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
  +habitat/task/lab_sensors@habitat.task.lab_sensors.ginstance_imagegoal_sensor=instance_imagegoal_sensor \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.current_subtask_sensor=current_subtask_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache=data/clip_embeddings/ovon_stretch_final_cache.pkl \
  habitat.task.lab_sensors.language_goal_sensor.cache=data/datasets/languagenav/hm3d/v5_final/embeddings/${split}_bert_embedding.pkl \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="Goat-v1" \
  habitat.task.measurements.distance_to_goal.type=GoatDistanceToGoal \
  habitat.task.measurements.success.type=GoatSuccess \
  habitat.task.measurements.spl.type=GoatSPL \
  habitat.task.measurements.soft_spl.type=GoatSoftSPL \
  ~habitat.task.measurements.distance_to_goal_reward \
  +habitat/task/measurements@habitat.task.measurements.goat_distance_to_goal_reward=goat_distance_to_goal_reward \
  habitat.simulator.type="OVONSim-v0" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split \
  habitat_baselines.eval.should_load_ckpt=False \
  habitat_baselines.should_load_agent_state=False \
  habitat_baselines.rl.policy.add_instance_linear_projection=True \
  habitat_baselines.rl.policy.croco_adapter=True \
  habitat_baselines.rl.policy.use_croco=True

touch $checkpoint_counter
