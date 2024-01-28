#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/eval/goat-%j.out
#SBATCH --error=slurm_logs/eval/goat-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --constraint="a40|rtx_6000|2080_ti"
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

DATA_PATH="data/datasets/iin/hm3d/v2/"
# eval_ckpt_path_dir="data/new_checkpoints/iin/ver/resnetclip_rgb_vc1_image/seed_1/ckpt.70.pth"
# tensorboard_dir="tb/iin/ver/resnetclip_rgb_vc1_image/seed_1/val_seen_ckpt_70/"
# split="val_seen"

echo "Evaluating ckpt: ${eval_ckpt_path_dir}"
echo "Data path: ${DATA_PATH}/${split}/${split}.json.gz"

srun python -um goat.run \
  --run-type eval \
  --exp-config config/experiments/ver_instance_imagenav.yaml \
  habitat_baselines.num_environments=2 \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat_baselines.checkpoint_folder=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.cache_instance_imagegoal_sensor=cache_instance_imagegoal_sensor \
  ~habitat.task.lab_sensors.instance_imagegoal_sensor \
  habitat.task.lab_sensors.cache_instance_imagegoal_sensor.cache=data/datasets/iin/hm3d/v2/${split}_embeddings/ \
  habitat.task.lab_sensors.cache_instance_imagegoal_sensor.image_cache_encoder="CLIP" \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.simulator.type="OVONSim-v0" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split

touch $checkpoint_counter
