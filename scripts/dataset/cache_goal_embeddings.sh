#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/dataset-%j.out
#SBATCH --error=slurm_logs/dataset-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --constraint=2080_ti
#SBATCH --exclude=conroy,ig-88
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

# SPLIT="val"
# NUM_TASKS=$1
# OUTPUT_PATH=$2
# NUM_SCENES=-1

srun python goat/utils/cache_image_goals.py \
  --split $split \
  --config $config \
  --input-path $input_path \
  --output-path $output_path \
  --scene $scene \
  --encoder $encoder
