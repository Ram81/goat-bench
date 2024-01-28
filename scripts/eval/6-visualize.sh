#!/bin/bash

scenes_dir=$1
data_path=$2
output_path=$3
cache=$4

count=0

glb_files=`ls ${scenes_dir}/*/*.semantic.glb`
for i in ${glb_files[@]}
do
  scene_id=`basename $i`
  base=${scene_id%.*}  # remove .gz
  base=${base%.*}  # remove .json

  echo "Scene: $base, ${data_path}"

  #echo "Submitting ${uuid}..."
  sbatch --job-name=goat \
    --output=slurm_logs/eval/visual-${base}.out \
    --error=slurm_logs/eval/visual-${base}.err \
    --gpus 1 \
    --cpus-per-task 6 \
    --constraint "a40|2080_ti|rtx_6000" \
    --exclude="fiona,irona,vicki,ephemeral-3,alexa" \
    --export=ALL,scene=$base,output_path=$output_path,data_path=$data_path,cache=$cache \
    scripts/eval/6-visualize-trajectory.sh
  count=$((count + 1))
done

