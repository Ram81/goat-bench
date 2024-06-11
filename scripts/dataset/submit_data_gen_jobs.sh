#!/bin/bash

scenes_dir=$1
split=$2
output_path=$3
task=$4
start_poses_per_object=2000
episodes_per_object=50

glb_files=`ls ${scenes_dir}/*/*.semantic.glb`
for i in ${glb_files[@]}
do
  scene_id=`basename $i`
  base=${scene_id%.*}  # remove .gz
  base=${base%.*}  # remove .json

  # if [ -f "${output_path}/${split}/content/${base}.pkl" ]; then
  #   echo "Skipping ${base}"
  #   continue
  # fi

  if [ -f "${output_path}/${split}/content/${base}.json.gz" ]; then
    echo "Skipping ${base}"
    continue
  fi

  echo "Submitting ${base}"
  if [[ $task == "iin" ]]; then
    sbatch --job-name=$split-${base} \
    --output=slurm_logs/dataset/iin-$split-${base}.out \
    --error=slurm_logs/dataset/iin-$split-${base}.err \
    --gpus 1 \
    --cpus-per-task 6 \
    --export=ALL,scene=$i,num_tasks=1,split=$split,num_scenes=1,output_path=$output_path,start_poses_per_object=$start_poses_per_object,episodes_per_object=$episodes_per_object \
    scripts/dataset/generate_iin.sh
  elif [[ $task == "lnav" ]]; then
    sbatch --job-name=$split-${base} \
    --output=slurm_logs/dataset/lnav-$split-${base}.out \
    --error=slurm_logs/dataset/lnav-$split-${base}.err \
    --gpus 1 \
    --cpus-per-task 6 \
    --export=ALL,scene=$i,num_tasks=1,split=$split,num_scenes=1,output_path=$output_path,start_poses_per_object=$start_poses_per_object,episodes_per_object=$episodes_per_object \
    scripts/dataset/generate_lnav.sh
  else
    echo "Invalid task: $task"
    exit 1
  fi
done

