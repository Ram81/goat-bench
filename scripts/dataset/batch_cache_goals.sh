#!/bin/bash

config_path=$1
input_path=$2
output_path=$3
split=$4

echo "Config: $config_path"
echo "Config: $input_path"
echo "Config: $output_path"
echo "Config: $split"

gz_files=`ls ${input_path}/${split}/content/*.json.gz`
for i in ${gz_files[@]}
do
  scene_id=`basename $i`
  base=${scene_id%.*}  # remove .gz
  base=${base%.*}  # remove .json

  if [ -f "${output_path}/${base}_embeddings.pkl" ]; then
    echo "Skipping ${base}"
    continue
  fi

  echo "Submitting ${base}"
  sbatch --job-name=$split-${base} \
  --output=slurm_logs/dataset/iin-$split-${base}.out \
  --error=slurm_logs/dataset/iin-$split-${base}.err \
  --gpus 1 \
  --cpus-per-task 6 \
  --export=ALL,scene=$base,config=$config_path,split=$split,input_path=$input_path,output_path=$output_path \
  scripts/dataset/cache_goal_embeddings.sh
done

