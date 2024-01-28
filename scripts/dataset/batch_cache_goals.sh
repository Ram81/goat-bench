#!/bin/bash

config_path=$1
input_path=$2
output_path=$3
split=$4
encoder=$5

echo "Config: $config_path"
echo "Config: $input_path"
echo "Config: $output_path"
echo "Config: $split, Encoder: $encoder"

gz_files=`ls ${input_path}/${split}/content/*.json.gz`
for i in ${gz_files[@]}
do
  scene_id=`basename $i`
  base=${scene_id%.*}  # remove .gz
  base=${base%.*}  # remove .json

  if [ -f "${output_path}/${base}_${encoder}_goat_embedding.pkl" ]; then
    echo "Skipping ${base}"
    continue
  fi

  echo "Submitting ${base}"
  sbatch --job-name=$split-${base} \
  --output=slurm_logs/dataset/iin-$split-${base}.out \
  --error=slurm_logs/dataset/iin-$split-${base}.err \
  --gpus 1 \
  --cpus-per-task 6 \
  --exclude="fiona,irona,vicki,ephemeral-3" \
  --export=ALL,scene=$base,config=$config_path,split=$split,input_path=$input_path,output_path=$output_path,encoder=$encoder \
  scripts/dataset/cache_goal_embeddings.sh
done

