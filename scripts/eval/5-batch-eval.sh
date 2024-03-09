#!/bin/bash

split=$1
ckpt_interval=$2
ckpt_dir=$3
tb_dir=$4
baseline_name=$5


count=4
num_ckpt_files=`ls ${ckpt_dir}/*.pth | wc -l`

echo "Num ckpt files: $num_ckpt_files, Interval: $ckpt_interval"
for (( i=$count; i<=$num_ckpt_files; i+=$ckpt_interval ));
do
  uuid="ckpt_${count}"

  tensorboard_dir="${tb_dir}/${uuid}"
  current_ckpt_dir="${ckpt_dir}/ckpt.${i}.pth"


  echo "Ckpt id: $uuid - $i, ${tensorboard_dir}, ${current_ckpt_dir}"

  #echo "Submitting ${uuid}..."
  sbatch --job-name=goat-${split}-${count} \
    --output=slurm_logs/eval/${baseline_name}-${split}-${count}.out \
    --error=slurm_logs/eval/${baseline_name}-${split}-${count}.err \
    --cpus-per-task 6 \
    --exclude="fiona,irona,vicki,ephemeral-3,alexa,sonny,xaea-12" \
    --export=ALL,eval_ckpt_path_dir=$current_ckpt_dir,tensorboard_dir=$tensorboard_dir,split=$split \
    scripts/eval/2-goat-eval.sh
    # scripts/eval/1-languagenav-eval.sh
  count=$((count + $ckpt_interval))
done

