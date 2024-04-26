#!/bin/bash

# Inference, and generate output json file
task=$1
shots=$5
model_path=$2
model=$3
model_arch=$4
partial_weight=$6
alpha=$7
capacity=$8
budget=$9
no_skewing=${10}
base_name=$(basename "${model}")
if [ -z $no_skewing ]; then
  weight_path="../setup/weights/${base_name}_${partial_weight}"
else 
  weight_path="../setup/weights-no-skew/${base_name}_${partial_weight}"
fi
skewing_path="../setup/skewing_matrix/${base_name}.pt"

python -u run_lm_eval_harness.py \
  --input-path results/${task}-${shots}.jsonl \
  --output-path results/${task}-${shots}-${base_name}-ours.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --partial_weight_ratio ${partial_weight} \
  --partial_weight_path ${weight_path} \
  --ours \
  --model-path ${model_path} \
  --skewing_matrix_path ${skewing_path} \
  --alpha ${alpha} \
  --capacity ${capacity} \
  --budget ${budget}

# Evaluate results
python -u evaluate_task_result.py \
  --result-file results/${task}-${shots}-${base_name}-ours.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}

rm results/${task}-${shots}-${base_name}-ours.jsonl
