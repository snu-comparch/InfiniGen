#!/bin/bash

# Inference, and generate output json file
task=$1
shots=$4
model=$2
model_arch=$3
base_name=$(basename "${model}")

python -u run_lm_eval_harness.py \
  --input-path results/${task}-${shots}.jsonl \
  --output-path results/${task}-${shots}-${base_name}-full.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}

# Evaluate results
python -u evaluate_task_result.py \
  --result-file results/${task}-${shots}-${base_name}-full.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}

rm results/${task}-${shots}-${base_name}-full.jsonl
