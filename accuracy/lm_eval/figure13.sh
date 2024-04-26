#!/bin/bash

shots=5
# Prepare dataset
echo "prepare dataset"
for task in copa openbookqa winogrande piqa rte; do
  python -u generate_task_data.py \
  --output-file "results/${task}-${shots}.jsonl" \
  --task-name ${task} \
  --num-fewshot ${shots} 
done

# Baseline
echo "full cache"
for task in copa openbookqa winogrande piqa rte; do
  bash full_cache.sh ${task} "facebook/opt-6.7b" opt ${shots}
done

# InfiniGen
partial=0.1
capacity=1.0
alpha=99 
budget=0.2

# w/o skewing
echo "InfiniGen w/o skewing"
for task in copa openbookqa winogrande piqa rte; do
  bash ours.sh ${task} "../setup/opt-model-no-skew/opt-6.7b" "facebook/opt-6.7b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget} "no-skew"
done


# w/ skewing
echo "InfiniGen w/ skewing"
for task in copa openbookqa winogrande piqa rte; do
  bash ours.sh ${task} "../setup/opt-model/opt-6.7b" "facebook/opt-6.7b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget}
done
