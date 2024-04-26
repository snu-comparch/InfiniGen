#!/bin/bash

shots=5
# Prepare dataset
echo "prepare dataset"
for task in openbookqa; do
  python -u generate_task_data.py \
  --output-file "results/${task}-${shots}.jsonl" \
  --task-name ${task} \
  --num-fewshot ${shots} 
done


# Threshold (alpha) sweep
partial=0.2
capacity=1.0
budget=0.2

echo "Threshold (alpha) sweep"
for alpha in 1 2 3 4 5 6 7 8 9; do
  echo alpha "${alpha}"
  bash ours.sh winogrande "../setup/opt-model/opt-13b" "facebook/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget}
done

# Partial weight sweep
alpha=4
capacity=1.0
budget=0.2

echo "Partial weight ratio sweep"
for partial in 0.1 0.2 0.4 0.6 0.8 1; do
  echo partial ratio "${partial}"
  bash ours.sh winogrande "../setup/opt-model/opt-13b" "facebook/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget}
done

