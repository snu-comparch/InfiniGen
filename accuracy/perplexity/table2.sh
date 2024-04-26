#!/bin/bash

partial=0.2
seqlen=2048

## OPT
alpha=4
budget=0.2

for size in 6.7b 13b 30b;do
  for dataset in "wikitext2" "ptb";do
    echo opt-$size ${dataset} 100% cache
    python opt.py --model "../setup/opt-model/opt-${size}" \
      --eval_dataset ${dataset} \
      --seq_len ${seqlen} \
      --eval_samples 0 \
      --model_name "opt-${size}" \
      --infinigen \
      --partial_weight_ratio ${partial} \
      --partial_weight_path "../setup/weights/opt-${size}_${partial}" \
      --alpha ${alpha} \
      --budget ${budget} \
      --capacity 1.0
  done
done

for size in 6.7b 13b 30b;do
  for dataset in "wikitext2" "ptb";do
    for evict in fifo lru counter;do
      echo opt-$size ${dataset} 80% cache evict ${evict}
      python opt.py --model "../setup/opt-model/opt-${size}" \
      --eval_dataset ${dataset} \
        --seq_len ${seqlen} \
        --eval_samples 0 \
        --model_name "opt-${size}" \
        --infinigen \
        --partial_weight_ratio ${partial} \
        --partial_weight_path "../setup/weights/opt-${size}_${partial}" \
        --alpha ${alpha} \
        --budget ${budget} \
        --capacity 0.8 \
        --eviction_policy ${evict}
    done
  done
done

## Llama-2
alpha=5
budget=0.2

for size in 7b 13b;do
  for dataset in "wikitext2" "ptb";do
    echo llama-2-${size} ${dataset} 100% cache
    python llama.py --model "${LLAMA_PATH}/llama-2-${size}" \
      --eval_dataset ${dataset} \
      --seq_len ${seqlen} \
      --eval_samples 0 \
      --model_name "llama-${size}" \
      --infinigen \
      --partial_weight_ratio ${partial} \
      --partial_weight_path "../setup/weights/llama-2-${size}_${partial}" \
      --skewing_matrix_path "../setup/skewing_matrix/llama-2-${size}.pt" \
      --alpha ${alpha} \
      --budget ${budget} \
      --capacity 1.0 
  done
done

for size in 7b 13b;do
  for dataset in "wikitext2" "ptb";do
    for evict in fifo lru counter;do
      echo llama-2-${size} ${dataset} 80% cache evict ${evict}
      python llama.py --model "${LLAMA_PATH}/llama-2-${size}" \
        --eval_dataset ${dataset} \
        --seq_len ${seqlen} \
        --eval_samples 0 \
        --model_name "llama-${size}" \
        --infinigen \
        --partial_weight_ratio ${partial} \
        --partial_weight_path "../setup/weights/llama-2-${size}_${partial}" \
        --skewing_matrix_path "../setup/skewing_matrix/llama-2-${size}.pt" \
        --alpha ${alpha} \
        --budget ${budget} \
        --capacity 0.8 \
        --eviction_policy ${evict}
    done
  done
done
