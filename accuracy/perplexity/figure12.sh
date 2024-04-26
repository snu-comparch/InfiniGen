#!/bin/bash

# InfiniGen
echo "== InfiniGen =="
partial=0.2
seqlen=2048
alpha=4.0
budget=0.2
echo opt-13b alpha $alpha budget $budget
python opt.py --model "../setup/opt-model/opt-13b" \
  --eval_dataset "wikitext2" \
  --seq_len ${seqlen} \
  --eval_samples 0 \
  --model_name "opt-13b" \
  --infinigen \
  --print_blk_ppl \
  --partial_weight_ratio ${partial} \
  --partial_weight_path "../setup/weights/opt-13b_${partial}" \
  --alpha ${alpha} \
  --budget ${budget} \
  --capacity 1.0 

partial=0.2
seqlen=4096
alpha=5.2
budget=0.2
echo llama-2 alpha $alpha budget $budget
python llama.py --model "${LLAMA_PATH}/llama-2-13b" \
  --eval_dataset "wikitext2" \
  --seq_len ${seqlen} \
  --eval_samples 0 \
  --model_name "llama-2-13b" \
  --infinigen \
  --print_blk_ppl \
  --partial_weight_ratio ${partial} \
  --partial_weight_path "../setup/weights/llama-2-13b_${partial}" \
  --skewing_matrix_path "../setup/skewing_matrix/llama-2-13b.pt" \
  --alpha ${alpha} \
  --budget ${budget} \
  --capacity 1.0 

echo "==============="

# H2O 
echo "==    H2O   =="
partial=0.2
seqlen=2048
heavy=0.01875
recent=0.01875
echo opt-13b heavy $heavy recent $recent
python opt.py --model "facebook/opt-13b" \
  --eval_dataset "wikitext2" \
  --seq_len ${seqlen} \
  --eval_samples 0 \
  --model_name "opt-13b" \
  --print_blk_ppl \
  --heavy_ratio ${heavy} \
  --recent_ratio ${recent}

partial=0.2
seqlen=4096
heavy=0.01875
recent=0.01875
echo llama-2-13b heavy $heavy recent $recent
python llama.py --model "${LLAMA_PATH}/llama-2-13b" \
  --eval_dataset "wikitext2" \
  --seq_len ${seqlen} \
  --eval_samples 0 \
  --model_name "llama-2-13b" \
  --print_blk_ppl \
  --heavy_ratio ${heavy} \
  --recent_ratio ${recent}

echo "==============="
