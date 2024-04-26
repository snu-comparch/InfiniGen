#!/bin/bash

CWD=${PWD}
cd ../transformers/src/transformers/models

for model in llama opt;do
  mv ${model}/modeling_${model}.py ${model}/modeling_${model}_orig.py
done

cd ${CWD}

# ========= InfiniGen ============
# generate opt models w/skewing
for size in 6.7b 13b 30b;do
  python gen_opt_model.py \
    --model "facebook/opt-${size}" \
    --output "./opt-model"
done

# generate skewing matrices for llama
for size in 7b 13b;do
  python gen_llama_skewing_matrix.py \
    --model "${LLAMA_PATH}/llama-2-${size}" \
    --output "./skewing_matrix" 
done


# generate partial weight matrices for prediction
PARTIAL_RATIO=0.2
# opt
for size in 6.7b 13b 30b;do
  python gen_partial_weight.py \
    --our_model_path "./opt-model/opt-${size}" \
    --model "facebook/opt-${size}" \
    --model_type "opt" \
    --partial_weight_ratio $PARTIAL_RATIO \
    --output "./weights"
done

# llama
for size in 7b 13b;do
  python gen_partial_weight.py \
    --skewing_matrix_path "./skewing_matrix/llama-2-${size}.pt" \
    --model "${LLAMA_PATH}/llama-2-${size}" \
    --model_type "llama" \
    --partial_weight_ratio $PARTIAL_RATIO \
    --output "./weights"
done


# ========= w/o skewing (figure 13)
PARTIAL_RATIO=0.1
python gen_partial_weight.py \
  --our_model_path "./opt-model/opt-6.7b" \
  --model "facebook/opt-6.7b" \
  --model_type "opt" \
  --partial_weight_ratio $PARTIAL_RATIO \
  --output "./weights"

python gen_opt_model.py \
  --model "facebook/opt-6.7b" \
  --output "./opt-model-no-skew" \
  --no_skewing

python gen_partial_weight.py \
  --our_model_path "./opt-model-no-skew/opt-6.7b" \
  --model "facebook/opt-6.7b" \
  --model_type "opt" \
  --partial_weight_ratio $PARTIAL_RATIO \
  --output "./weights-no-skew"

# ========= partial weight sweep (figure 17)
for PARTIAL_RATIO in 0.1 0.4 0.6 0.8 1.0;do
  python gen_partial_weight.py \
    --our_model_path "./opt-model/opt-13b" \
    --model "facebook/opt-13b" \
    --model_type "opt" \
    --partial_weight_ratio $PARTIAL_RATIO \
    --output "./weights"
done
