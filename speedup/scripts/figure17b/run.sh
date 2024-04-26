FLEXGEN_PATH=$PWD/../../flexgen
rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../infinigen/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../infinigen/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py

for PARTIAL_WEIGHT_RATIO in 0.1 0.2 0.4 0.6 0.8 1.0
do
  CMD="--model huggingface/opt-13b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"
  CMD=$CMD" --alpha 4 --partial-weight-ratio $PARTIAL_WEIGHT_RATIO --max-num-kv 409"
  python -m flexgen.flex_opt $CMD
done
