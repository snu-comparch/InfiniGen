# Language Modeling Evaluation
This directory contains source code for evaluating the language modeling performance. You can reproduce perplexity and accuracy results in the paper. Some of the codes are referenced from the H2O (NeurIPS'23) GitHub repository.

## Getting Started (60 minutes)
We evaluate accuracy using the HuggingFace Transformers library. Also, please
fetch llama-2 from [here](https://llama.meta.com/llama-downloads). Follow the
instructions and set up Llama-2. You may also need to convert the model to
huggingface format using the `convert_llama_weight_to_hf.py` in
`transformers/src/transformers/models/llama`.

NOTE: We recommend using a GPU with a large VRAM size. We evaluate accuracy using `A100-80GB GPU`.

```sh
git clone -b v4.35-release https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

After setting up the library and llama models, generate the partial weights and skewing matrix. You can safely ignore the uninitialized weight warning.
```sh
cd setup
export LLAMA_PATH=/path/to/llama-2
bash setup.sh
```

For a "Hello world"-sized example, please run the following command (10 minutes):
```
cd lm_eval
mkdir results
python -u generate_task_data.py --output-file results/openbookqa-5.jsonl --task-name openbookqa --num-fewshot 5
bash ours.sh openbookqa ../setup/opt-model/opt-6.7b facebook/opt-6.7b opt 5 0.2 4 1.0 0.2 
```

## Run Experiments (40 hours)
You can reproduce the experimental results from Figure 11-13 and Table 2 by running the following commands:

```sh
cd scripts
sh run_all.sh
```

If you want to reproduce the results for a specific figure, please `sh run.sh` in each corresponding directory. For example,
```
cd scripts/figure11.sh
sh run.sh
```
