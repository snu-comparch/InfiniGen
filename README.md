# InfiniGen
## Overview
This repository is for the OSDI'24 artifact evaluation of paper "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management".
- Getting Started (10 minutes)
- Run Experiments

## Getting Started (10 minutes)
```sh
git clone https://github.com/snu-comparch/infinigen
conda create -n infinigen python=3.9
conda activate infinigen
pip install -r requirements.txt
```
## Run Experiments
We provide scripts for accuracy and speedup evaluations.
You can find source codes for accuracy evaluation in `accuracy` directory and speedup evaluation in `speedup` direcotry.
Experiments for accuracy takes around 40 hours, while speedup evalution takes 7 hours.
