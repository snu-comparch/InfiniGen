from transformers import LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import os

def set_symlink(model_type, fname):
    model_path = "../transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("../src/" + fname)
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        exit(0)
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        exit(0)
    curr_dir = os.getcwd()
    os.chdir(model_path)
    if os.path.exists(f'modeling_{model_type}.py'):
        cmd = f"rm modeling_{model_type}.py"
        os.system(cmd)
    cmd = f"ln -s {linker_path} modeling_{model_type}.py"
    os.system(cmd)
    os.chdir(curr_dir)

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_wikitext2(nsamples, seed, seqlen, model):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    try: 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc

def get_ptb(nsamples, seed, seqlen, model):
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    try: 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')
    return testenc

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model)
