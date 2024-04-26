from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
import os
from utils import *

### Parameters

def process_options():
  parser = argparse.ArgumentParser(description="Llama-2 Model")
  parser.add_argument("--model", required=True, 
                      help='Llama-2 model to load')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  return parser

def main():
    parser = process_options()
    args = parser.parse_args()

    ### Model load
    set_symlink("llama", "modeling_llama_orig.py")

    model_name = os.path.basename(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    head_dim = model.model.layers[0].self_attn.head_dim
    n_head = model.model.layers[0].self_attn.num_heads
    n_layer = config.num_hidden_layers

    ### Generation
    file_path = "./pg19_firstbook.txt"

    with open(file_path, 'r') as file:
        prompt = file.read()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[:, :2048]

    print("Start Generation")

    generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    query_v = {}
    key_v = {}

    for i, layer in enumerate(model.model.layers):
        query_v[str(i)] = layer.self_attn.rope_query
        key_v[str(i)] = layer.self_attn.rope_key

    ### Gen Skewing Matrix A
    A = torch.zeros(n_layer, n_head, head_dim, head_dim).to('cuda').to(torch.float16)
    for name in query_v:
        layer = int(name)
        query = query_v[name]
        key = key_v[name]

        for head in range(n_head):
            in_q = query[0, head]
            in_k = key[0, head]
            uq, sq, vq = torch.svd(in_q.to(torch.float))
            uk, sk, vk = torch.svd(in_k.to(torch.float))
            s = sq * sk
            a = torch.zeros(head_dim, head_dim).to('cuda')
            _, ind = s.sort()
            r,c = a.shape
            A[layer, head] = a.scatter(-1, ind.unsqueeze(0).repeat(r,1), vq).to(torch.float16)

    save_dir = args.output
    if not os.path.exists(save_dir):
        os.system(f"mkdir -p {save_dir}")
    torch.save(A, save_dir + "/" + model_name + ".pt")

if __name__ == "__main__":
    main()
