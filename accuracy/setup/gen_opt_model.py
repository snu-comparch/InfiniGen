from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
import os
from utils import *

def process_options():
  parser = argparse.ArgumentParser(description="OPT Model")
  parser.add_argument("--model", default="facebook/opt-6.7b", 
                      help='OPT model to load')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  parser.add_argument("--no_skewing", action='store_true', 
                      help='whether to skew weight')
  return parser

def main():
    parser = process_options()
    args = parser.parse_args()

    ### Model load
    set_symlink("opt", "modeling_opt_orig.py")

    model_name = os.path.basename(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    head_dim = model.model.decoder.layers[0].self_attn.head_dim
    n_head = model.model.decoder.layers[0].self_attn.num_heads

    ### Add hook
    query_v = {}
    key_v = {}

    def get_query(name):
      def hook(model, input, output):
        query_v[name] = output
      return hook
    def get_key(name):
      def hook(model, input, output):
        key_v[name] = output
      return hook

    for i, layer in enumerate(model.model.decoder.layers):
        query = layer.self_attn.q_proj.register_forward_hook(get_query("%d"%(i)))
        key = layer.self_attn.k_proj.register_forward_hook(get_key("%d"%(i)))

    ### Generation
    file_path = "./pg19_firstbook.txt"

    with open(file_path, 'r') as file:
        prompt = file.read()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[:, :2048]

    print("Start Generation")

    generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    ### New weight generation
    for name in query_v:
        layer = int(name)
        query = query_v[name][0]
        query = query * (head_dim ** -0.5)
        key = key_v[name][0]

        wq = model.model.decoder.layers[layer].self_attn.q_proj.weight.data
        bq = model.model.decoder.layers[layer].self_attn.q_proj.bias.data
        wk = model.model.decoder.layers[layer].self_attn.k_proj.weight.data
        bk = model.model.decoder.layers[layer].self_attn.k_proj.bias.data
        
        new_wq = torch.cat((wq.transpose(-1,-2), bq.unsqueeze(0)), dim = 0) * (head_dim**-0.5)
        new_wk = torch.cat((wk.transpose(-1,-2), bk.unsqueeze(0)), dim = 0)
        
        if not args.no_skewing:
            for h in range(n_head):
                start = h * head_dim
                end = (h+1) * head_dim
                uq, sq, vq = torch.svd(query[:, start:end].to(torch.float))
                uk, sk, vk = torch.svd(key[:, start:end].to(torch.float))
                uq = uq.to(torch.float16)
                sq = sq.to(torch.float16)
                vq = vq.to(torch.float16)
                uk = uk.to(torch.float16)
                sk = sk.to(torch.float16)
                vk = vk.to(torch.float16)
                s = sq * sk

                A = torch.zeros(head_dim, head_dim).to('cuda').to(torch.float16)
                _, ind = s.sort()
                r,c = A.shape
                A = A.scatter(-1, ind.unsqueeze(0).repeat(r,1), vq) 
                new_wq[:, start:end] = new_wq[:, start:end] @ A
                new_wk[:, start:end] = new_wk[:, start:end] @ A

        model.model.decoder.layers[layer].self_attn.q_proj.weight.data = new_wq
        model.model.decoder.layers[layer].self_attn.k_proj.weight.data = new_wk

    save_dir = args.output + "/" + model_name
    if not os.path.exists(save_dir):
        os.system(f"mkdir -p {save_dir}")

    model.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
