from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import os
from utils import *

def process_options():
  parser = argparse.ArgumentParser(description="Generate partial weight")
  parser.add_argument("--our_model_path", default=None, 
                      help='our OPT model')
  parser.add_argument("--skewing_matrix_path", default=None, 
                      help='path to skewing matrix')
  parser.add_argument("--model", default="facebook/opt-6.7b", 
                      help='model')
  parser.add_argument("--model_type", default = "opt", 
                      help='model arch (opt, llama)')
  parser.add_argument("--partial_weight_ratio", required=False, default=0.1, 
                      help='Ours: partial weight ratio')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  return parser
    
def main():
    ## get arguments
    parser = process_options()
    args = parser.parse_args()
    file_path = "./pg19_firstbook.txt"

    fname = f"modeling_{args.model_type}_ours_setup.py"
    set_symlink(args.model_type, fname)

    if args.our_model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(args.our_model_path, torch_dtype=torch.float16).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()

    if args.skewing_matrix_path is not None:
        A = torch.load(args.skewing_matrix_path).to('cuda').to(torch.float16)
        if args.model_type == 'llama':
            for layer_num, layer in enumerate(model.model.layers):
                layer.self_attn.skewing_matrix = A[layer_num]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    prompt = ["The bartender refused to serve the patron because the patron was drunk.\n\nThe girl politely declined the hamburger because she was a vegetarian.\n\nThe spy discovered the enemy's location because the spy bugged the enemy's phone.\n\nI tossed the ball upwards therefore the ball hit the ceiling.\n\nThe rider fell to the ground because the bull bucked the rider.\n\nThe pair of students came under scrutiny by the teacher because the students both received excellent grades."]
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    if args.model_type == "opt":
        for layer in model.model.decoder.layers:
            layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)
    elif args.model_type == "llama":
        for layer in model.model.layers:
            layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)

    print("Start Generation")
    
    generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    basepath = args.output + "/" + os.path.basename(os.path.normpath(args.model)) + "_%s"%(args.partial_weight_ratio)
    if not os.path.exists(basepath):
        os.system("mkdir -p %s"%(basepath))

    if args.model_type == "opt":
        for layer in range(len(model.model.decoder.layers)):
            partial_weight = model.model.decoder.layers[layer].self_attn.partial_weight_q
            torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")
    elif args.model_type == "llama":
        for layer in range(len(model.model.layers)):
            partial_weight = model.model.layers[layer].self_attn.partial_weight_q
            torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")

if __name__ == "__main__":
    main()
