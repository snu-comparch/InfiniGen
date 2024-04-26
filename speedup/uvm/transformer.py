import torch
import transformerlayer
import time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '--embed_dim', type=int, default=4096,
    help='model dim.'
)
parser.add_argument(
    '--ffn_dim', type=int, default=4096*4,
    help='ffn dim'
)
parser.add_argument(
    '--enable_bias', action='store_true',
    help='enable bias for projections'
)
parser.add_argument(
    '--n_head', type=int, default=32,
    help='num heads'
)
parser.add_argument(
    '--do_layer_norm_before', action='store_true',
    help='do layernorm before attention/ffn'
)
parser.add_argument(
    '--n_layer', type=int, default=32,
    help='num layers'
)
parser.add_argument(
    '--bsz', type=int, default=4,
    help='batch size'
)
parser.add_argument(
    '--prompt_len', type=int, default=2048,
    help='length of input prompt'
)
parser.add_argument(
    '--gen_len', type=int, default=1024,
    help='lenght of output'
)
parser.add_argument(
    '--is_h2o', action='store_true',
    help='enable h2o'
)
parser.add_argument(
    '--h2o_ratio', type=float, default=0.2,
    help='ratio of heavy hitter'
)
parser.add_argument(
    '--runs', type=int, default=1,
    help='number of runs'
)
args = parser.parse_args()

### Parameters ###
embed_dim = args.embed_dim
ffn_dim = args.ffn_dim
bias = args.enable_bias
n_head = args.n_head
do_layer_norm_before = args.do_layer_norm_before
n_layer = args.n_layer

bsz = args.bsz
prompt_len = args.prompt_len
gen_len = args.gen_len
##################

new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        'allocate.so', 'uvm_malloc', 'uvm_free')
torch.cuda.memory.change_current_allocator(new_alloc)

##################
"""
def prefetch(name):
  def hook(attn, input, output):
    ind = output[2]
    if int(name) < (n_layer - 1) and ind is not None:
        next_layer = transformer[int(name) + 1]
        torch.gather(next_layer.past_key_value[0], 2, ind)
        torch.gather(next_layer.past_key_value[1], 2, ind)

  return hook

for layer_num in range(n_layer):
    attn_in = transformer[layer_num].self_attn.register_forward_hook(prefetch("%d"%(layer_num)))
"""
#################

total_prefill = 0.0
total_decode = 0.0
for run in range(args.runs):
    transformer = [transformerlayer.TransformerLayer(embed_dim, ffn_dim, bias, n_head, do_layer_norm_before, args.is_h2o, args.h2o_ratio) for _ in range(n_layer)]

    prompt = torch.rand(bsz, prompt_len, embed_dim).to(torch.float16).to('cuda')
    new_input = torch.rand(bsz, 1, embed_dim).to(torch.float16).to('cuda')

    # Warmup
    for i in range(n_layer):
        prompt = transformer[i].forward(prompt)

    for i in range(n_layer):
        transformer[i].self_attn.past_key_value = None
        if args.is_h2o:
            transformer[i].self_attn.i = 0
            transformer[i].self_attn.acc = None

    start = time.time()
    for i in range(n_layer):
        prompt = transformer[i].forward(prompt)

    prefill_time = time.time() - start
    start = time.time()

    for k in range(gen_len - 1):
        for i in range(n_layer):
            new_input = transformer[i].forward(new_input)

    decode_time = time.time() - start

    total_prefill += prefill_time
    total_decode += decode_time

    del transformer

prefill_time = total_prefill / float(args.runs)
decode_time = total_decode / float(args.runs)

print("+++++++++++++++++++++++++++++++++++++++++++++++++")
if args.is_h2o:
    print("UVM + H2O")
else:
    print("UVM")

print("input: " + str(prompt_len) + " output: " + str(gen_len) + " bsz: " + str(bsz))
print("+++++++++++++++++++++++++++++++++++++++++++++++++")
print("Total: " + str(prefill_time + decode_time) + " Prefill: " + str(prefill_time) + " Decode: " + str(decode_time))
print("=================================================")
