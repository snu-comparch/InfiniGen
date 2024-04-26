import torch
import torch.nn as nn
import copy
import argparse
import math
from datautils import *


def get_llama(model, seqlen):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, device_map='cpu', torch_dtype=torch.float16)

    model.seqlen = seqlen
    return model


@torch.no_grad()
def llama_eval(model, testenc, dev, eval_sample, ours, print_chunk = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    if eval_sample:
        nsamples = eval_sample
    else:
        nsamples = min(1000, testenc.numel() // model.seqlen)
    print("nsamples: ", nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    prev_hidden = []

    density = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if ours:
                if i >= 2:
                    layer.self_attn.previous_hidden_states = prev_hidden[j]
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            if ours:
                if i >= 1:
                    cur_bsz = layer.self_attn.current_hidden_states.shape[0]
                    cur_tgt_len = layer.self_attn.current_hidden_states.shape[1]
                    cur_device = layer.self_attn.current_hidden_states.device
                    cur_dtype = layer.self_attn.current_hidden_states.dtype
                    if i == 1:
                        prev_hidden.append(layer.self_attn.current_hidden_states)
                    else:
                        prev_hidden[j] = layer.self_attn.current_hidden_states
            if ours and layer.self_attn.density is not None:
                density.append(layer.self_attn.density)
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(i, end=' ',flush=True)
    print()

    if ours:
        density = sum(density) / len(density) * 100
        retain_ratio = (1 - math.sqrt(1 - (density/100))) * 100
        #print("density %f"%(density))
        print("retain ratio %f"%((retain_ratio)))

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss(reduction='none')
#        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#        neg_log_likelihood = loss.float() * model.seqlen
        neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).to(torch.float)
        nlls.append(neg_log_likelihood)
    nlls = torch.stack(nlls)
    for seqlen in range(int(model.seqlen/256)):
        start = seqlen * 256
        end = (seqlen+1)*256
        per_token_ppl = torch.exp(nlls[:, start:end].sum(dim=0) / nsamples)
        if seqlen == 0:
            var = torch.var(per_token_ppl[25:], correction=0)
        else:
            var = torch.var(per_token_ppl, correction=0)
        q1 = torch.quantile(per_token_ppl, 0.25, interpolation='nearest')
        q2 = torch.quantile(per_token_ppl, 0.5, interpolation='nearest')
        q3 = torch.quantile(per_token_ppl, 0.75, interpolation='nearest')
        ppl = torch.exp(nlls[:, start:end].sum() / (nsamples*256))
        if print_chunk:
            print("seqlen: ", end)
            print("perplexity, variance, q1, q2, q3: ", ppl.item(), var.item(), q1.item(), q2.item(), q3.item())

    print("Total")
    ppl = torch.exp(nlls.sum() / (nsamples * model.seqlen))
    print("Perplexity: ", ppl.item())


    model.config.use_cache = use_cache

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='llama model to load; pass `/path/to/llama')
    parser.add_argument('--eval_dataset', type=str, help='evaluation dataset')
    parser.add_argument('--seq_len', type=int, help='model sequence length')
    parser.add_argument('--eval_samples', type=int, default=0, help='number of sample evaluation dataset')
    parser.add_argument('--model_name', type=str, help='name of the model')
    parser.add_argument('--print_blk_ppl', action='store_true', help='')
    
    ## H2O
    parser.add_argument("--heavy_ratio", type=float, default=None)
    parser.add_argument("--recent_ratio", type=float, default=None)

    ## InfiniGen 
    parser.add_argument('--infinigen', action='store_true', help='')
    parser.add_argument("--partial_weight_ratio", type=float, default=None)
    parser.add_argument("--partial_weight_path", type=str)
    parser.add_argument("--skewing_matrix_path", type=str)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--budget", type=float, default=0.0)
    parser.add_argument("--capacity", type=float, default=0.0)
    parser.add_argument("--eviction_policy", type=str, default="lru")
    
    args = parser.parse_args()

    if args.infinigen:
        set_symlink("llama", "modeling_llama_ours.py")
    else:
        set_symlink("llama", "modeling_llama_orig.py")
    
    model = get_llama(args.model, args.seq_len)
    
    ## H2O
    if args.heavy_ratio is not None:
        import sys
        sys.path.append("../lm_eval")
        from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
        
        model.config.heavy_ratio = args.heavy_ratio
        model.config.recent_ratio = args.recent_ratio
        if not os.path.exists(f"../h2o_model/{args.model_name}.pt"):
            os.system("mkdir ../h2o_model")
            checkpoint = copy.deepcopy(model.state_dict())
            torch.save(checkpoint, f"../h2o_model/{args.model_name}.pt")
        model = convert_kvcache_llama_heavy_recent(model, model.config)
        model.load_state_dict(torch.load(f"../h2o_model/{args.model_name}.pt"))
        model = model.to(torch.float16)
    
    ## InfiniGen
    if args.infinigen:
        A = torch.load(args.skewing_matrix_path)
        for layer in range(len(model.model.layers)):
            model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
            model.model.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
            model.model.layers[layer].self_attn.skewing_matrix = A[layer]

        for layer in range(len(model.model.layers)):
            model.model.layers[layer].self_attn.alpha = args.alpha
            model.model.layers[layer].self_attn.budget = args.budget
            model.model.layers[layer].self_attn.capacity = args.capacity
            model.model.layers[layer].self_attn.eviction_policy = args.eviction_policy

    model.eval()
    
    dataset = args.eval_dataset
    testloader = get_loaders(dataset, model=args.model, seqlen=model.seqlen)
    
    llama_eval(model, testloader, 'cuda', args.eval_samples, args.infinigen, args.print_blk_ppl)
