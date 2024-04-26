import torch
import torch.nn as nn
import copy
import argparse
import math
from datautils import *


def get_opt_base(model, seq_len):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map = 'cpu')
    
    model.seqlen = seq_len
    return model

@torch.no_grad()
def opt_eval(model, testenc, eval_samples, ours, print_chunk = False):
    print('Evaluating ...')
    
    dev = torch.device('cuda:0')
    testenc = testenc.input_ids
    if eval_samples:
        nsamples = eval_samples
    else:
        nsamples = min(1000, testenc.numel() // model.seqlen)
    print("nsamples: ", nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
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
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    prev_hidden = []

    density = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if ours:
                if i >= 2:
                    layer.self_attn.previous_hidden_states = prev_hidden[j]
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            if ours:
                if i >= 1:
                    cur_bsz = layer.self_attn.current_hidden_states.shape[0]
                    cur_tgt_len = layer.self_attn.current_hidden_states.shape[1]
                    cur_device = layer.self_attn.current_hidden_states.device
                    cur_dtype = layer.self_attn.current_hidden_states.dtype
                    if i == 1:
                        prev_hidden.append(torch.cat((layer.self_attn.current_hidden_states, torch.ones(cur_bsz, cur_tgt_len, 1).to(cur_device).to(cur_dtype)), dim = -1))
                    else:
                        prev_hidden[j] = torch.cat((layer.self_attn.current_hidden_states, torch.ones(cur_bsz, cur_tgt_len, 1).to(cur_device).to(cur_dtype)), dim = -1)

            if ours and layer.self_attn.density is not None:
                density.append(layer.self_attn.density)

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(i, end=' ',flush=True)
    print()

    if ours:
        density = sum(density) / len(density) * 100
        retain_ratio = (1 - math.sqrt(1 - (density/100))) * 100
        #print("density %f"%(density))
        print("retain ratio %f"%((retain_ratio)))

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
#        loss_fct = nn.CrossEntropyLoss()
#        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#        neg_log_likelihood = loss.float() * model.seqlen
        loss_fct = nn.CrossEntropyLoss(reduction='none')
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

    parser.add_argument('--model', type=str, help='OPT model to load; pass `facebook/opt-X`.')
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
        set_symlink("opt", "modeling_opt_ours.py")
    else:
        set_symlink("opt", "modeling_opt_orig.py")

    model = get_opt_base(args.model, args.seq_len)
    
    ## H2O
    if args.heavy_ratio is not None:
        import sys
        sys.path.append("../lm_eval")
        from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
        
        model.config.heavy_ratio = args.heavy_ratio
        model.config.recent_ratio = args.recent_ratio
        if not os.path.exists(f"../h2o_model/{args.model_name}.pt"):
            os.system("mkdir ../h2o_model")
            checkpoint = copy.deepcopy(model.state_dict())
            torch.save(checkpoint, f"../h2o_model/{args.model_name}.pt")
        model = convert_kvcache_opt_heavy_recent(model, model.config)
        model.load_state_dict(torch.load(f"../h2o_model/{args.model_name}.pt"))
        model = model.to(torch.float16)
    
    ## InfiniGen
    if args.infinigen:
        for layer in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
            model.model.decoder.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt", map_location = 'cuda')

        for layer in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[layer].self_attn.alpha = args.alpha
            model.model.decoder.layers[layer].self_attn.budget = args.budget
            model.model.decoder.layers[layer].self_attn.capacity = args.capacity
            model.model.decoder.layers[layer].self_attn.eviction_policy= args.eviction_policy


    model.eval()
    
    dataset = args.eval_dataset
    testloader = get_loaders(dataset, model="facebook/" + args.model_name, seqlen=model.seqlen)

    opt_eval(model, testloader, args.eval_samples, args.infinigen, args.print_blk_ppl)
