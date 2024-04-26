import torch
from torch import nn
from typing import Tuple

class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        h2o_ratio: float,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))

        self.acc = None
        self.ratio = h2o_ratio
        self.i = 0
        self.past_key_value = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def _heavy_hitter_pruning(self, k, v, attn_weights, hh_k):
        # k, v: (s, b * n_head, head_dim)
        # attn_weights: (b * n_head, s, s)
        aggr_attn = torch.sum(attn_weights, 1)
        # (b * n_head, hh_k)
        _, topk_indices = aggr_attn[:, :].topk(
            min(hh_k, aggr_attn.shape[1]), dim=1)

        # select heavy-hitters
        # k, v: (b * n_head, s, head_dim)
        k_t = k.transpose(1, 0)
        v_t = v.transpose(1, 0)
        dim0_indices = torch.arange(k_t.size(0))[:, None]
        dim0_indices = dim0_indices.expand_as(topk_indices)
        # (b * n_head, hh_k, head_dim)
        k_hh_t = k_t[dim0_indices, topk_indices]
        v_hh_t = v_t[dim0_indices, topk_indices]
        # (hh_k, b * n_head, head_dim)
        k = k_hh_t.transpose(1, 0)
        v = v_hh_t.transpose(1, 0)
        # new shape (hh_k, b * n_head)
        aggr_attn = aggr_attn.transpose(0, 1)
        dim1_indices = torch.arange(aggr_attn.size(1)).unsqueeze(0)
        # (hh_k * 2, b * n_head)
        acc = aggr_attn[topk_indices.transpose(0, 1), dim1_indices]
        return k, v, acc

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        
        # get key/value proj
        if self.past_key_value is not None:
            # reuse k, v, self_attention
            k = self._shape(self.k_proj(hidden_states), -1, bsz).squeeze()
            v = self._shape(self.v_proj(hidden_states), -1, bsz).squeeze()
            key_states = self.past_key_value[0]
            key_states[:, :, -1] = k
            value_states = self.past_key_value[1]
            value_states[:, :, -1] = v
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # update kv cache
        #past_key_value = (key_states, value_states)

        # reshape
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # qkt
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # masking
        if self.i == 0: # prefill
            mask = torch.triu(torch.ones(attn_weights.shape).to('cuda'), diagonal=1) * -10000
            attn_weights = attn_weights + mask

        # softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)

        # sv
        attn_output = torch.bmm(attn_weights, value_states)

        # reshape
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        ##### h2o ####
        if self.acc is None:
            self.hh = int(attn_weights.shape[-1] * self.ratio)
            key_states, value_states, self.acc = self._heavy_hitter_pruning(key_states.permute(1,0,2), value_states.permute(1,0,2), attn_weights, self.hh)
            key_states = key_states.permute(1, 0, 2)
            value_states = value_states.permute(1, 0, 2)
            self.past_key_value = (torch.cat((key_states.reshape(bsz, self.num_heads, key_states.shape[-2], key_states.shape[-1]), torch.zeros(bsz, self.num_heads, 1, key_states.shape[-1]).to('cuda').to(torch.float16)), dim = -2),
                              torch.cat((value_states.reshape(bsz, self.num_heads, value_states.shape[-2], value_states.shape[-1]), torch.zeros(bsz, self.num_heads, 1, key_states.shape[-1]).to('cuda').to(torch.float16)), dim = -2))

        else:
            temp_attn = attn_weights.squeeze(1).transpose(0, 1)
            self.acc = torch.cat((self.acc, torch.zeros(1, bsz * self.num_heads).to('cuda')), dim=0)
            self.acc = self.acc + temp_attn
            kick_ind = self.acc.argmin(dim=0).squeeze()

            # reduce accumulated result
            indices = kick_ind.unsqueeze(0)
            self.acc.scatter_(0, indices, self.acc[-1].unsqueeze(0).clone())
            self.acc = self.acc[:-1]

            # modify kv cache
            indices = kick_ind.view(-1, 1).expand(-1, self.head_dim).unsqueeze(1)
            key_states.scatter_(1, indices, key_states[:, -1].unsqueeze(1))
            value_states.scatter_(1, indices, value_states[:, -1].unsqueeze(1))
            #key_states = key_states[:, :-1]
            #value_states = value_states[:, :-1]
            self.past_key_value = (key_states.reshape(bsz, self.num_heads, key_states.shape[-2], key_states.shape[-1]),
                              value_states.reshape(bsz, self.num_heads, value_states.shape[-2], value_states.shape[-1]))
            
        self.i += 1
        return attn_output
