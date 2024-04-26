import torch
from torch import nn
from typing import Tuple

class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
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
        
        self.past_key_value = None
        self.src_s = 0


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        if tgt_len > 1:
            self.src_s = tgt_len
        else:
            self.src_s += 1

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        
        # get key/value proj
        if self.past_key_value is not None:
            # reuse k, v, self_attention
            k = self._shape(self.k_proj(hidden_states), -1, bsz).squeeze()
            v = self._shape(self.v_proj(hidden_states), -1, bsz).squeeze()
            key_states = self.past_key_value[0, :, :, :self.src_s]
            key_states[:, :, -1] = k
            self.past_key_value[0, :, :, self.src_s] = k
            value_states = self.past_key_value[1, :, :, :self.src_s]
            value_states[:, :, -1] = v
            self.past_key_value[1, :, :, self.src_s] = k
        else:
            # self_attention
            self.past_key_value = torch.zeros((2, bsz, self.num_heads, 2048, self.head_dim), dtype=torch.float16, device=torch.device('cuda')) 
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            self.past_key_value[0, :, :, :tgt_len] = key_states
            self.past_key_value[1, :, :, :tgt_len] = value_states

        # update kv cache
        #self.past_key_value = (key_states, value_states)

        # reshape
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # qkt
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        # masking
        if attn_weights.shape[1] > 1: # prefill
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

        return attn_output
