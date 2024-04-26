import torch
from torch import nn
from typing import Tuple

class TransformerLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            bias,
            n_head, 
            do_layer_norm_before,
            is_h2o,
            h2o_ratio
            ):
        super().__init__()
        self.embed_dim = embed_dim
        if is_h2o:
            import h2o_attention
            self.self_attn = h2o_attention.SelfAttention(
                embed_dim=self.embed_dim,
                num_heads=n_head,
                bias=bias,
                h2o_ratio=h2o_ratio
            )
        else:
            import selfattention
            self.self_attn = selfattention.SelfAttention(
                embed_dim=self.embed_dim,
                num_heads=n_head,
                bias=bias,
            )

        self.do_layer_norm_before = do_layer_norm_before
        self.activation_fn = nn.ReLU()

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=True, dtype=torch.float16, device=torch.device('cuda')
        )
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias, dtype=torch.float16, device=torch.device('cuda'))
        self.final_layer_norm = nn.LayerNorm(
                self.embed_dim, elementwise_affine=True, dtype=torch.float16, device=torch.device('cuda')
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        with torch.no_grad():
            residual = hidden_states

            # OPT: 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states
            )
            hidden_states = residual + hidden_states

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            # Fully Connected
            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states

            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)

            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

            hidden_states = self.fc2(hidden_states)

            hidden_states = (residual + hidden_states).view(hidden_states_shape)

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states
