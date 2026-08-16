import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RoPE

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.max_seq_len = max_seq_len
        self.theta = theta

        std = (2 / (d_model + d_model)) ** 0.5

        self.q_proj_weight = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_model, d_model), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )

        self.k_proj_weight = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_model, d_model), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )

        self.v_proj_weight = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_model, d_model), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )

        self.o_proj_weight = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_model, d_model), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )

        self.RoPE = RoPE(self.d_k, theta, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):

        seq_len = x.shape[-2]

        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype = torch.bool),
            diagonal = 0
        )

        # Make Sure Mask and x on the Same Device
        mask = mask.to(x.device)

        Q = einsum(
            x, self.q_proj_weight,
            "... d_in, d_out d_in -> ... d_out"
        )

        Q_multi = rearrange(
            Q,
            "... seq_len (head d_k) -> ... head seq_len d_k", head = self.num_heads
        )
        
        Q_RoPE = self.RoPE(Q_multi, token_positions)

        K = einsum(
            x, self.k_proj_weight,
            "... d_in, d_out d_in -> ... d_out"
        )

        K_multi = rearrange(
            K,
            "... seq_len (head d_k) -> ... head seq_len d_k", head = self.num_heads
        )

        K_RoPE = self.RoPE(K_multi, token_positions)

        V = einsum(
            x, self.v_proj_weight,
            "... d_in, d_out d_in -> ... d_out"
        )

        V_multi = rearrange(
            V,
            "... seq_len (head d_k) -> ... head seq_len d_k", head = self.num_heads
        )

        # Shape (..., head, seq_len, d_k)
        out = scaled_dot_product_attention(Q_RoPE, K_RoPE, V_multi, mask)

        out = rearrange(
            out,
            "... head seq_len d_k -> ... seq_len (head d_k)"
        )

        out = out @ self.o_proj_weight.T

        return out