import torch
import torch.nn as nn
from cs336_basics.attention.rmsnorm import RMSNorm
from cs336_basics.attention.multihead_self_attention_with_rope import MultiHeadSelfAttentionRoPE
from cs336_basics.attention.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttentionRoPE(
            d_model, num_heads, max_seq_len, theta
        )

        self.ln2 = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):

        # Output after First RMSNorm & MultiHeadSelfAttentionRoPE
        out_1 = self.attn(self.ln1(x))

        # Residual
        out_1 = x + out_1

        # Pass Output to RMSNorm & SwiGLU
        out_2 = self.ffn(self.ln2(out_1))

        # Residual
        out_2 = out_1 + out_2

        return out_2