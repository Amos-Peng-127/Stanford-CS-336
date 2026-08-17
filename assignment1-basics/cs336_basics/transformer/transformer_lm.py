import torch
import torch.nn as nn
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.embedding import Embedding
from cs336_basics.attention.rmsnorm import RMSNorm
from cs336_basics.transformer.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta
                ) for _ in range(num_layers)
            ]
        ) 

        self.rmsnorm = RMSNorm(d_model)

        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):

        # Embedding
        out = self.embedding(x)

        # Transformer Blocks
        for l in self.layers:
            out = l(out)

        # Norm
        out = self.rmsnorm(out)

        # Output Linear
        out = self.linear(out)

        # # Softmax
        # out = softmax(out, dim = -1)

        return out