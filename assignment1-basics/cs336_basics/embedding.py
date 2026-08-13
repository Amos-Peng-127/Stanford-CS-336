import torch
import torch.nn as nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    
    def __init__(
        self,
        vocab_size: int, # Size of the Vocab
        d_model: int, # Dim of the Embedding Vectors, i.e., d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        # weight (vocab_size, d_model)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    vocab_size, d_model,
                    device = device, dtype = dtype
                ),
                # mean, std, lower bound, upper bound
                mean = 0, std = 1, a = -3, b = 3
            )
        )
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # token_ids has shape (batch_size, sequence_length) and integer dtype, usually torch.long.
        # Each token ID selects one row from self.weight of shape (vocab_size, d_model).
        # The result has shape (batch_size, sequence_length, d_model).
        out = self.weight[token_ids]
        return out