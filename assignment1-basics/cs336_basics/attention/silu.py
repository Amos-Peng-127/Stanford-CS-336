import torch
import torch.nn as nn
from einops import rearrange, einsum

class SiLU(nn.Module):
    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        silu = torch.mul(x, torch.sigmoid(x))

        return silu