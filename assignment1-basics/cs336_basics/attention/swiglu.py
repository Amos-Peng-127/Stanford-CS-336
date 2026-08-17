import torch
import torch.nn as nn
from einops import rearrange, einsum

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_ff = d_ff
        std = (2 / (d_model + d_ff)) ** 0.5
        self.W1 = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_ff, d_model), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )
        self.W2 = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_model, d_ff), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )
        self.W3 = nn.init.trunc_normal_(
            nn.Parameter(
                torch.empty((d_ff, d_model), dtype = dtype, device = device)
            ),
            mean = 0,
            std = std,
            a = -3 * std, 
            b = 3 * std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        w1_x = einsum(
            x, self.W1,
            "... d_model, ... d_ff d_model -> ... d_ff"
        )

        w3_x = einsum(
            x, self.W3,
            "... d_model, ... d_ff d_model -> ... d_ff"
        )

        silu = torch.mul(w1_x, torch.sigmoid(w1_x))

        swiglu = einsum(
            self.W2, torch.mul(silu, w3_x),
            "... d_model d_ff, ... d_ff -> ... d_model"
        )

        return swiglu