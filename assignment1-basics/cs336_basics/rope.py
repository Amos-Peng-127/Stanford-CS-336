import torch
import torch.nn as nn
from einops import einsum, rearrange

class RoPE(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        theta_ik = einsum(
            torch.arange(max_seq_len, device = device, dtype = torch.float32),
            1 / theta ** ((2 * torch.arange(1, d_k // 2 + 1, device = device, dtype = torch.float32) - 2) / d_k),
            "position, pair -> position pair"

        )
        
        # Shape (row, column, position, pair)
        R_ik = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(theta_ik),
                        -torch.sin(theta_ik)
                    ],
                    dim = 0
                ),
                torch.stack(
                    [
                        torch.sin(theta_ik),
                        torch.cos(theta_ik)
                    ],
                    dim = 0
                )
            ],
            dim = 0
        )
        
        # Rearrange R_ik Shape (position, pair, row, column)
        R_ik = rearrange(
            R_ik,
            "row col pos pair -> pos pair row col"
        )

        assert R_ik.shape == (max_seq_len, d_k // 2, 2, 2)

        self.register_buffer(
            "R_ik", R_ik, persistent = False
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Rotation Matrix for Token Positions
        # Shape (seq_len, pair, row, col) Not max_seq_len
        r_ik = self.R_ik[token_positions]

        # Transform x to x Pair: [x1, x2,...] -> [[x1, x2], ...]
        x_rearrange = rearrange(
            x,
            "... (pair col) -> ... pair col",
            col = 2
        )
        
        # Calculate the q'(i)
        out = einsum(
            x_rearrange, r_ik,
            "... col, ... row col -> ... row" 
        )

        out = rearrange(
            out,
            "... pair row -> ... (pair row)"
        )
        return out