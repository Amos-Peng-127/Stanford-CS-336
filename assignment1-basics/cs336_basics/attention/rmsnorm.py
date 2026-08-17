import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Construct the RMSNorm module.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.init.ones_(
            nn.Parameter(
                torch.empty(
                    d_model, device = device, dtype = dtype
                )
            )
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., d_model)
        and return a tensor of the same shape.
        """
        # Upcast Input to Prevent Overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Calculation
        rms_a = torch.sqrt(
            torch.mean(torch.square(x), dim = -1, keepdim = True) + self.eps
        )

        rms_norm = torch.mul(x  / rms_a, self.weight)

        # Downcast to Original Type
        return rms_norm.to(in_dtype)