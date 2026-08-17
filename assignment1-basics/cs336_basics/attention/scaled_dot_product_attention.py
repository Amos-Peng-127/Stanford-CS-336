import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.attention.softmax import softmax

def scaled_dot_product_attention(
    Q: torch.Tensor, # (n, d_k)
    K: torch.Tensor, # (m, d_k)
    V: torch.Tensor, # (m, d_v)
    mask: torch.Tensor, # (n, m)
):
    # Calculate Q @ k.T / sqrt(d_k)
    d_k = Q.shape[-1]
    scores = einsum(
        Q, K,
        "... n d_k, ... m d_k -> ... n m"
    ) / torch.sqrt(torch.tensor(d_k))

    # Adding -inf To any Entry of the Mask Matrix that is False
    scores = torch.where(mask, scores, -torch.inf)
    pre_softmax = softmax(scores, dim = -1)

    # Dot Product w/ Value
    out = pre_softmax @ V


    return out