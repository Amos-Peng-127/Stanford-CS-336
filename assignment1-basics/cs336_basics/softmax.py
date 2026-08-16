import torch

def softmax(
    x: torch.Tensor,
    dim: int
):
    max_x, max_x_indice = torch.max(x, dim = dim, keepdim = True)

    exp_x = torch.exp(x - max_x)
    
    out = exp_x / torch.sum(exp_x, dim = dim, keepdim = True)

    return out 