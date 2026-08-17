import torch

def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
):  
    # Loss = 1 / (D * m) * -log(exp(z_y - m) / sum(exp(z_j - m)))
    #      = 1 / (D * m) * (-(z_y - m) + log(sum(exp(z_j - m)))

    # 1. m
    max_logits, _ = torch.max(logits, dim = -1, keepdim = True)

    # 2. z_y
    target_logits = torch.gather(logits, dim = -1, index = torch.unsqueeze(targets, dim = -1))

    # 3. exp(z_j - m)
    shifted_exp_logits = torch.exp(logits - max_logits)

    # 4. sum(exp(z_j - m))
    sum_exp = torch.sum(shifted_exp_logits, dim = -1, keepdim = True)

    # 5. log(sum(exp(z_j - m)))
    log_normalizer = torch.log(sum_exp)

    return torch.mean(-(target_logits - max_logits) + log_normalizer)
