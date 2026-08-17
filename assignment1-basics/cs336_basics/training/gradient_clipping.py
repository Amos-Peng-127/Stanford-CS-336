import torch
from collections.abc import Iterable

def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):  
    if max_l2_norm <= 0:
        raise ValueError("Max_L2_Norm should be positive")

    epsilon = 1e-6
    
    # 1. Materialize the parameter iterable
    parameters = list(parameters)
    
    # 2. Initialize the accumulator on the gradient device
    found_grad = False

    for parameter in parameters:
        if parameter.grad is not None:
            sum_squared_grad_norms = torch.tensor(
                0.0,
                device = parameter.grad.device,
                dtype = parameter.grad.dtype)
            
            found_grad = True
            break
    
    if not found_grad:
        return

    # 3. Compute the global gradient L2 norm
    for parameter in parameters:
        if parameter.grad is None:
            continue
        
        # squared_grad_norm = ||parameter.grad||²
        squared_grad_norm = torch.linalg.vector_norm(parameter.grad, ord = 2) ** 2 
        
        # sum_squared_grad_norms = Σ ||parameter.grad||²
        sum_squared_grad_norms += squared_grad_norm

    # global_grad_norm = sqrt(sum_squared_grad_norms)
    global_grad_norm = torch.sqrt(sum_squared_grad_norms)

    # 4. Scale all gradients in-place with a shared clipping factor
    if global_grad_norm >= max_l2_norm:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            
            # Multiply clip_scale Factor In-place: clip_scale = max_norm / (global_grad_norm + epsilon)
            parameter.grad.mul_(max_l2_norm / (global_grad_norm + epsilon))