from collections.abc import Callable
from typing import Optional
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr = 1e-3,
        betas = (0.9, 0.999), eps = 1e-8,
        weight_decay: float = 0.01
    ):
        beta1, beta2 = betas
        if lr < 0 or beta1 < 0 or beta1 >= 1 or \
            beta2 < 0 or beta2 >= 1 or weight_decay < 0 or eps < 0:
            raise ValueError(f"Invalid Parameter")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Get State Associated with p.
                state = self.state[p]
                
                # Initialize State
                if len(state) == 0:
                    state["t"] = 0
                    state["m_t"] = torch.zeros_like(p.data)
                    state["v_t"] = torch.zeros_like(p.data)

                # Get Iteration Number, m_t, v_t From the State
                t = state.get("t")
                m_t = state.get("m_t")
                v_t = state.get("v_t")

                # Get the Gradient of Loss with Respect to p
                grad = p.grad.data
                
                # Update m_t, v_t
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * torch.mul(grad, grad)

                # Bias Correction
                m_t_hat = m_t / (1 - beta1 ** (t + 1))
                v_t_hat = v_t / (1 - beta2 ** (t + 1))

                # Update Weight Tensor
                p.data = (1 - lr * weight_decay) * p.data - lr * m_t_hat / (torch.sqrt(v_t_hat) + eps)

                # Increment Iteration Number
                state["t"] = t + 1
                state["m_t"] = m_t
                state["v_t"] = v_t
        return loss