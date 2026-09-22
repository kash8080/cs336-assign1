from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "dr": weight_decay,
            "b1": betas[0],
            "b2": betas[1],
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            dr = group["dr"]  # Get the decay rate.
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]  # Get the epsilon for stability
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)

                m = state.get("m")  # first moment vector. same shape as p
                v = state.get("v")  # second moment vector. same shape as p
                t = state.get("t", 1)  # counter

                grad = p.grad.detach()  # Get the gradient of loss with respect to p.

                with torch.no_grad():

                    lra = lr * ((math.sqrt(1 - math.pow(b2, t))) / (1 - math.pow(b1, t)))
                    p.add_( - lr * dr * p)

                    # update first moment estimate
                    m = b1 * m + (1 - b1) * grad

                    # update second moment estimate
                    v = b2 * v + (1 - b2) * torch.pow(grad, 2)

                    p.add_( - lra * (m / (torch.sqrt(v) + eps)))

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
