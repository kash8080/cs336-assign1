import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Iterable


def gradient_clipping(
    max_l2_norm: float,
    parameters: Iterable[torch.nn.Parameter],
    eps: float = 1e-6,
) -> float:
    """Clip gradients in place to `max_l2_norm`. Returns the pre-clip total L2 norm."""

    # Save the gradients because parameters may be a one-use generator.
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]

    if not grads:
        return 0.0

    g2 = torch.sqrt(sum(grad.detach().pow(2).sum() for grad in grads))

    if g2 > max_l2_norm:
        scale = torch.clamp(
            max_l2_norm / (g2 + eps),
            max=1.0,
        )

        with torch.no_grad():

            for grad in grads:
                grad.mul_(scale)

    return g2.item()
