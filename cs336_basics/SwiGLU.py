import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.Linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: torch.Tensor = None,
        w2_weight: torch.Tensor = None,
        w3_weight: torch.Tensor = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, weight=w1_weight, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, weight=w2_weight, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, weight=w3_weight, device=device, dtype=dtype)

    # SwiGLU(𝑥,𝑊1,𝑊2,𝑊3) = 𝑊2(SiLU(𝑊1𝑥)⊙𝑊3𝑥)
    def forward(
        self, in_features: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:

        a = self.silu(self.w1(in_features))

        b = self.w3(in_features)

        c = self.w2(a * b)

        return c

    # SiLU(x) = x * sigmoid(x)
    # = x * 1/ (1+e^-x)
    def silu(self, x: Tensor) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return x * sigmoid_x
