import torch.nn as nn
import torch as torch
import math as math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, device=None, dtype=None):
        # 2. Crucial step: Initialize the parent class
        super().__init__()

        # Construct a linear
        # transformation module. This function should accept the following parameters:
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters

        self.in_features = in_features
        self.out_features = out_features

        if weight is None:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
            std = math.sqrt(2.0 / (in_features + out_features))
            nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)
        else:
            self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input.
        return x @ self.weight.T
