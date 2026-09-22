import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> torch.Tensor:
    max_f = in_features.max(dim=dim, keepdim=True).values

    x_shifted = in_features - max_f

    exp_x = torch.exp(x_shifted)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)
