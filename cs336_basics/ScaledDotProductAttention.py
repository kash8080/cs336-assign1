import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.Softmax import softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    
    dim = Q.shape[-1]

    # shape - ( ... queries keys)
    numerator = Q@K.transpose(-2,-1)

    pre_softmax = numerator / math.sqrt(dim)

    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(mask= ~mask, value= -torch.inf)

    softmaxxed = softmax(pre_softmax, dim=-1)

    # (..., queries, d_v)
    return softmaxxed @ V
