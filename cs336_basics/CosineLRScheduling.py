import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor


def cosine_lr_scheduling(
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    it: int,
) -> torch.Tensor:

    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                ((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi
            )
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate
