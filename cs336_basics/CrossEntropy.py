import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor

# RMSNorm(𝑎𝑖) =(𝑎𝑖 * 𝑔𝑖 )/RMS(𝑎)

def cross_entropy(
    inputs: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."],
) -> torch.Tensor:
    
    # flatten any leading batch/seq dims -> (N, vocab) and (N,)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    targets = targets.reshape(-1)

    
    max_f = inputs.max(dim=-1, keepdim=True).values
    x_shifted = inputs - max_f

    exp_x= torch.exp(x_shifted)
    logged_x = x_shifted - torch.log(exp_x.sum(dim=-1, keepdim=True))

    loss = - torch.gather( logged_x, -1, targets.unsqueeze(-1)).squeeze(-1)

    return loss.mean()