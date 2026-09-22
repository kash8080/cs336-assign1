import torch.nn as nn
import torch as torch
import math as math


# RMSNorm(𝑎𝑖) =(𝑎𝑖 * 𝑔𝑖 )/RMS(𝑎)
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        weights: torch.Tensor = None,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # Construct the RMSNorm module. This function should accept the following parameters:
        self.d_model = d_model
        self.eps = eps

        # for the trainable gi
        if weights is None:
            # create some dummy values
            self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        else:
            self.weight = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape
        # (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        dtype = x.dtype

        # to prevent overflow doing square calculating RMS
        x = x.to(torch.float32)

        # x is B,S,D
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # rms is B,S,1

        # returns (B, S, D)
        normalised = x / rms

        # weight is (D)
        # while multplying pytorch makes it same shape as normalised and multiply with last dimension only
        # so weight first becomes (1,1,D)
        # then multiplying (B, S, D) with (1,1,D)
        result = normalised * self.weight

        return result.to(dtype)
