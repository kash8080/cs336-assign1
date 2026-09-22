import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.Linear import Linear

class RotaryPositionalEmbedding(nn.Module):

    # theta here is basically the base. eg : 10000 
    def __init__(self, d_k: int, theta: float,  max_seq_len: int, device=None) :
        super().__init__()
        # Construct the
        # RoPE module and create buffers if needed.
        # theta: float Θ value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be input
        # device: torch.device | None = None Device to store the buffer on

        
        # first let's create index of all pairs
        pairs_idx = torch.arange(0, d_k, 2, device = device)

        # formula for frequencoes = 1 / pow(base, (2*i) / d
        # shape = (d_k/2,)
        freqs = theta ** (- pairs_idx/ d_k)

        # shape - (max_seq_len,)
        positions = torch.arange( max_seq_len, device=device)

        # do a multiplication of position and freqs. but the shapes don't match
        # so unsqueeze to add additional empty dimention in both to make positions vertical and freqs horizontal. 
        # so that we get a 2d matrix as a result. 
        # angle = m * freq 
        # fina shape = (max_seq_len, d_k / 2)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)


        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor :
        # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        # that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
        # that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        # 𝑥 along the sequence dimension.
        # print("x", x.shape, "pos", token_positions.shape, "cos_buf", self.cos.shape)

        #build odd even pairs
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # shape = (max_seq_len, d_k / 2)
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        rotated_even = x_even*cos - x_odd*sin
        rotated_odd = x_even*sin + x_odd*cos

        # flatten 
        # [[10, 11],
        # [20, 21],
        # [30, 31]]
        # to 
        # [10, 11, 20, 21, 30, 31]
        return torch.stack([rotated_even,rotated_odd], dim=-1).flatten(-2)

        
