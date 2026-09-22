import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        rms1_weights: torch.Tensor = None,
        rms2_weights: torch.Tensor = None,
        q_proj_weight: Float[Tensor, " d_model d_model"] = None,
        k_proj_weight: Float[Tensor, " d_model d_model"] = None,
        v_proj_weight: Float[Tensor, " d_model d_model"] = None,
        o_proj_weight: Float[Tensor, " d_model d_model"] = None,
        w1_weight: torch.Tensor = None,
        w2_weight: torch.Tensor = None,
        w3_weight: torch.Tensor = None,
    ):
        # 2. Crucial step: Initialize the parent class
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.ln1 = RMSNorm(d_model, weights=rms1_weights)
        self.ln2 = RMSNorm(d_model, weights=rms2_weights)

        self.ffn = SwiGLU(
            d_model, d_ff, w1_weight=w1_weight, w2_weight=w2_weight, w3_weight=w3_weight
        )
        

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        # for attention
        x = self.ln1(in_features)
        x = self.attn(
            x,  ## rms value
            token_positions if token_positions is not None else torch.arange(in_features.shape[-2], device=in_features.device),
        )
        x = x + in_features

        # for FFN i.e swiglu
        y = self.ln2(x)
        y = self.ffn(y)
        y = y + x

        return y
