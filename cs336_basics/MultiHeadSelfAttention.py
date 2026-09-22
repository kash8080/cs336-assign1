import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.ScaledDotProductAttention import scaled_dot_product_attention
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.Linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_model"] = None ,
        k_proj_weight: Float[Tensor, " d_model d_model"] = None,
        v_proj_weight: Float[Tensor, " d_model d_model"] = None,
        o_proj_weight: Float[Tensor, " d_model d_model"] = None,
        theta: float = None,
        max_seq_len: int = None,
    ):
        # 2. Crucial step: Initialize the parent class
        super().__init__()

        self.head_dim = int(d_model // num_heads)
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, weight=q_proj_weight)
        self.k_proj = Linear(d_model, d_model, weight=k_proj_weight)
        self.v_proj = Linear(d_model, d_model, weight=v_proj_weight)
        self.output_proj = Linear(d_model, d_model, weight=o_proj_weight)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(self.head_dim, theta, max_seq_len)


    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:

        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)

        # seq_len d_model -> num_heads seq_len head_dim
        Qh = Q.reshape(*Q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        Kh = K.reshape(*K.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        Vh = V.reshape(*V.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)

        if token_positions is not None and self.rope is not None:
            Qh = self.rope(Qh, token_positions)
            Kh = self.rope(Kh, token_positions)

        tokens_len = Q.shape[-2]

        mask = torch.tril(
            torch.full((tokens_len, tokens_len), True, device=in_features.device)
        )
       
        At = scaled_dot_product_attention(Qh, Kh, Vh, mask)

        A = At.transpose(-3, -2)
        A_merged = A.reshape(*A.shape[:-2], self.d_model)

        # (seq × d_model) @ (d_model × d_model)
        return self.output_proj(A_merged)
