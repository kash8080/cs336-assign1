import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        checkpoint_blocks: bool = False,
    ):
        # 2. Crucial step: Initialize the parent class
        super().__init__()

        # Wrap each TransformerBlock in its own activation checkpoint: forward keeps only
        # each block's input, and backward recomputes one block at a time.
        self.checkpoint_blocks = checkpoint_blocks

        # Token embedding matrix. Shape is (vocab_size, d_model).
        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )

        # Weights of affine transform for RMSNorm applied to the output of the final transformer block.
        # Shape is (d_model, ).
        self.ln_final = RMSNorm(d_model)

        # Weights of the language model output embedding.
        # Shape is (vocab_size, d_model).
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:

        output = self.token_embeddings(in_indices)

        tp = torch.arange(in_indices.shape[-1], device=in_indices.device)
        for layer in self.layers:
            # Only checkpoint when building a graph; under no_grad there is nothing to save.
            if self.checkpoint_blocks and torch.is_grad_enabled():
                output = checkpoint(layer, output, tp, use_reentrant=False)
            else:
                output = layer(output, tp)

        output = self.ln_final(output)
        return self.lm_head(output)

