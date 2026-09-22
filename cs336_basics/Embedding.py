import torch.nn as nn
import torch as torch
import math as math


# HOW `self.weight[token_ids]` WORKS
# ---------------------------------
# This is not matrix multiplication. It is PyTorch's integer-array indexing.
# Both values are tensors, but `token_ids` is used as a collection of indices
# into the first dimension (the rows) of `weight`.
#
# Suppose the embedding matrix has shape `(4, 3)`: four tokens, each with a
# three-dimensional embedding. Its contents might be:
#
#     weight = tensor([
#         [10, 11, 12],  # embedding for token 0
#         [20, 21, 22],  # embedding for token 1
#         [30, 31, 32],  # embedding for token 2
#         [40, 41, 42],  # embedding for token 3
#     ])
#
# Now suppose the input contains these token IDs:
#
#     token_ids = tensor([
#         [2, 0],
#         [1, 3],
#     ])
#
# Although `token_ids` is a 2D tensor, each individual number in it is treated
# as a row number in `weight`. Therefore `weight[token_ids]` produces:
#
#     tensor([
#         [
#             [30, 31, 32],  # weight[2]
#             [10, 11, 12],  # weight[0]
#         ],
#         [
#             [20, 21, 22],  # weight[1]
#             [40, 41, 42],  # weight[3]
#         ],
#     ])
#
# Conceptually, PyTorch performs the following lookup for every position:
#
#     output[0, 0] = weight[token_ids[0, 0]]  # weight[2]
#     output[0, 1] = weight[token_ids[0, 1]]  # weight[0]
#     output[1, 0] = weight[token_ids[1, 0]]  # weight[1]
#     output[1, 1] = weight[token_ids[1, 1]]  # weight[3]
#
# In general, the shapes are:
#
#     token_ids.shape          = (batch_size, sequence_length)
#     weight.shape             = (vocab_size, embedding_dim)
#     weight[token_ids].shape  = (batch_size, sequence_length, embedding_dim)
#
# More generally, the output shape is:
#
#     token_ids.shape + (embedding_dim,)
#
# PyTorch replaces each scalar token ID with the entire corresponding row of
# `weight`. Because only one index is supplied in `weight[token_ids]`, only the
# first (vocabulary) dimension is indexed; the full final embedding dimension
# is preserved. If two comma-separated indices were supplied instead, as in
# `weight[row_ids, column_ids]`, PyTorch would index both matrix dimensions.
#
# `token_ids` must have an integer dtype (normally `torch.long`), and every ID
# must be in the range `0` through `num_embeddings - 1`.
class Embedding(nn.Module) :
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) :
        super().__init__()
        # Construct an
        # embedding module. This function should accept the following parameters:
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=1,
            a = -3,
            b=3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        # Lookup the embedding vectors
        # for the given token IDs.
        return self.weight[token_ids]
