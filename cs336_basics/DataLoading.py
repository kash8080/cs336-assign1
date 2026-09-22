import torch.nn as nn
import torch as torch
import math as math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np


class DataLoading():
    def __init__(self, batch_size: int, context_length: int, device: str):
        # 2. Crucial step: Initialize the parent class
        super().__init__()

        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def load(self, dataset: npt.NDArray) -> tuple[torch.Tensor, torch.Tensor]:

        n = len(dataset)

        # randomly pick sequence start positions
        # this may create duplicate or miss some seq but that's okay for long run
        # we can also pick all seqs and shuffle them
        starts = np.random.randint(0, n- self.context_length, self.batch_size)

        # shape = [batch_size,1]
        batch_starts_col = starts[:,None]

        # seq to add to start vectors, shape = [1, context length]
        incrementing_seq = np.arange(self.context_length)[None,:]

        # creates a 2d matrix [batch_size, context_length] as we are adding an array to the first and only element to the row
        indexes = batch_starts_col + incrementing_seq

        inputs_np = dataset[indexes]
        targets_np = dataset[indexes + 1]

        # convert to tensors with device mentioned
        inputs = torch.tensor(inputs_np, dtype=torch.long, device=self.device)
        targets = torch.tensor(targets_np, dtype=torch.long, device=self.device)

        return inputs, targets