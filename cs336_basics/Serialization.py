"""Checkpoint (de)serialization shared by train.py and the test adapters."""
import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer, and iteration count to `out`."""
    torch.save(
        {
            "state_model": model.state_dict(),
            "state_opt": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location=None,
) -> int:
    """Restore model and optimizer from `src`; return the saved iteration count."""
    state = torch.load(src, map_location=map_location)
    model.load_state_dict(state["state_model"])
    optimizer.load_state_dict(state["state_opt"])
    return state["iteration"] or 0


def load_model(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    map_location=None,
) -> int:
    """Restore only the model weights from `src` (for inference); return the saved iteration count."""
    state = torch.load(src, map_location=map_location)
    model.load_state_dict(state["state_model"])
    return state["iteration"] or 0
