"""Checkpoint save/load utilities.

Centralizes the structure of a training checkpoint: model, EMA, optimizer,
scheduler (if any), step count, and config. Keeping this in one place
makes resume logic trivial and ensures we always save what is needed to
reproduce a state exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    ema: Any = None,
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path : str or Path
        Where to save. Parent directory is created if missing.
    model : nn.Module
        The model being trained (live weights, not EMA).
    optimizer : optim.Optimizer
        The optimizer (saves its internal state, e.g. momentum buffers).
    step : int
        Current training step.
    ema : EMA, optional
        EMA wrapper. If provided, its state is included.
    extra : dict, optional
        Any additional metadata to store (e.g. config, metrics).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    ema: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint and restore the given objects in place.

    Parameters
    ----------
    path : str or Path
        Checkpoint file path.
    model : nn.Module
        Model to load weights into.
    optimizer : optim.Optimizer, optional
        Optimizer to restore. Skip if you only want to sample/evaluate.
    ema : EMA, optional
        EMA wrapper to restore. Required if the checkpoint contains EMA
        state and you want to sample from it.
    map_location : str or torch.device, optional
        Where to load the tensors. Defaults to ``"cpu"`` so the checkpoint
        can be loaded on machines without the original training GPU.

    Returns
    -------
    dict
        The loaded checkpoint dict (so callers can access ``step``,
        ``extra``, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    return checkpoint
