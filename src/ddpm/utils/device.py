"""Device auto-detection."""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    prefer_cuda : bool, optional
        If True (default), use CUDA when available. If False, force CPU
        regardless of CUDA availability.

    Returns
    -------
    torch.device
        ``cuda`` if available and ``prefer_cuda``, else ``cpu``.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
