"""Reproducibility helpers.

Seeds all known sources of randomness in a typical PyTorch training pipeline.
Calling :func:`set_seed` at the start of a script makes runs reproducible
up to non-deterministic CUDA kernels (which can be made deterministic at
the cost of speed; see :func:`set_seed` documentation).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Parameters
    ----------
    seed : int
        Seed value to use across all RNGs.
    deterministic : bool, optional
        If True, also force PyTorch to use deterministic CUDA algorithms.
        This slows training down (sometimes significantly) but guarantees
        bit-exact reproducibility across runs on the same hardware.
        Defaults to False.

    Notes
    -----
    The five sources of randomness seeded here:

    1. Python's built-in ``random``.
    2. NumPy.
    3. PyTorch CPU.
    4. PyTorch CUDA (all GPUs).
    5. The ``PYTHONHASHSEED`` environment variable, which controls hash
       randomization for sets and dicts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
