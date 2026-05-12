"""Noise schedules for the forward diffusion process.

A schedule defines the sequence of betas (beta_t for t = 0, ..., T-1) that
controls how much noise is added at each step. From these betas we later
derive alpha_t = 1 - beta_t and alpha_bar_t = prod(alpha_s for s <= t),
which are the quantities actually used in the forward and reverse processes.

Two schedules are provided:
- linear: beta_t grows linearly between two endpoints (Ho et al., 2020).
- cosine: alpha_bar_t follows a cosine curve, which destroys information
  more uniformly across timesteps (Nichol & Dhariwal, 2021). Recommended
  for low-resolution images.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

ScheduleName = Literal["linear", "cosine"]


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tensor:
    """Linear schedule from the original DDPM paper.

    Betas grow linearly from ``beta_start`` to ``beta_end``. Default values
    match Ho et al. (2020) for T=1000 on 32x32 images.

    Returns a 1-D float64 tensor of shape ``(num_timesteps,)``.
    """
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(
            f"Expected 0 < beta_start < beta_end < 1, got ({beta_start}, {beta_end})"
        )

    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> Tensor:
    """Cosine schedule from the Improved DDPM paper.

    Instead of defining beta_t directly, this schedule defines alpha_bar_t
    as a cosine curve and recovers beta_t from consecutive ratios:
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped to ``max_beta`` to
    avoid singularities near t = T.

    Returns a 1-D float64 tensor of shape ``(num_timesteps,)``.
    """
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
    if s <= 0.0 or not 0.0 < max_beta < 1.0:
        raise ValueError(f"Invalid hyperparameters: s={s}, max_beta={max_beta}")

    # We need T+1 values of alpha_bar to compute T betas as consecutive ratios.
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_timesteps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    alpha_bar = f / f[0]  # normalize so that alpha_bar[0] == 1

    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, max=max_beta)


def get_beta_schedule(name: ScheduleName, num_timesteps: int, **kwargs) -> Tensor:
    """Factory dispatching to the schedule selected by ``name``.

    This indirection lets configs reference schedules by string
    (e.g. ``schedule: cosine`` in YAML) without scattering string checks
    across the codebase.
    """
    if name == "linear":
        return linear_beta_schedule(num_timesteps, **kwargs)
    if name == "cosine":
        return cosine_beta_schedule(num_timesteps, **kwargs)
    raise ValueError(f"Unknown schedule '{name}'. Expected 'linear' or 'cosine'.")