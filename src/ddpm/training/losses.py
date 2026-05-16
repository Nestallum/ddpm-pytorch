"""Training losses for DDPM.

The DDPM loss is a simple MSE between the noise predicted by the model
and the ground-truth noise sampled during the forward process. This is
the "simplified" objective from Ho et al. (2020), eq. 14.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def ddpm_loss(eps_pred: Tensor, eps_true: Tensor) -> Tensor:
    """MSE loss between predicted and true noise.

    Parameters
    ----------
    eps_pred : Tensor
        Noise predicted by the model, shape ``(B, C, H, W)``.
    eps_true : Tensor
        Ground-truth noise sampled during the forward process, same shape.

    Returns
    -------
    Tensor
        Scalar loss (mean over all elements).
    """
    return F.mse_loss(eps_pred, eps_true)
