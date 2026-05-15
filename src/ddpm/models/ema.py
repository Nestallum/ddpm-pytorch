"""Exponential Moving Average of model parameters.

Maintains a separate copy of the model weights, updated at every training
step as a moving average. Sampling from the EMA copy produces noticeably
better images than sampling from the live training weights, since EMA
smooths out the noise introduced by SGD.

Standard practice in diffusion: see Ho et al. (2020), Song et al. (2020),
and virtually every reference implementation.
"""

from __future__ import annotations

from copy import deepcopy

from torch import nn


class EMA:
    """Exponential moving average wrapper for ``nn.Module`` parameters.

    Stores a shadow copy of the model and updates it at each step with
    ``theta_ema = beta * theta_ema + (1 - beta) * theta``.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters should be tracked.
    beta : float, optional
        Decay rate. Higher means more inertia (slower to adapt). Typical
        values in diffusion are 0.999 to 0.9999. Defaults to 0.9999.
    update_after_step : int, optional
        Number of steps to wait before starting to update the EMA. During
        this warmup, the EMA copy mirrors the live model exactly. Avoids
        polluting the EMA with the initial random weights. Defaults to 0.
    update_every : int, optional
        Update frequency. ``1`` means every step. Higher values speed up
        training slightly at no quality cost. Defaults to 1.
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must lie in (0, 1), got {beta}")
        if update_after_step < 0 or update_every < 1:
            raise ValueError(
                f"Invalid hyperparameters: update_after_step={update_after_step}, "
                f"update_every={update_every}"
            )

        self.beta = beta
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0

        # Deep copy the model: same architecture, separate weights, eval mode.
        # We freeze the EMA model so its parameters are never touched by
        # autograd or the optimizer.
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update the EMA weights from the current model.

        Should be called once per training step, after the optimizer step.
        """
        self.step += 1

        # Warmup: mirror the live model exactly until update_after_step.
        if self.step <= self.update_after_step:
            self._copy_weights(model)
            return

        # Skip if not on an update boundary.
        if self.step % self.update_every != 0:
            return

        # In-place EMA update over parameters AND buffers.
        # Buffers (e.g. GroupNorm running stats if any) also need to be tracked.
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters(), strict=True):
            p_ema.data.mul_(self.beta).add_(p.data, alpha=1.0 - self.beta)
        for b_ema, b in zip(self.ema_model.buffers(), model.buffers(), strict=True):
            b_ema.data.copy_(b.data)

    def _copy_weights(self, model: nn.Module) -> None:
        """Reset the EMA to mirror the live model exactly."""
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters(), strict=True):
            p_ema.data.copy_(p.data)
        for b_ema, b in zip(self.ema_model.buffers(), model.buffers(), strict=True):
            b_ema.data.copy_(b.data)

    def state_dict(self) -> dict:
        """Return the state dict (for checkpointing)."""
        return {
            "ema_model": self.ema_model.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load from a state dict (resume from checkpoint)."""
        self.ema_model.load_state_dict(state_dict["ema_model"])
        self.step = state_dict["step"]
