"""Gaussian diffusion process: forward and reverse.

This module orchestrates the diffusion process around a noise schedule.
It precomputes all the coefficients derived from betas (alpha_t, alpha_bar_t,
and their square roots) and exposes three core operations:

- ``q_sample``: forward process (x_0 -> x_t in a single step using alpha_bar).
- ``p_sample``: reverse process (one DDPM step, stochastic).
- ``ddim_sample``: full deterministic sampling loop with optional step skipping.

The model predicting epsilon is passed as an argument to the sampling
methods, keeping the diffusion process decoupled from the architecture.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ddpm.diffusion.schedules import ScheduleName, get_beta_schedule

ModelPrediction = Literal["epsilon"]  # extensible later to "x0" or "v"


class GaussianDiffusion(nn.Module):
    """Container for the forward and reverse Gaussian diffusion processes.

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion steps T.
    schedule : {"linear", "cosine"}
        Beta schedule to use. Forwarded to ``get_beta_schedule``.
    prediction_type : {"epsilon"}
        What the model is trained to predict. Currently only ``"epsilon"``
        (the noise) is supported, which is the most common parametrization.
    schedule_kwargs : dict, optional
        Extra keyword arguments forwarded to the schedule constructor
        (e.g. ``{"beta_start": 1e-4, "beta_end": 0.02}`` for linear).
    """

    def __init__(
        self,
        num_timesteps: int,
        schedule: ScheduleName = "cosine",
        prediction_type: ModelPrediction = "epsilon",
        schedule_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        if prediction_type != "epsilon":
            raise NotImplementedError(
                f"Only epsilon prediction is supported, got '{prediction_type}'"
            )

        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # 1. Compute the schedule in float64 for numerical stability.
        betas = get_beta_schedule(schedule, num_timesteps, **(schedule_kwargs or {}))

        # 2. Derive all alpha-related quantities. We keep float64 for the
        # cumulative product, then cast everything to float32 once buffers
        # are registered (precision matters for cumprod over 1000 steps).
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # alphas_cumprod_prev[t] = alpha_bar at step t-1, with a leading 1.0
        # so that index 0 corresponds to "before any noise was added".
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        # 3. Register everything as buffers (cast to float32). Buffers move
        # with .to(device) and are saved in checkpoints, but are not trained.
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float32))
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.to(torch.float32))

        # 4. Precompute coefficients used in q_sample (forward) and later in
        # sampling. Naming follows the original Ho et al. (2020) reference
        # implementation for clarity.
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().to(torch.float32))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - alphas_cumprod).sqrt().to(torch.float32),
        )

        # Useful for the reverse process.
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            (1.0 / alphas_cumprod).sqrt().to(torch.float32),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            (1.0 / alphas_cumprod - 1.0).sqrt().to(torch.float32),
        )

        # Quantities used by p_sample (DDPM reverse step). The posterior
        # q(x_{t-1} | x_t, x_0) is a Gaussian whose mean and variance have
        # closed-form expressions (see Ho et al. 2020, eq. 6-7).
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # Clip the first element (which would be 0) to avoid log(0) downstream.
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        # Coefficients used to compute the posterior mean from x_0 and x_t.
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

        self.register_buffer("posterior_variance", posterior_variance.to(torch.float32))
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_log_variance_clipped.to(torch.float32),
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1.to(torch.float32))
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2.to(torch.float32))

    # ----------------------------------------------------------------------
    # Forward process
    # ----------------------------------------------------------------------

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample x_t from the forward process given x_0 and t.

        Applies the closed-form ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise``,
        which lets us jump directly from x_0 to any x_t without iterating.

        Parameters
        ----------
        x_start : Tensor
            Clean images, shape ``(B, C, H, W)``, values in ``[-1, 1]``.
        t : Tensor
            Timesteps, shape ``(B,)``, dtype ``long``, values in ``[0, T-1]``.
        noise : Tensor, optional
            Gaussian noise to use. If ``None``, sampled fresh. Useful for
            reproducibility in tests.

        Returns
        -------
        Tensor
            Noisy images x_t, same shape as ``x_start``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m_ab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_ab * x_start + sqrt_1m_ab * noise

    # ----------------------------------------------------------------------
    # Reverse process (DDPM and DDIM samplers)
    # ----------------------------------------------------------------------

    def _predict_x_start_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Recover the estimated x_0 from x_t and the predicted noise eps.

        Inverts the closed-form forward equation
        ``x_t = sqrt(ᾱ) x_0 + sqrt(1-ᾱ) eps`` to solve for x_0.
        """
        sqrt_recip = _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
    ) -> Tensor:
        """Sample x_{t-1} from x_t using the DDPM reverse step.

        Parameters
        ----------
        model : nn.Module
            The noise-prediction network. Must take ``(x_t, t)`` and return
            an epsilon prediction of the same shape as ``x_t``.
        x_t : Tensor
            Current noisy sample, shape ``(B, C, H, W)``.
        t : Tensor
            Current timesteps, shape ``(B,)``, dtype long.
        clip_denoised : bool, optional
            If True, clamp the predicted x_0 to ``[-1, 1]`` before computing
            the posterior mean. Stabilizes sampling and matches the reference
            DDPM implementation. Defaults to True.

        Returns
        -------
        Tensor
            ``x_{t-1}``, same shape as ``x_t``.
        """
        # 1. Predict noise and recover x_0 estimate.
        eps = model(x_t, t)
        x_start = self._predict_x_start_from_eps(x_t, t, eps)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)

        # 2. Posterior mean = coef1 * x_0 + coef2 * x_t.
        coef1 = _extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_start + coef2 * x_t

        # 3. Add noise scaled by sqrt(variance), but only when t > 0.
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Full DDPM sampling: start from pure noise, denoise for T steps.

        Parameters
        ----------
        model : nn.Module
            The noise-prediction network.
        shape : tuple of int
            Shape of the batch to generate, e.g. ``(B, C, H, W)``.
        device : torch.device or str
            Device on which to allocate the samples.

        Returns
        -------
        Tensor
            Generated samples in ``[-1, 1]``, shape ``shape``.
        """
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        device: torch.device | str = "cpu",
        clip_denoised: bool = True,
    ) -> Tensor:
        """DDIM sampler with optional step skipping.

        Parameters
        ----------
        model : nn.Module
            The noise-prediction network.
        shape : tuple of int
            Shape of the batch to generate.
        num_inference_steps : int, optional
            Number of denoising steps. Less than ``num_timesteps`` for
            accelerated sampling. Defaults to 50.
        eta : float, optional
            Controls stochasticity. ``eta=0`` is deterministic DDIM
            (default and canonical), ``eta=1`` recovers DDPM-equivalent
            variance.

            Note: with deterministic DDIM (eta=0), small models can
            exhibit drift toward the [-1, 1] boundary on long trajectories
            (e.g. 250+ steps), since prediction errors accumulate without
            stochastic decorrelation. If samples appear over-saturated,
            try eta=0.5 to 1.0, or use fewer steps (50 is usually optimal).
        device : torch.device or str
            Device on which to allocate the samples.
        clip_denoised : bool, optional
            Clamp predicted x_0 to ``[-1, 1]``.

        Returns
        -------
        Tensor
            Generated samples in ``[-1, 1]``.
        """
        if num_inference_steps > self.num_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) cannot exceed "
                f"num_timesteps ({self.num_timesteps})"
            )

        # Build DDIM timestep grid. Convention from diffusers / guided-diffusion:
        # evenly-spaced indices from 0 to T-1 inclusive, then reversed so we
        # iterate from t=T-1 down to t=0. The "previous" alpha_bar at the last
        # step (t=0) is the virtual alpha_bar_{-1} = 1.0.
        step_indices = torch.linspace(
            0, self.num_timesteps - 1, num_inference_steps, device=device
        ).round().long().flip(0)

        x = torch.randn(shape, device=device)
        for i in range(num_inference_steps):
            t = torch.full((shape[0],), step_indices[i].item(), device=device, dtype=torch.long)

            ab_t = _extract(self.alphas_cumprod, t, x.shape)
            if i < num_inference_steps - 1:
                t_prev = torch.full(
                    (shape[0],), step_indices[i + 1].item(), device=device, dtype=torch.long
                )
                ab_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                ab_prev = torch.ones_like(ab_t)

            # 1. Predict noise and recover x_0 estimate.
            eps = model(x, t)
            x_start = self._predict_x_start_from_eps(x, t, eps)
            if clip_denoised:
                x_start = x_start.clamp(-1.0, 1.0)

            # 2. DDIM update with optional stochastic component.
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
            dir_xt = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = ab_prev.sqrt() * x_start + dir_xt + sigma * noise

        return x


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _extract(buffer: Tensor, t: Tensor, target_shape: torch.Size) -> Tensor:
    """Gather buffer values at indices t and reshape for broadcasting.

    Given a 1-D buffer of shape ``(T,)`` and a batch of timesteps of shape
    ``(B,)``, returns a tensor of shape ``(B, 1, 1, 1)`` for 2D images of
    shape ``(B, C, H, W)``, so that each scalar can broadcast over its
    corresponding image during multiplication. More generally, the output
    matches the rank of ``target_shape`` (e.g. ``(B, 1, 1, 1, 1)`` for a
    5D input), so it can multiply tensors of any shape.
    """
    out = buffer.gather(0, t)
    return out.view(-1, *([1] * (len(target_shape) - 1)))
