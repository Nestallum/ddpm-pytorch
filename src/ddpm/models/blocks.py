"""Reusable building blocks for the U-Net.

Defines:
- ``SinusoidalTimestepEmbedding``: encodes scalar timesteps as fixed-frequency
  sinusoidal vectors (same construction as positional encoding in Transformers).
- ``ResidualBlock``: two-convolution residual block with timestep injection.
- ``AttentionBlock``: self-attention over spatial positions (used at low
  resolutions only).
- ``Downsample`` / ``Upsample``: strided convolutions for resolution changes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Encode integer timesteps as sinusoidal feature vectors.

    Same construction as the positional encoding from Vaswani et al. (2017).
    Maps an integer timestep ``t`` (or a batch ``(B,)``) to a continuous
    embedding ``(B, dim)`` that the rest of the network can consume.

    Parameters
    ----------
    dim : int
        Dimension of the embedding. Must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        # Frequencies geometrically spaced from 1 to 1/10000.
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=1)  # (B, dim)


class ResidualBlock(nn.Module):
    """Residual block with two GroupNorm-SiLU-Conv layers and timestep injection.

    The timestep embedding is projected to the channel dimension and added
    between the two convolutions, which is how the network is conditioned
    on the noise level.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_emb_dim : int
        Dimension of the timestep embedding (after the MLP).
    num_groups : int, optional
        Number of groups for GroupNorm. Defaults to 8.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Projects the (already non-linear) time embedding to the channel dim.
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Match channels for the residual connection if needed.
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding, broadcast over spatial dims.
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions, applied at the bottleneck.

    Flattens the ``(H, W)`` grid into a sequence and runs single-head
    self-attention with a residual connection.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    num_groups : int, optional
        Number of groups for GroupNorm. Defaults to 8.
    """

    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels**-0.5

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Flatten spatial dims into a sequence of length h*w.
        q = q.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        k = k.view(b, c, h * w)  # (B, C, HW)
        v = v.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)

        attn = torch.softmax(q @ k * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Halve the spatial resolution with a strided 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Double the spatial resolution with nearest-neighbor + 3x3 convolution.

    Using NN-upsample followed by a conv (instead of a transposed conv)
    avoids the checkerboard artefacts often produced by ConvTranspose2d.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
