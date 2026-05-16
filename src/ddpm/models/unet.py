"""U-Net architecture for noise prediction in DDPMs.

A symmetric encoder-decoder with skip connections, conditioned on the
diffusion timestep through sinusoidal embeddings injected at every
residual block. Designed to be light enough for FashionMNIST 28x28 and
CIFAR-10 32x32, while following the structural conventions of the
reference DDPM implementation (Ho et al., 2020).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ddpm.models.blocks import (
    AttentionBlock,
    Downsample,
    ResidualBlock,
    SinusoidalTimestepEmbedding,
    Upsample,
)


class UNet(nn.Module):
    """U-Net that predicts the noise epsilon added to a noisy image.

    The network takes a noisy image ``x_t`` and an integer timestep ``t``,
    and outputs a tensor of the same shape as ``x_t`` interpreted as the
    predicted noise. The architecture follows a classical U shape with two
    downsamples (resolution divided by 4 at the bottleneck), self-attention
    at the bottleneck, and matching upsamples with skip connections.

    Parameters
    ----------
    in_channels : int, optional
        Number of input image channels (1 for FashionMNIST, 3 for CIFAR-10).
        Defaults to 1.
    base_channels : int, optional
        Number of channels in the first conv. Channels are doubled at each
        downsample. Defaults to 64.
    channel_mults : tuple of int, optional
        Multipliers applied to ``base_channels`` at each resolution level.
        Length determines the depth of the U-Net. Defaults to ``(1, 2, 4)``.
    num_res_blocks : int, optional
        Number of residual blocks per resolution level. Defaults to 2.
    time_emb_dim : int, optional
        Dimension of the timestep embedding after the MLP. Defaults to 256.
    num_groups : int, optional
        Number of groups for GroupNorm in all blocks. Defaults to 8.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        # 1. Timestep embedding: sinusoidal -> MLP for non-linearity.
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 2. Initial projection from image channels to base_channels.
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 3. Encoder: list of (ResBlocks + optional Downsample) per level.
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(current_channels, out_channels, time_emb_dim, num_groups)
                )
                current_channels = out_channels
                channels.append(current_channels)
            # Downsample after each level except the last.
            if level < len(channel_mults) - 1:
                self.down_blocks.append(Downsample(current_channels))
                channels.append(current_channels)

        # 4. Bottleneck: ResBlock + Attention + ResBlock.
        self.mid_block1 = ResidualBlock(
            current_channels, current_channels, time_emb_dim, num_groups
        )
        self.mid_attn = AttentionBlock(current_channels, num_groups)
        self.mid_block2 = ResidualBlock(
            current_channels, current_channels, time_emb_dim, num_groups
        )

        # 5. Decoder: mirrors the encoder, with skip connections.
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            # +1 to account for the channels added by the encoder at this level.
            for _ in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                self.up_blocks.append(
                    ResidualBlock(
                        current_channels + skip_channels,
                        out_channels,
                        time_emb_dim,
                        num_groups,
                    )
                )
                current_channels = out_channels
            # Upsample after each level except the last.
            if level < len(channel_mults) - 1:
                self.up_blocks.append(Upsample(current_channels))

        # 6. Final projection back to in_channels.
        self.final_norm = nn.GroupNorm(num_groups, current_channels)
        self.final_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the noise added to ``x`` at timestep ``t``.

        Parameters
        ----------
        x : Tensor
            Noisy image batch, shape ``(B, in_channels, H, W)``.
        t : Tensor
            Timesteps, shape ``(B,)``, dtype long.

        Returns
        -------
        Tensor
            Predicted noise, same shape as ``x``.
        """
        # Embed timestep.
        t_emb = self.time_mlp(t)

        # Encoder, collecting skip features for the decoder.
        x = self.init_conv(x)
        skips = [x]
        for block in self.down_blocks:
            x = block(x, t_emb) if isinstance(block, ResidualBlock) else block(x)
            skips.append(x)

        # Bottleneck.
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder: concatenate skip features before each ResBlock.
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
            else:
                x = block(x)

        # Final projection.
        x = self.final_conv(F.silu(self.final_norm(x)))
        return x
