"""Entry point for sampling from a trained DDPM.

Usage:
    uv run python scripts/sample.py \\
        --checkpoint outputs/fashion_mnist_baseline/checkpoints/step_0030000.pt \\
        --config outputs/fashion_mnist_baseline/config.yaml \\
        --num-samples 64 \\
        --sampler ddim \\
        --steps 50

Loads a trained model from a checkpoint, generates ``num_samples`` images
using the requested sampler, and saves them as a grid plus individual
PNGs under the run's ``samples/`` directory.

By default, samples are generated from the EMA weights (the only thing
that matters for inference quality).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from ddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from ddpm.models.ema import EMA
from ddpm.models.unet import UNet
from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.config import load_config
from ddpm.utils.device import get_device
from ddpm.utils.logger import get_logger
from ddpm.utils.seed import set_seed

log = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained DDPM.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint .pt file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML used at training time.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sampling (lower if you hit OOM).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddim",
        help="Sampling algorithm.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (DDIM only; DDPM uses num_timesteps).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity (0=deterministic, 1=DDPM-equivalent variance).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save samples. Defaults to <checkpoint_parent>/../samples_final/.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling RNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Setup.
    cfg = load_config(args.config)
    device = get_device()
    set_seed(args.seed)
    log.info(f"Device: {device}")

    # 2. Output directory.
    if args.output_dir is None:
        ckpt_path = Path(args.checkpoint)
        output_dir = ckpt_path.parent.parent / "samples_final"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # 3. Rebuild model and diffusion from config.
    model = UNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        num_groups=cfg.model.num_groups,
    ).to(device)

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule=cfg.diffusion.schedule,
        prediction_type=cfg.diffusion.prediction_type,
    ).to(device)

    ema = EMA(model, beta=cfg.ema.beta)
    ema.ema_model.to(device)

    # 4. Load checkpoint. We pass model and ema but no optimizer (inference only).
    ckpt = load_checkpoint(args.checkpoint, model=model, ema=ema, map_location=device)
    log.info(f"Loaded checkpoint from step {ckpt['step']}")

    # We sample from the EMA weights, not the live model.
    ema.ema_model.eval()

    # 5. Generate samples in batches.
    all_samples = []
    n_remaining = args.num_samples
    while n_remaining > 0:
        batch = min(args.batch_size, n_remaining)
        shape = (batch, cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)

        if args.sampler == "ddpm":
            samples = diffusion.p_sample_loop(ema.ema_model, shape=shape, device=device)
        else:  # ddim
            samples = diffusion.ddim_sample(
                ema.ema_model,
                shape=shape,
                num_inference_steps=args.steps,
                eta=args.eta,
                device=device,
            )

        all_samples.append(samples.cpu())
        n_remaining -= batch
        log.info(f"Generated {args.num_samples - n_remaining}/{args.num_samples}")

    samples = torch.cat(all_samples, dim=0)
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    # 6. Save grid + individual PNGs.
    grid_path = output_dir / f"grid_{args.sampler}_{args.steps}steps.png"
    nrow = int(args.num_samples**0.5)
    save_image(make_grid(samples, nrow=nrow), grid_path)
    log.info(f"Saved grid to {grid_path}")

    individual_dir = output_dir / f"individual_{args.sampler}_{args.steps}steps"
    individual_dir.mkdir(exist_ok=True)
    for i, img in enumerate(samples):
        save_image(img, individual_dir / f"{i:04d}.png")
    log.info(f"Saved {args.num_samples} individual images to {individual_dir}")


if __name__ == "__main__":
    main()
