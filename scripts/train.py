"""Entry point for training a DDPM.

Usage:
    uv run python scripts/train.py --config configs/fashion_mnist.yaml

Loads the config (merged with base.yaml), seeds RNGs, builds all
components, and launches the training loop. Outputs (checkpoints,
samples, TensorBoard logs) are written under ``outputs/<run_name>/``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from ddpm.data.datasets import get_dataloader
from ddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from ddpm.models.ema import EMA
from ddpm.models.unet import UNet
from ddpm.training.trainer import Trainer
from ddpm.utils.config import load_config, save_config
from ddpm.utils.device import get_device
from ddpm.utils.logger import get_logger
from ddpm.utils.seed import set_seed

log = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the dataset-specific YAML config.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to the base config (merged with --config).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for the output directory. Defaults to timestamp.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for run outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load and merge configs.
    cfg = load_config(args.config, base_path=args.base_config)
    log.info(f"Loaded config from {args.config} (base: {args.base_config})")

    # 2. Set up output directory.
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{cfg.data.name}_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    log.info(f"Run directory: {output_dir}")

    # 3. Reproducibility.
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    log.info(f"Seed set to {cfg.seed} (deterministic={cfg.deterministic})")

    # 4. Device.
    device = get_device()
    log.info(f"Using device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 5. Data.
    dataloader = get_dataloader(
        name=cfg.data.name,
        batch_size=cfg.data.batch_size,
        root=cfg.data.root,
        train=True,
        num_workers=cfg.data.num_workers,
    )
    log.info(
        f"Dataset: {cfg.data.name} ({len(dataloader.dataset)} samples, "
        f"batch_size={cfg.data.batch_size})"
    )

    # 6. Model.
    model = UNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        num_groups=cfg.model.num_groups,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: UNet with {n_params / 1e6:.2f}M parameters")

    # 7. Diffusion process.
    diffusion = GaussianDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule=cfg.diffusion.schedule,
        prediction_type=cfg.diffusion.prediction_type,
    ).to(device)
    log.info(f"Diffusion: T={cfg.diffusion.num_timesteps}, schedule={cfg.diffusion.schedule}")

    # 8. EMA.
    ema = EMA(
        model,
        beta=cfg.ema.beta,
        update_after_step=cfg.ema.update_after_step,
        update_every=cfg.ema.update_every,
    )
    ema.ema_model.to(device)
    log.info(f"EMA: beta={cfg.ema.beta}, warmup={cfg.ema.update_after_step} steps")

    # 9. Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    log.info(f"Optimizer: AdamW (lr={cfg.optim.lr})")

    # 10. Trainer + launch.
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        optimizer=optimizer,
        ema=ema,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
