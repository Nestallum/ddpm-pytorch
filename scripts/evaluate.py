"""Entry point for evaluating a trained DDPM with FID and Inception Score.

Usage:
    uv run python scripts/evaluate.py \\
        --checkpoint outputs/fashion_mnist_baseline/checkpoints/step_0030000.pt \\
        --config outputs/fashion_mnist_baseline/config.yaml \\
        --num-samples 50000 \\
        --samplers ddim50 ddim250 ddpm1000

Generates ``num_samples`` images with each requested sampler, then
computes FID against the test set and Inception Score on the generated
images. Results are saved as JSON under the run's directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from ddpm.data.datasets import get_dataset
from ddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from ddpm.evaluation.fid import compute_fid
from ddpm.evaluation.inception_score import compute_inception_score
from ddpm.models.ema import EMA
from ddpm.models.unet import UNet
from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.config import load_config
from ddpm.utils.device import get_device
from ddpm.utils.logger import get_logger
from ddpm.utils.seed import set_seed

log = get_logger()


# Mapping from sampler key (CLI-friendly) to (algorithm, steps).
SAMPLER_CONFIGS = {
    "ddim50": ("ddim", 50, 0.0),
    "ddim250": ("ddim", 250, 1.0),  # eta=1 to avoid deterministic drift
    "ddpm1000": ("ddpm", 1000, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--samplers",
        type=str,
        nargs="+",
        default=["ddim50", "ddim250", "ddpm1000"],
        choices=list(SAMPLER_CONFIGS.keys()),
        help="Which samplers to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def export_real_test_set(cfg, output_dir: Path) -> Path:
    """Export the test set as PNGs in a directory (for clean-fid)."""
    real_dir = output_dir / "real"
    if real_dir.exists() and any(real_dir.iterdir()):
        log.info(f"Real test set already exported to {real_dir}")
        return real_dir
    real_dir.mkdir(parents=True, exist_ok=True)

    log.info("Exporting test set images to disk for FID computation...")
    test_set = get_dataset(name=cfg.data.name, root=cfg.data.root, train=False)
    for i, (img, _) in enumerate(test_set):
        # img is in [-1, 1], save in [0, 1]
        save_image((img + 1.0) / 2.0, real_dir / f"{i:05d}.png")
    log.info(f"Exported {len(test_set)} real images to {real_dir}")
    return real_dir


def generate_samples(
    model,
    diffusion: GaussianDiffusion,
    cfg,
    device: torch.device,
    sampler: str,
    steps: int,
    eta: float,
    num_samples: int,
    batch_size: int,
    output_dir: Path,
) -> Path:
    """Generate samples and save them as individual PNGs."""
    if output_dir.exists() and len(list(output_dir.glob("*.png"))) >= num_samples:
        log.info(f"Samples already exist in {output_dir}, skipping generation")
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n_generated = 0
    while n_generated < num_samples:
        batch = min(batch_size, num_samples - n_generated)
        shape = (batch, cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)

        if sampler == "ddpm":
            samples = diffusion.p_sample_loop(model, shape=shape, device=device)
        else:
            samples = diffusion.ddim_sample(
                model, shape=shape, num_inference_steps=steps, eta=eta, device=device
            )

        samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
        for j in range(batch):
            save_image(samples[j], output_dir / f"{n_generated + j:05d}.png")

        n_generated += batch
        log.info(f"[{sampler}-{steps}] Generated {n_generated}/{num_samples}")

    return output_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    set_seed(args.seed)
    log.info(f"Device: {device}, samplers: {args.samplers}")

    # 1. Resolve directories.
    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent.parent
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    # 2. Build model and load checkpoint (EMA weights only).
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
    load_checkpoint(args.checkpoint, model=model, ema=ema, map_location=device)
    log.info("Loaded EMA weights for evaluation")

    # 3. Export real test set.
    real_dir = export_real_test_set(cfg, eval_dir)

    # 4. For each sampler: generate -> FID -> IS.
    results: dict[str, dict[str, float]] = {}
    for sampler_key in args.samplers:
        sampler, steps, eta = SAMPLER_CONFIGS[sampler_key]
        fake_dir = eval_dir / f"samples_{sampler_key}"

        generate_samples(
            ema.ema_model,
            diffusion,
            cfg,
            device,
            sampler=sampler,
            steps=steps,
            eta=eta,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            output_dir=fake_dir,
        )

        log.info(f"Computing FID for {sampler_key}...")
        fid_score = compute_fid(real_dir, fake_dir)

        log.info(f"Computing IS for {sampler_key}...")
        is_mean, is_std = compute_inception_score(fake_dir, device=device)

        results[sampler_key] = {
            "fid": fid_score,
            "is_mean": is_mean,
            "is_std": is_std,
        }

    # 5. Save results JSON.
    results_path = eval_dir / "metrics.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {results_path}")

    # 6. Pretty print.
    log.info("=" * 60)
    log.info("Final results:")
    log.info(f"{'Sampler':<15} {'FID':>10} {'IS':>20}")
    for key, m in results.items():
        log.info(f"{key:<15} {m['fid']:>10.4f}   {m['is_mean']:>8.4f} ± {m['is_std']:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
