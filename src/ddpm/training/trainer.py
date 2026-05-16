"""Trainer orchestrating the DDPM training loop.

Combines the model, diffusion process, dataloader, optimizer, EMA,
logging, and checkpointing into a single ``Trainer.train()`` call.
Designed to be reusable across datasets by passing a different config.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from ddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from ddpm.models.ema import EMA
from ddpm.training.losses import ddpm_loss
from ddpm.utils.checkpoint import save_checkpoint
from ddpm.utils.logger import get_logger

log = get_logger()


class Trainer:
    """Orchestrates the DDPM training loop.

    Parameters
    ----------
    model : nn.Module
        The noise-prediction model (typically a UNet).
    diffusion : GaussianDiffusion
        The diffusion process (q_sample for training, p_sample for eval).
    dataloader : DataLoader
        Yields batches of ``(images, labels)``. Labels are ignored.
    optimizer : optim.Optimizer
        Optimizer for the model parameters.
    ema : EMA
        Exponential moving average wrapper.
    cfg : DictConfig
        Full configuration (used for logging and to access training params).
    device : torch.device
        Device on which to train.
    output_dir : str or Path
        Directory for checkpoints, samples, and TensorBoard logs.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        ema: EMA,
        cfg: DictConfig,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.ema = ema
        self.cfg = cfg
        self.device = device

        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.samples_dir = self.output_dir / "samples"
        self.tb_dir = self.output_dir / "tensorboard"
        for d in (self.ckpt_dir, self.samples_dir, self.tb_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.step = 0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop based on ``cfg.training``."""
        train_cfg = self.cfg.training
        total_steps = train_cfg.total_steps
        log_every = train_cfg.log_every
        sample_every = train_cfg.sample_every
        ckpt_every = train_cfg.ckpt_every

        log.info(f"Starting training for {total_steps} steps on {self.device}")
        log.info(f"Output directory: {self.output_dir}")

        self.model.train()
        data_iter = self._infinite_iter(self.dataloader)
        pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
        loss_acc = 0.0
        loss_count = 0
        t_start = time.time()

        while self.step < total_steps:
            x, _ = next(data_iter)
            x = x.to(self.device, non_blocking=True)

            loss = self._train_step(x)

            loss_acc += loss
            loss_count += 1
            self.step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}")

            if self.step % log_every == 0:
                avg_loss = loss_acc / loss_count
                self.writer.add_scalar("train/loss", avg_loss, self.step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.step)
                loss_acc, loss_count = 0.0, 0

            if self.step % sample_every == 0:
                self._generate_and_log_samples()

            if self.step % ckpt_every == 0 or self.step == total_steps:
                self._save_checkpoint()

        pbar.close()
        elapsed = time.time() - t_start
        log.info(f"Training complete: {self.step} steps in {elapsed / 60:.1f} min")
        self.writer.close()

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, x: torch.Tensor) -> float:
        """Run one optimizer step on a batch of clean images."""
        b = x.size(0)
        t = torch.randint(0, self.diffusion.num_timesteps, (b,), device=self.device)
        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise)

        eps_pred = self.model(x_t, t)
        loss = ddpm_loss(eps_pred, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping (configurable, common stabilizer in diffusion).
        if self.cfg.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
        self.optimizer.step()
        self.ema.update(self.model)

        return loss.item()

    # ------------------------------------------------------------------
    # Periodic actions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_and_log_samples(self) -> None:
        """Generate a small batch with EMA weights and log a grid."""
        sample_cfg = self.cfg.sampling
        n_samples = sample_cfg.n_preview
        shape = (
            n_samples,
            self.cfg.model.in_channels,
            self.cfg.data.image_size,
            self.cfg.data.image_size,
        )

        self.ema.ema_model.eval()
        samples = self.diffusion.ddim_sample(
            self.ema.ema_model,
            shape=shape,
            num_inference_steps=sample_cfg.preview_steps,
            device=self.device,
        )
        # Map [-1, 1] back to [0, 1] for visualization.
        samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
        grid = make_grid(samples, nrow=int(n_samples**0.5))

        # Save to disk and log to TensorBoard.
        save_image(grid, self.samples_dir / f"step_{self.step:07d}.png")
        self.writer.add_image("samples/ema", grid, self.step)
        log.info(f"Step {self.step}: saved {n_samples} preview samples")

    def _save_checkpoint(self) -> None:
        """Save the current training state."""
        path = self.ckpt_dir / f"step_{self.step:07d}.pt"
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            ema=self.ema,
        )
        log.info(f"Step {self.step}: saved checkpoint to {path.name}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infinite_iter(loader: DataLoader):
        """Yield batches indefinitely, looping over the dataloader."""
        while True:
            yield from loader
