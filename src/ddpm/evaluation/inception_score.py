"""Inception Score (IS) evaluation.

The IS measures both the confidence of an InceptionV3 classifier on
generated images AND the diversity across classes. Higher is better.

We use the implementation from torchmetrics, which is well-maintained
and matches the standard formula from Salimans et al. (2016).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms

from ddpm.utils.logger import get_logger

log = get_logger()


def compute_inception_score(
    fake_dir: str | Path,
    splits: int = 10,
    device: torch.device | str = "cuda",
    batch_size: int = 64,
) -> tuple[float, float]:
    """Compute Inception Score on a directory of generated images.

    Parameters
    ----------
    fake_dir : str or Path
        Directory containing generated images.
    splits : int, optional
        Number of splits for computing IS mean and std. Defaults to 10.
    device : torch.device or str, optional
        Device to run InceptionV3 on.
    batch_size : int, optional
        Batch size for the InceptionV3 forward passes.

    Returns
    -------
    tuple of (float, float)
        ``(mean, std)`` of the Inception Score across splits.
    """
    fake_dir = Path(fake_dir)
    log.info(f"Computing Inception Score on {fake_dir}")

    # IS expects uint8 RGB tensors of shape (N, 3, H, W).
    inception = InceptionScore(splits=splits, normalize=False).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3),  # FashionMNIST: 1 -> 3
            transforms.PILToTensor(),  # uint8 [0, 255]
        ]
    )

    image_files = sorted(fake_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No PNG files found in {fake_dir}")

    batch_imgs: list[torch.Tensor] = []
    for i, img_path in enumerate(image_files):
        img = Image.open(img_path).convert("L")  # ensure grayscale source
        batch_imgs.append(transform(img))
        if len(batch_imgs) == batch_size or i == len(image_files) - 1:
            batch = torch.stack(batch_imgs).to(device)
            inception.update(batch)
            batch_imgs = []

    mean, std = inception.compute()
    log.info(f"IS = {mean.item():.4f} ± {std.item():.4f}")
    return float(mean.item()), float(std.item())
