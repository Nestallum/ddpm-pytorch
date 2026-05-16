"""FID (Fréchet Inception Distance) evaluation.

Wraps the ``clean-fid`` library, which is the modern reference
implementation (Parmar et al., 2022). Unlike older FID implementations,
clean-fid handles image resizing consistently across PyTorch and TensorFlow,
which removes a major source of result discrepancies between papers.

For grayscale datasets (FashionMNIST), images are replicated to 3 channels
before being passed to InceptionV3.
"""

from __future__ import annotations

from pathlib import Path

from cleanfid import fid

from ddpm.utils.logger import get_logger

log = get_logger()


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    mode: str = "clean",
    num_workers: int = 4,
    batch_size: int = 64,
) -> float:
    """Compute FID between two directories of images.

    Parameters
    ----------
    real_dir : str or Path
        Directory containing real images (the reference).
    fake_dir : str or Path
        Directory containing generated images.
    mode : str, optional
        clean-fid mode. ``"clean"`` is the recommended modern setting.
    num_workers : int, optional
        DataLoader workers for image preprocessing.
    batch_size : int, optional
        Batch size for the InceptionV3 forward passes.

    Returns
    -------
    float
        FID score (lower is better).
    """
    real_dir = str(real_dir)
    fake_dir = str(fake_dir)
    log.info(f"Computing FID: real={real_dir}, fake={fake_dir}")

    score = fid.compute_fid(
        real_dir,
        fake_dir,
        mode=mode,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    log.info(f"FID = {score:.4f}")
    return float(score)
