"""Dataset loaders for DDPM training.

Wraps torchvision datasets with the normalization expected by the
forward diffusion process: images must lie in ``[-1, 1]`` so that they
are compatible with the centered Gaussian noise added in ``q_sample``.

Currently supports:
- FashionMNIST (1x28x28, grayscale)
- CIFAR-10     (3x32x32, RGB)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DatasetName = Literal["fashion_mnist", "cifar10"]


def _scale_to_neg_one_one(x: Tensor) -> Tensor:
    """Map a tensor from [0, 1] to [-1, 1].

    Defined at module level (not as a lambda) so that DataLoader workers
    can pickle the transform on Windows / Python 3.14+.
    """
    return 2.0 * x - 1.0


def _build_transform() -> transforms.Compose:
    """Standard preprocessing: tensor in [0, 1] then rescaled to [-1, 1].

    The shift to ``[-1, 1]`` centers the data around 0, matching the
    distribution of the noise added during the forward process.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),               # uint8 [0, 255] -> float [0, 1]
            transforms.Lambda(_scale_to_neg_one_one),   # [0, 1] -> [-1, 1]
        ]
    )


def get_dataset(
    name: DatasetName,
    root: str | Path = "data",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """Instantiate a torchvision dataset preprocessed for DDPM training.

    Parameters
    ----------
    name : {"fashion_mnist", "cifar10"}
        Which dataset to load.
    root : str or Path, optional
        Root directory where the raw files are stored or downloaded.
        Defaults to ``"data"`` (relative to the working directory).
    train : bool, optional
        Whether to load the training split. Defaults to True. The test
        split is reserved for FID evaluation and should not be touched
        during training.
    download : bool, optional
        Download the dataset if it is not already present locally.
        Defaults to True.

    Returns
    -------
    Dataset
        A torchvision dataset yielding ``(image, label)`` tuples where
        ``image`` is a tensor in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If ``name`` does not refer to a supported dataset.
    """
    transform = _build_transform()
    root = str(root)

    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=train, download=download, transform=transform)
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    raise ValueError(f"Unknown dataset '{name}'. Expected 'fashion_mnist' or 'cifar10'.")


def get_dataloader(
    name: DatasetName,
    batch_size: int,
    root: str | Path = "data",
    train: bool = True,
    num_workers: int = 4,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    drop_last: bool | None = None,
) -> DataLoader:
    """Return a ``DataLoader`` over the requested dataset.

    Defaults are tuned for training: shuffle and drop_last are enabled when
    ``train=True``, disabled otherwise. Pin memory is on by default for
    faster host-to-device transfers on CUDA.

    Parameters
    ----------
    name : {"fashion_mnist", "cifar10"}
        Dataset identifier (forwarded to ``get_dataset``).
    batch_size : int
        Mini-batch size.
    root : str or Path, optional
        Dataset root directory. Defaults to ``"data"``.
    train : bool, optional
        Whether to use the train split. Defaults to True.
    num_workers : int, optional
        Number of subprocesses for data loading. Defaults to 4. On Windows
        this can be set to 0 if multiprocessing causes issues.
    shuffle : bool, optional
        Whether to shuffle. If None, defaults to ``train``.
    pin_memory : bool, optional
        Defaults to True (faster H2D transfer with CUDA).
    drop_last : bool, optional
        Whether to drop the last incomplete batch. If None, defaults to
        ``train`` (drop during training for stable batch sizes).

    Returns
    -------
    DataLoader
        Iterable over batches.
    """
    if shuffle is None:
        shuffle = train
    if drop_last is None:
        drop_last = train

    dataset = get_dataset(name=name, root=root, train=train, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
