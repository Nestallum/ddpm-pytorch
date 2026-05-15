"""YAML configuration loading with hierarchical overrides.

Configs are split into a ``base.yaml`` shared across all experiments and
dataset-specific files (e.g. ``fashion_mnist.yaml``) that override only
the fields that differ. The loader merges them with :func:`OmegaConf.merge`
so dataset-specific values take precedence.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> DictConfig:
    """Load a YAML config, optionally merged with a base config.

    Parameters
    ----------
    config_path : str or Path
        Path to the main configuration file (typically dataset-specific).
    base_path : str or Path, optional
        Path to a base config that ``config_path`` overrides. If None,
        only ``config_path`` is loaded.

    Returns
    -------
    DictConfig
        The merged configuration as an OmegaConf object. Access fields via
        dot notation (e.g. ``cfg.training.batch_size``).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if base_path is not None:
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        base_cfg = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)

    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a config to disk in YAML format.

    Useful for snapshotting the exact config used for a training run
    inside the run's output directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f)
