# DDPM (PyTorch)

A clean, from-scratch implementation of **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) and **DDIM** sampling (Song et al., 2020) in PyTorch.

> **Status:** 🚧 Work in progress.

## Overview

This repository implements DDPMs end-to-end without relying on `diffusers` or other high-level libraries. The goal is pedagogical clarity and faithfulness to the original papers, with production-grade engineering practices (modular code, externalised configs, reproducibility, tests).

**Targets:**
- FashionMNIST (28×28, grayscale) — primary benchmark
- CIFAR-10 (32×32, RGB) — stretch goal

**Implements:**
- Forward / reverse Gaussian diffusion
- Linear and cosine noise schedules
- UNet with sinusoidal timestep embeddings
- DDPM (stochastic) and DDIM (deterministic, accelerated) sampling
- EMA of model weights for high-quality sampling
- FID and Inception Score evaluation

## Project structure

```
ddpm-pytorch/
├── src/ddpm/             # Core package
│   ├── data/             # Datasets and preprocessing
│   ├── models/           # UNet, EMA
│   ├── diffusion/        # Noise schedules, forward/reverse processes
│   ├── training/         # Training loop, losses
│   ├── evaluation/       # FID, Inception Score
│   └── utils/            # Seed, logger, device, config, checkpoint
├── configs/              # YAML configs (base + dataset-specific overrides)
├── scripts/              # Entry points: train.py, sample.py, evaluate.py
└── tests/                # Sanity checks (pytest)
```

## Setup

Requires **Python 3.14+** and **`uv`** (https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Nestallum/ddpm-pytorch.git
cd ddpm-pytorch
uv sync --extra dev
```

This creates a `.venv/`, installs all dependencies (including PyTorch with CUDA 13.0 for Linux/Windows), and generates a lockfile for full reproducibility.

## Usage

> Coming soon.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. ICLR. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

## License

MIT.