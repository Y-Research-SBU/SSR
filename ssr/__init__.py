"""SSR + Sparse Autoencoder training."""

from . import _bootstrap  # noqa: F401  — must run before torch imports

from .model import SSR, build_ssr
from .losses import CombinedSAELoss, ContrastiveWithSAE, LossWeights, SAEWithUnsupervisedCL
from .sparse_autoencoder import SparseAutoencoder

__all__ = [
    "SSR",
    "CombinedSAELoss",
    "ContrastiveWithSAE",
    "LossWeights",
    "SAEWithUnsupervisedCL",
    "SparseAutoencoder",
    "build_ssr",
]
