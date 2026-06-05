"""GPU-resident postings for the largest (hottest) corpus latents."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .compact_postings import CompactPostings

logger = logging.getLogger(__name__)


@dataclass
class GpuHotLatentCache:
    """Columnar GPU slice for a subset of compact latent rows."""

    is_hot_row: np.ndarray  # bool[n_compact_rows]
    gpu_row_start: np.ndarray  # int64[n_compact_rows], -1 if cold
    gpu_row_end: np.ndarray
    gpu_doc_idx: torch.Tensor
    gpu_values: torch.Tensor
    nbytes: int
    n_hot_rows: int
    hot_posting_fraction: float

    def hot_row(self, compact_row: int) -> bool:
        return bool(self.is_hot_row[int(compact_row)])


def build_gpu_hot_latent_cache(
    compact: CompactPostings,
    *,
    budget_bytes: int = 8 * 1024**3,
    device: str = "cuda",
) -> GpuHotLatentCache | None:
    """Upload largest latents until ``budget_bytes`` (doc_idx + values)."""
    if not torch.cuda.is_available():
        return None
    offsets = compact.offsets
    sizes = np.diff(offsets)
    n_rows = int(sizes.shape[0])
    if n_rows == 0:
        return None

    order = np.argsort(-sizes, kind="mergesort")
    is_hot = np.zeros(n_rows, dtype=np.bool_)
    total = 0
    total_postings = int(sizes.sum())
    for ri in order:
        nbytes = int(sizes[ri]) * 8
        if nbytes <= 0:
            continue
        if total + nbytes > budget_bytes:
            continue
        is_hot[ri] = True
        total += nbytes

    n_hot = int(is_hot.sum())
    if n_hot == 0:
        return None

    gpu_row_start = np.full(n_rows, -1, dtype=np.int64)
    gpu_row_end = np.full(n_rows, -1, dtype=np.int64)
    doc_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    pos = 0
    for ri in range(n_rows):
        if not is_hot[ri]:
            continue
        s, e = int(offsets[ri]), int(offsets[ri + 1])
        gpu_row_start[ri] = pos
        pos_e = pos + (e - s)
        gpu_row_end[ri] = pos_e
        doc_parts.append(compact.doc_idx[s:e])
        val_parts.append(compact.values[s:e])
        pos = pos_e

    doc_cat = np.concatenate(doc_parts)
    val_cat = np.concatenate(val_parts)
    torch_dev = torch.device(
        "cuda" if device == "cuda" else device
    )
    gpu_doc = torch.from_numpy(doc_cat).to(device=torch_dev, dtype=torch.int32)
    gpu_val = torch.from_numpy(val_cat).to(device=torch_dev, dtype=torch.float32)
    hot_postings = int(sizes[is_hot].sum())
    frac = hot_postings / max(total_postings, 1)
    logger.info(
        "GPU hot latent cache: %d/%d rows, %.2f GiB, %.1f%% corpus postings",
        n_hot,
        n_rows,
        total / (1024**3),
        100.0 * frac,
    )
    return GpuHotLatentCache(
        is_hot_row=is_hot,
        gpu_row_start=gpu_row_start,
        gpu_row_end=gpu_row_end,
        gpu_doc_idx=gpu_doc,
        gpu_values=gpu_val,
        nbytes=total,
        n_hot_rows=n_hot,
        hot_posting_fraction=float(frac),
    )


def ensure_gpu_hot_cache(
    index,
    *,
    budget_bytes: int = 8 * 1024**3,
    device: str = "cuda",
    force_rebuild: bool = False,
) -> GpuHotLatentCache | None:
    """Attach ``index.gpu_hot_cache`` (build once per index + budget)."""
    scan = index.compact
    cache_key = int(budget_bytes)
    prev_key = getattr(index, "_gpu_hot_cache_key", None)
    if (
        not force_rebuild
        and getattr(index, "gpu_hot_cache", None) is not None
        and prev_key == cache_key
    ):
        return index.gpu_hot_cache
    cache = build_gpu_hot_latent_cache(
        scan,
        budget_bytes=budget_bytes,
        device=device,
    )
    index.gpu_hot_cache = cache
    index._gpu_hot_cache_key = cache_key
    return cache
