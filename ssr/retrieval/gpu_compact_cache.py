"""GPU-resident compact inverted-index postings (doc_idx + values)."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def ensure_gpu_compact_postings(
    index,
    *,
    device: str = "cuda",
    max_bytes: int | None = None,
) -> bool:
    """Upload ``index.compact`` doc_idx/values once; return False if over budget."""
    if not torch.cuda.is_available():
        return False
    if getattr(index, "_gpu_compact_ready", False):
        return True

    compact = index.compact
    n = int(compact.doc_idx.shape[0])
    need = n * 8
    if max_bytes is not None and need > int(max_bytes):
        logger.info(
            "Skip full GPU compact cache: need %.2f GiB > budget %.2f GiB",
            need / (1024**3),
            int(max_bytes) / (1024**3),
        )
        return False

    torch_dev = torch.device(
        "cuda" if device == "cuda" else device
    )
    doc = compact.doc_idx
    val = compact.values
    if val.dtype != np.float32:
        val = val.astype(np.float32, copy=False)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    gpu_doc = torch.from_numpy(np.ascontiguousarray(doc)).to(
        device=torch_dev, dtype=torch.int32, non_blocking=True
    )
    gpu_val = torch.from_numpy(np.ascontiguousarray(val)).to(
        device=torch_dev, dtype=torch.float32, non_blocking=True
    )
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)
    index._gpu_compact_doc_idx = gpu_doc
    index._gpu_compact_values = gpu_val
    index._gpu_compact_ready = True
    logger.info(
        "GPU compact postings: %d entries, %.2f GiB, upload %.0f ms",
        n,
        need / (1024**3),
        ms,
    )
    return True
