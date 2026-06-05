"""Compact sparse token representations for SSR MaxSim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass
class SparseTokenEmbeddings:
    """Per-sequence token embeddings as fixed-width top-k sparse rows.

    indices: (n_tokens, k) int32 — latent dimension ids, -1 for padding
    values:  (n_tokens, k) float32 — activation values (L2-normalized per token)
    """

    indices: np.ndarray
    values: np.ndarray
    n_latents: int

    @property
    def n_tokens(self) -> int:
        return int(self.indices.shape[0])

    @property
    def k(self) -> int:
        return int(self.indices.shape[1]) if self.indices.ndim == 2 else 0


def _l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return values / norms


def dense_tokens_to_sparse(
    token_embeddings: Tensor | np.ndarray,
    *,
    n_latents: int,
    topk: int | None = None,
    prune_topk: int | None = None,
    min_value: float = 1e-8,
) -> SparseTokenEmbeddings:
    """Convert (n_tokens, n_latents) dense/sparse-ish rows to packed top-k arrays."""
    if isinstance(token_embeddings, Tensor):
        x = token_embeddings.detach().float().cpu().numpy()
    else:
        x = np.asarray(token_embeddings, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D token embeddings, got shape {x.shape}")

    n_tok, dim = x.shape
    if dim != n_latents:
        raise ValueError(f"Expected n_latents={n_latents}, got {dim}")

    k_cap = topk or min(32, dim)
    if prune_topk is not None:
        k_cap = min(k_cap, prune_topk)

    indices = np.full((n_tok, k_cap), -1, dtype=np.int32)
    values = np.zeros((n_tok, k_cap), dtype=np.float32)

    for t in range(n_tok):
        row = x[t]
        nz = np.flatnonzero(row > min_value)
        if nz.size == 0:
            continue
        if nz.size > k_cap:
            order = np.argpartition(-row[nz], k_cap - 1)[:k_cap]
            nz = nz[order]
        vals = row[nz].astype(np.float32, copy=False)
        vals = vals / max(np.linalg.norm(vals), 1e-12)
        k_eff = nz.size
        indices[t, :k_eff] = nz.astype(np.int32, copy=False)
        values[t, :k_eff] = vals

    return SparseTokenEmbeddings(indices=indices, values=values, n_latents=n_latents)


def prune_sparse_rows(
    sparse: SparseTokenEmbeddings,
    prune_k: int,
) -> SparseTokenEmbeddings:
    """Keep at most prune_k highest-value active dimensions per token."""
    if prune_k <= 0 or prune_k >= sparse.k:
        return sparse
    n_tok = sparse.n_tokens
    indices = np.full((n_tok, prune_k), -1, dtype=np.int32)
    values = np.zeros((n_tok, prune_k), dtype=np.float32)
    for t in range(n_tok):
        row_idx = sparse.indices[t]
        row_val = sparse.values[t]
        valid = row_idx >= 0
        if not np.any(valid):
            continue
        row_idx = row_idx[valid]
        row_val = row_val[valid]
        if row_idx.size > prune_k:
            order = np.argpartition(-row_val, prune_k - 1)[:prune_k]
            row_idx = row_idx[order]
            row_val = row_val[order]
        k_eff = row_idx.size
        indices[t, :k_eff] = row_idx
        values[t, :k_eff] = row_val
    return SparseTokenEmbeddings(
        indices=indices, values=values, n_latents=sparse.n_latents
    )


def batch_dense_to_sparse(
    embeddings: Sequence[Tensor | np.ndarray],
    *,
    n_latents: int,
    topk: int | None = None,
    prune_topk: int | None = None,
) -> List[SparseTokenEmbeddings]:
    return [
        dense_tokens_to_sparse(
            emb,
            n_latents=n_latents,
            topk=topk,
            prune_topk=prune_topk,
        )
        for emb in embeddings
    ]


def sparse_to_coo_tensor(
    sparse: SparseTokenEmbeddings,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    block_id: int | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """Build (n_tokens, n_latents) torch.sparse_coo_tensor from packed rows.

    If ``block_id`` is set, only include columns in
    ``[block_id * block_size, (block_id + 1) * block_size)``.
    """
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lo = 0
    hi = sparse.n_latents
    if block_id is not None and block_size is not None:
        lo = block_id * block_size
        hi = min(lo + block_size, sparse.n_latents)
    for t in range(sparse.n_tokens):
        for j in range(sparse.k):
            c = int(sparse.indices[t, j])
            if c < 0:
                break
            if c < lo or c >= hi:
                continue
            rows.append(t)
            cols.append(c)
            vals.append(float(sparse.values[t, j]))

    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64),
            torch.zeros(0, dtype=dtype),
            size=(sparse.n_tokens, sparse.n_latents),
            device=device,
        ).coalesce()

    idx = torch.tensor([rows, cols], dtype=torch.int64)
    val = torch.tensor(vals, dtype=dtype)
    return torch.sparse_coo_tensor(
        idx,
        val,
        size=(sparse.n_tokens, sparse.n_latents),
        device=device,
    ).coalesce()


def save_sparse_corpus(
    path: str | Path,
    doc_ids: Sequence[str],
    embeddings: Sequence[SparseTokenEmbeddings],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        doc_ids=np.asarray(list(doc_ids), dtype=object),
        n_latents=np.int32(embeddings[0].n_latents if embeddings else 0),
        indices=np.asarray([e.indices for e in embeddings], dtype=object),
        values=np.asarray([e.values for e in embeddings], dtype=object),
    )


def load_sparse_corpus(path: str | Path) -> tuple[list[str], list[SparseTokenEmbeddings]]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    doc_ids = [str(x) for x in data["doc_ids"].tolist()]
    n_latents = int(data["n_latents"])
    embeddings = [
        SparseTokenEmbeddings(
            indices=np.asarray(ind),
            values=np.asarray(val),
            n_latents=n_latents,
        )
        for ind, val in zip(data["indices"], data["values"])
    ]
    return doc_ids, embeddings
