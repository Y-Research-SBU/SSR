"""Batched sparse COO helpers for global index merge (internal)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch

from .sparse_repr import SparseTokenEmbeddings

# Efficiency benchmarks: store COO values as fp16 (~half corpus disk vs fp32).
SYNTHETIC_VALUE_DTYPE = torch.float16


@dataclass(frozen=True)
class SparseEmbeddingBank:
    """Token-level embeddings stored as flat sparse COO matrices.

    Corpus layout: rows ``doc_id * doc_tokens + token_id`` (shape ``[n_docs * doc_tokens, n_latents]``).
    Query layout: rows ``query_id * query_tokens + token_id``.
    """

    n_latents: int
    topk: int
    doc_tokens: int
    query_tokens: int
    n_docs: int
    n_queries: int
    query_coo: torch.Tensor
    corpus_shard_dir: Path
    n_corpus_shards: int
    shard_size_docs: int = 10_000
    data_dir: Path | None = None

    def load_corpus_shard(
        self, shard_id: int
    ) -> tuple[torch.Tensor, int, int]:
        """Return ``(coo, global_doc_start, n_docs_in_shard)``."""
        path = self.corpus_shard_dir / f"corpus_shard_{shard_id:05d}.pt"
        blob = torch.load(path, map_location="cpu", weights_only=True)
        coo = blob["sparse_coo"].coalesce()
        return coo, int(blob["global_doc_start"]), int(blob["n_docs"])

    def query_as_sparse_list(self) -> List[SparseTokenEmbeddings]:
        """Materialize 1k-scale query set as per-query SparseTokenEmbeddings."""
        out: List[SparseTokenEmbeddings] = []
        for qi in range(self.n_queries):
            out.append(
                flat_coo_slice_to_sparse(
                    self.query_coo,
                    row_start=qi * self.query_tokens,
                    row_end=(qi + 1) * self.query_tokens,
                    n_latents=self.n_latents,
                    topk=self.topk,
                )
            )
        return out

    def iter_corpus_shards(self) -> Iterator[tuple[torch.Tensor, int, int]]:
        """Yield ``(coo, global_doc_start, n_docs_in_shard)``."""
        for shard_id in range(self.n_corpus_shards):
            path = self.corpus_shard_dir / f"corpus_shard_{shard_id:05d}.pt"
            blob = torch.load(path, map_location="cpu", weights_only=True)
            coo = blob["sparse_coo"].coalesce()
            n_docs = int(blob["n_docs"])
            global_start = int(blob["global_doc_start"])
            yield coo, global_start, n_docs

    def corpus_doc_ids(self) -> List[str]:
        return [str(i) for i in range(self.n_docs)]


def fetch_bank_docs_by_global_ids(
    bank: SparseEmbeddingBank,
    global_doc_ids: np.ndarray,
    *,
    latent_remap: np.ndarray | None = None,
) -> List[SparseTokenEmbeddings]:
    """Load documents by global doc id (loads only shards that contain requested ids)."""
    if global_doc_ids.size == 0:
        return []
    ids = np.asarray(global_doc_ids, dtype=np.int64)
    shard_size = bank.shard_size_docs
    by_shard: Dict[int, List[tuple[int, int]]] = defaultdict(list)
    for pos, gid in enumerate(ids):
        shard_id = min(int(gid) // shard_size, bank.n_corpus_shards - 1)
        by_shard[shard_id].append((pos, int(gid)))

    out: List[SparseTokenEmbeddings | None] = [None] * len(ids)
    for shard_id, items in by_shard.items():
        coo, global_start, _n_docs = bank.load_corpus_shard(shard_id)
        for pos, gid in items:
            local_doc = gid - global_start
            doc = flat_coo_slice_to_sparse(
                coo,
                row_start=local_doc * bank.doc_tokens,
                row_end=(local_doc + 1) * bank.doc_tokens,
                n_latents=bank.n_latents,
                topk=bank.topk,
            )
            if latent_remap is not None:
                from .global_index import remap_sparse_embeddings

                doc = remap_sparse_embeddings(doc, latent_remap)
            out[pos] = doc
    if any(d is None for d in out):
        raise RuntimeError("fetch_bank_docs_by_global_ids: missing document(s)")
    return out  # type: ignore[return-value]


def generate_random_token_coo(
    n_tokens: int,
    n_latents: int,
    topk: int,
    *,
    seed: int | None = None,
    dtype: torch.dtype = SYNTHETIC_VALUE_DTYPE,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create ``(n_tokens, n_latents)`` sparse COO with exactly ``topk`` nnz per row."""
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    rows = torch.arange(n_tokens, device=device, dtype=torch.int64).repeat_interleave(topk)
    cols = torch.randint(
        0,
        n_latents,
        (n_tokens, topk),
        generator=g,
        device=device,
        dtype=torch.int64,
    ).reshape(-1)
    vals = torch.rand(n_tokens, topk, generator=g, device=device, dtype=dtype).reshape(-1)
    row_norms = vals.view(n_tokens, topk).float().norm(dim=1).clamp(min=1e-8)
    vals = (vals.float() / row_norms.repeat_interleave(topk)).to(dtype)

    indices = torch.stack([rows, cols], dim=0)
    return torch.sparse_coo_tensor(
        indices,
        vals,
        size=(n_tokens, n_latents),
        device=device,
        dtype=dtype,
    ).coalesce()


def flat_coo_slice_to_sparse(
    coo: torch.Tensor,
    *,
    row_start: int,
    row_end: int,
    n_latents: int,
    topk: int,
) -> SparseTokenEmbeddings:
    """Extract one sequence (doc or query) from a flat COO matrix."""
    coo = coo.coalesce()
    idx = coo.indices()
    vals = coo.values()
    mask = (idx[0] >= row_start) & (idx[0] < row_end)
    if not mask.any():
        return SparseTokenEmbeddings(
            indices=np.full((row_end - row_start, topk), -1, dtype=np.int32),
            values=np.zeros((row_end - row_start, topk), dtype=np.float32),
            n_latents=n_latents,
        )

    local_rows = (idx[0][mask] - row_start).cpu().numpy()
    cols = idx[1][mask].cpu().numpy()
    values = vals[mask].cpu().numpy().astype(np.float32)

    n_tok = row_end - row_start
    indices = np.full((n_tok, topk), -1, dtype=np.int32)
    values_out = np.zeros((n_tok, topk), dtype=np.float32)
    counts = np.zeros(n_tok, dtype=np.int32)

    for r, c, v in zip(local_rows, cols, values):
        t = int(r)
        j = int(counts[t])
        if j < topk:
            indices[t, j] = int(c)
            values_out[t, j] = float(v)
            counts[t] += 1

    return SparseTokenEmbeddings(indices=indices, values=values_out, n_latents=n_latents)


def sparse_list_to_flat_coo(
    docs: Sequence[SparseTokenEmbeddings],
    *,
    doc_tokens: int,
    dtype: torch.dtype = SYNTHETIC_VALUE_DTYPE,
) -> torch.Tensor:
    """Pack encoded documents into a flat ``(n_docs * doc_tokens, n_latents)`` COO on CPU."""
    if not docs:
        n_latents = 0
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64),
            torch.zeros(0, dtype=dtype),
            size=(0, n_latents),
        ).coalesce()

    n_latents = int(docs[0].n_latents)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for local_d, doc in enumerate(docs):
        if int(doc.n_latents) != n_latents:
            raise ValueError("All documents must share the same n_latents")
        base = int(local_d) * int(doc_tokens)
        n_tok = min(int(doc.n_tokens), int(doc_tokens))
        for t in range(n_tok):
            row_idx = base + t
            for j in range(doc.k):
                c = int(doc.indices[t, j])
                if c < 0:
                    break
                v = float(doc.values[t, j])
                if v <= 0.0:
                    continue
                rows.append(row_idx)
                cols.append(c)
                vals.append(v)

    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64),
            torch.zeros(0, dtype=dtype),
            size=(len(docs) * doc_tokens, n_latents),
        ).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.int64)
    values = torch.tensor(vals, dtype=dtype)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(len(docs) * doc_tokens, n_latents),
        dtype=dtype,
    ).coalesce()


def coo_chunk_to_doc_sparse_list(
    coo: torch.Tensor,
    *,
    n_docs: int,
    doc_tokens: int,
    n_latents: int,
    topk: int,
) -> List[SparseTokenEmbeddings]:
    """Convert one corpus shard COO into per-document SparseTokenEmbeddings."""
    return [
        flat_coo_slice_to_sparse(
            coo,
            row_start=d * doc_tokens,
            row_end=(d + 1) * doc_tokens,
            n_latents=n_latents,
            topk=topk,
        )
        for d in range(n_docs)
    ]


def estimate_coo_nnz(n_rows: int, topk: int) -> int:
    return n_rows * topk


def estimate_coo_bytes(
    n_rows: int,
    topk: int,
    *,
    value_bytes: int = 2,
) -> int:
    """Estimate on-disk COO payload: int64 row/col indices + value array."""
    nnz = estimate_coo_nnz(n_rows, topk)
    return nnz * (8 + 8 + value_bytes)
