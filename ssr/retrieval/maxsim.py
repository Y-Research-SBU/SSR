"""Exact and batched MaxSim on sparse token embeddings."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .inverted_index import QueryBlockPlan, build_query_block_plan
from .sparse_repr import SparseTokenEmbeddings, sparse_to_coo_tensor


def _maxsim_from_token_sim_matrix(sim: torch.Tensor) -> float:
    if sim.layout != torch.sparse_coo:
        sim = sim.to_sparse_coo()
    dense = sim.to_dense()
    return float(dense.max(dim=1).values.sum().item())


def maxsim_pair_blockwise(
    query: SparseTokenEmbeddings,
    document: SparseTokenEmbeddings,
    *,
    block_size: int,
    n_latents: int,
    plan: QueryBlockPlan | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Exact MaxSim: accumulate per-block token–token dots, then max + sum."""
    if query.n_tokens == 0 or document.n_tokens == 0:
        return 0.0
    if plan is None:
        plan = build_query_block_plan(
            query, block_size=block_size, n_latents=n_latents
        )
    n_qt = query.n_tokens
    n_dt = document.n_tokens
    acc = torch.zeros((n_qt, n_dt), device=device, dtype=torch.float32)
    for block_id in plan.active_block_ids:
        q_b = sparse_to_coo_tensor(
            query,
            device=device,
            block_id=block_id,
            block_size=block_size,
        )
        if q_b._nnz() == 0:
            continue
        d_b = sparse_to_coo_tensor(
            document,
            device=device,
            block_id=block_id,
            block_size=block_size,
        )
        if d_b._nnz() == 0:
            continue
        sim = torch.sparse.mm(q_b, d_b.transpose(0, 1))
        acc += sim.to_dense()
    return float(acc.max(dim=1).values.sum().item())


def maxsim_pair(
    query: SparseTokenEmbeddings,
    document: SparseTokenEmbeddings,
    *,
    device: torch.device | str = "cpu",
    block_size: Optional[int] = None,
    n_latents: Optional[int] = None,
) -> float:
    """Exact MaxSim between one query and one document (sparse dot + max + sum)."""
    if block_size is not None and n_latents is not None:
        return maxsim_pair_blockwise(
            query,
            document,
            block_size=block_size,
            n_latents=n_latents,
            device=device,
        )
    if query.n_tokens == 0 or document.n_tokens == 0:
        return 0.0

    q = sparse_to_coo_tensor(query, device=device)
    d = sparse_to_coo_tensor(document, device=device)
    sim = torch.sparse.mm(q, d.transpose(0, 1))
    return _maxsim_from_token_sim_matrix(sim)


def maxsim_query_vs_documents(
    query: SparseTokenEmbeddings,
    documents: Sequence[SparseTokenEmbeddings],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    block_size: Optional[int] = None,
    n_latents: Optional[int] = None,
) -> np.ndarray:
    """Score one query against many documents using block-wise sparse matmul."""
    if not documents:
        return np.array([], dtype=np.float32)
    use_blocks = block_size is not None and n_latents is not None
    plan = (
        build_query_block_plan(query, block_size=block_size, n_latents=n_latents)
        if use_blocks
        else None
    )
    scores = np.zeros(len(documents), dtype=np.float32)

    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        batch_scores = []
        for doc in documents[start:end]:
            if doc.n_tokens == 0:
                batch_scores.append(0.0)
                continue
            if use_blocks:
                batch_scores.append(
                    maxsim_pair_blockwise(
                        query,
                        doc,
                        block_size=block_size,
                        n_latents=n_latents,
                        plan=plan,
                        device=device,
                    )
                )
            else:
                q = sparse_to_coo_tensor(query, device=device)
                d = sparse_to_coo_tensor(doc, device=device)
                sim = torch.sparse.mm(q, d.transpose(0, 1))
                batch_scores.append(_maxsim_from_token_sim_matrix(sim))
        scores[start:end] = np.asarray(batch_scores, dtype=np.float32)
    return scores


def maxsim_query_vs_doc_indices(
    query: SparseTokenEmbeddings,
    corpus_chunk: Sequence[SparseTokenEmbeddings],
    doc_indices: Union[Sequence[int], np.ndarray],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    block_size: Optional[int] = None,
    n_latents: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact MaxSim for ``query`` vs ``corpus_chunk[i]`` for ``i in doc_indices`` only."""
    if len(doc_indices) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    idx = np.asarray(doc_indices, dtype=np.int64)
    docs = [corpus_chunk[int(i)] for i in idx]
    scores = maxsim_query_vs_documents(
        query,
        docs,
        device=device,
        batch_size=batch_size,
        block_size=block_size,
        n_latents=n_latents,
    )
    return idx, scores


def maxsim_torch_matches_blockwise(
    query: SparseTokenEmbeddings,
    documents: Sequence[SparseTokenEmbeddings],
    *,
    block_size: int,
    n_latents: int,
    device: str = "cpu",
    atol: float = 1e-4,
) -> bool:
    """True if full sparse.mm MaxSim equals block-accumulated MaxSim per document."""
    full = maxsim_query_vs_documents(
        query, documents, device=device, block_size=None, n_latents=None
    )
    blocked = maxsim_query_vs_documents(
        query,
        documents,
        device=device,
        block_size=block_size,
        n_latents=n_latents,
    )
    return np.allclose(full, blocked, atol=atol, rtol=0)


def maxsim_queries_vs_documents_batched(
    queries: Sequence[SparseTokenEmbeddings],
    documents: Sequence[SparseTokenEmbeddings],
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """All-pairs MaxSim matrix (n_queries, n_docs) for small candidate sets."""
    out = np.zeros((len(queries), len(documents)), dtype=np.float32)
    for qi, query in enumerate(queries):
        out[qi] = maxsim_query_vs_documents(
            query, documents, device=device, batch_size=64
        )
    return out


def maxsim_chunk_dense_fallback(
    query_dense: torch.Tensor,
    docs_dense: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Dense MaxSim (ash,bth->abst) for validation on tiny sets."""
    q = query_dense.to(device)
    d = docs_dense.to(device)
    scores = torch.einsum("qsh,dth->qdst", q, d)
    return scores.max(dim=-1).values.sum(dim=-1)
