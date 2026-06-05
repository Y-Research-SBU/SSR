"""Multi-stage sparse MaxSim retriever for SSR."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple  # noqa: F401 used by RetrieverConfig

import numpy as np
import torch
from tqdm import tqdm

from .inverted_index import build_chunk_index, coarse_maxsim_via_index
from .maxsim import maxsim_query_vs_documents
from .sparse_repr import (
    SparseTokenEmbeddings,
    batch_dense_to_sparse,
    dense_tokens_to_sparse,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Shared config for pruned (fast) and exact (equivalent MaxSim) retrievers."""

    mode: Literal["pruned", "exact"] = "pruned"
    corpus_chunk_size: int = 5000
    block_size: int = 512
    prune_topk: int = 8
    final_topk: int | None = None
    n_candidates: int = 2000
    query_batch_size: int = 16
    encode_batch_size: int = 8
    fine_batch_size: int = 64
    use_fine_rerank: bool = True
    score_backend: Literal["auto", "cpu", "cuda", "index", "inverted", "torch"] = "auto"
    # Exact retrieval: index-first (skip zero-overlap docs), optional GPU on candidates only.
    use_index_first: bool = True
    index_then_gpu: bool = True
    use_brute_force_torch: bool = False
    gpu_maxsim_max_candidates: int = 16_384
    # When True, score with index top-k then exact GPU MaxSim on those candidates only.
    gpu_rerank_index_pool: bool = True
    use_doc_latent_pruning: bool = False
    # Block-UB candidate pool then exact MaxSim on pool only (lossy if pool < n_docs).
    index_candidate_pool: int = 0
    query_latent_top_k: int = 0
    index_two_phase: bool = False
    two_phase_pool_size: int = 32_768  # tuned: ~86ms/query, top-100 min≥0.97 on synthetic_1m
    coarse_topk: int = 8
    # NumPy segment aggregation (much faster than per-posting Python on large indexes).
    use_vectorized_index: bool = True
    index_parallel_workers: int = 0  # reserved; stamp path uses qt-parallel Numba
    # Posting accumulate: cpu | cuda (full H2D) | hybrid | auto (hybrid if CUDA).
    index_accum_device: Literal["cpu", "cuda", "auto", "hybrid"] = "cpu"
    gpu_hot_latent_budget_gb: float = 8.0
    # Synthetic bank: build one global index over all shards, then score all queries.
    bank_global_two_phase: bool = True
    latent_reorder_mode: Literal["none", "frequency", "cooc"] = "frequency"
    cooc_sample_rate: float = 0.01
    index_cache_dir: Path | None = None
    force_rebuild_index: bool = False
    save_index_cache: bool = True


class SparseMaxSimRetriever:
    """Multi-stage pruned retrieval (coarse top-N + optional fine rerank)."""

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()

    def encode_queries(
        self,
        model,
        queries: Sequence[str],
        *,
        n_latents: int,
        device: str,
        show_progress: bool = True,
    ) -> List[SparseTokenEmbeddings]:
        cfg = self.config
        out: List[SparseTokenEmbeddings] = []
        iterator = range(0, len(queries), cfg.encode_batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Encode queries",
                total=(len(queries) + cfg.encode_batch_size - 1)
                // cfg.encode_batch_size,
            )
        for start in iterator:
            batch = list(queries[start : start + cfg.encode_batch_size])
            embs = model.encode(
                batch,
                batch_size=len(batch),
                is_query=True,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            if not isinstance(embs, list):
                embs = [embs]
            k_final = cfg.final_topk or getattr(
                getattr(model, "sae_module", None), "topk", 32
            )
            sparse_batch = batch_dense_to_sparse(
                embs,
                n_latents=n_latents,
                topk=k_final,
            )
            out.extend(sparse_batch)
        return out

    def encode_corpus(
        self,
        model,
        corpus: Sequence[str],
        *,
        n_latents: int,
        device: str,
        show_progress: bool = True,
    ) -> List[SparseTokenEmbeddings]:
        cfg = self.config
        out: List[SparseTokenEmbeddings] = []
        iterator = range(0, len(corpus), cfg.encode_batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Encode corpus",
                total=(len(corpus) + cfg.encode_batch_size - 1)
                // cfg.encode_batch_size,
            )
        k_final = cfg.final_topk or getattr(
            getattr(model, "sae_module", None), "topk", 32
        )
        for start in iterator:
            batch = list(corpus[start : start + cfg.encode_batch_size])
            embs = model.encode(
                batch,
                batch_size=1,
                is_query=False,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            if not isinstance(embs, list):
                embs = [embs]
            out.extend(
                batch_dense_to_sparse(
                    embs,
                    n_latents=n_latents,
                    topk=k_final,
                )
            )
        return out

    def retrieve(
        self,
        *,
        query_sparse: Sequence[SparseTokenEmbeddings],
        query_sparse_fine: Sequence[SparseTokenEmbeddings] | None,
        corpus_sparse: Sequence[SparseTokenEmbeddings],
        corpus_ids: Sequence[str],
        top_k: int,
        device: str = "cpu",
        show_progress: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        """Return top_k (corpus_id, score) per query."""
        cfg = self.config
        n_queries = len(query_sparse)
        fine_queries = query_sparse_fine or query_sparse

        max_heap_size = max(top_k, cfg.n_candidates)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]

        chunk_starts = range(0, len(corpus_sparse), cfg.corpus_chunk_size)
        if show_progress:
            chunk_starts = tqdm(
                chunk_starts,
                desc="Retrieve (chunked inverted index)",
            )

        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + cfg.corpus_chunk_size, len(corpus_sparse))
            chunk = corpus_sparse[chunk_start:chunk_end]
            n_latents = chunk[0].n_latents if chunk else 0
            index = build_chunk_index(
                chunk,
                n_latents=n_latents,
                block_size=cfg.block_size,
            )

            for q_start in range(0, n_queries, cfg.query_batch_size):
                q_end = min(q_start + cfg.query_batch_size, n_queries)
                for local_qi, qi in enumerate(range(q_start, q_end)):
                    doc_local, scores = coarse_maxsim_via_index(
                        query_sparse[qi],
                        index,
                        top_docs=cfg.n_candidates,
                    )
                    for doc_loc, score in zip(doc_local, scores):
                        global_doc_idx = int(chunk_start + doc_loc)
                        cid = corpus_ids[global_doc_idx]
                        heap = heaps[qi]
                        entry = (float(score), cid)
                        if len(heap) < max_heap_size:
                            heapq.heappush(heap, entry)
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, entry)

        results: List[List[Tuple[str, float]]] = []
        for qi in range(n_queries):
            heap = heaps[qi]
            candidates = sorted(heap, key=lambda x: x[0], reverse=True)
            if not cfg.use_fine_rerank or not candidates:
                results.append([(cid, sc) for sc, cid in candidates[:top_k]])
                continue

            cand_ids = [cid for _, cid in candidates[: cfg.n_candidates]]
            id_to_idx = {corpus_ids[i]: i for i in range(len(corpus_ids))}
            cand_indices = [id_to_idx[c] for c in cand_ids if c in id_to_idx]
            cand_docs = [corpus_sparse[i] for i in cand_indices]
            fine_scores = maxsim_query_vs_documents(
                fine_queries[qi],
                cand_docs,
                device=device,
                batch_size=cfg.fine_batch_size,
            )
            reranked = sorted(
                zip([cand_ids[i] for i in range(len(cand_indices))], fine_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            results.append(reranked[:top_k])

        return results


def build_retriever(config: RetrieverConfig):
    """Factory: ``exact`` -> ExactSparseMaxSimRetriever, else pruned pipeline."""
    if config.mode == "exact":
        from .exact_retriever import ExactSparseMaxSimRetriever

        return ExactSparseMaxSimRetriever(config)
    return SparseMaxSimRetriever(config)
