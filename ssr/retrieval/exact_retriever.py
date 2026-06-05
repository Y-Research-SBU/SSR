"""Exact sparse MaxSim retrieval (index-first; optional GPU on candidates only)."""

from __future__ import annotations

import heapq
import logging
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .global_index import load_or_build_global_index_from_bank, remap_sparse_embeddings
from .global_index_cache import default_cache_dir
from .inverted_index import (
    batch_exact_maxsim_via_index,
    build_chunk_index,
    build_chunk_index_from_flat_coo,
    exact_maxsim_via_index,
    query_candidate_doc_ids,
)
from .maxsim import maxsim_query_vs_doc_indices, maxsim_query_vs_documents
from .retriever import RetrieverConfig, SparseMaxSimRetriever
from .sparse_repr import SparseTokenEmbeddings
from .index_docs import materialize_sparse_docs_from_index
from .sparse_tensors import (
    SparseEmbeddingBank,
    coo_chunk_to_doc_sparse_list,
)
from .timing_stats import RetrievalTimingStats

logger = logging.getLogger(__name__)


class ExactSparseMaxSimRetriever:
    """Exact MaxSim via inverted index (default) or legacy brute-force torch.sparse.

    Index-first pipeline (``use_index_first=True``):

    **Synthetic bank (``bank_global_two_phase=True``, default):**
    1. Ingest *all* corpus shards into one global inverted index (optional latent reorder
       by co-activation / frequency into blocks).
    2. Run retrieval for all queries against that index (no per-shard index rebuild).

    **MTEB / per-chunk (legacy interleaved):**
    Build index per chunk, score queries, merge heaps, repeat.
    """

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig(mode="exact")
        self._encoder = SparseMaxSimRetriever(self.config)
        self.last_retrieval_timing: RetrievalTimingStats | None = None

    def encode_queries(self, *args, **kwargs):
        return self._encoder.encode_queries(*args, **kwargs)

    def encode_corpus(self, *args, **kwargs):
        return self._encoder.encode_corpus(*args, **kwargs)

    def retrieve(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        corpus: Sequence[SparseTokenEmbeddings],
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str = "cpu",
        show_progress: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        cfg = self.config
        if cfg.use_brute_force_torch and not cfg.use_index_first:
            return self._retrieve_torch_sparse(
                queries=queries,
                corpus=corpus,
                corpus_ids=corpus_ids,
                top_k=top_k,
                score_device=score_device,
                show_progress=show_progress,
            )
        use_gpu = (
            cfg.index_then_gpu
            and str(score_device).startswith("cuda")
            and torch.cuda.is_available()
        )
        if use_gpu:
            return self._retrieve_index_first(
                queries=queries,
                corpus=corpus,
                corpus_ids=corpus_ids,
                top_k=top_k,
                score_device=score_device,
                show_progress=show_progress,
                global_offset=0,
            )
        return self._retrieve_index_first(
            queries=queries,
            corpus=corpus,
            corpus_ids=corpus_ids,
            top_k=top_k,
            score_device="cpu",
            show_progress=show_progress,
            global_offset=0,
            force_cpu_index=True,
        )

    def retrieve_from_bank(
        self,
        bank: SparseEmbeddingBank,
        *,
        top_k: int = 100,
        score_device: str = "cpu",
        show_progress: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        queries = bank.query_as_sparse_list()
        corpus_ids = bank.corpus_doc_ids()
        cfg = self.config
        if cfg.use_brute_force_torch and not cfg.use_index_first:
            return self._retrieve_bank_torch(
                queries=queries,
                bank=bank,
                corpus_ids=corpus_ids,
                top_k=top_k,
                score_device=score_device,
                show_progress=show_progress,
            )
        if cfg.bank_global_two_phase:
            bank_dir = getattr(bank, "data_dir", None)
            return self._retrieve_bank_global_two_phase(
                queries=queries,
                bank=bank,
                corpus_ids=corpus_ids,
                top_k=top_k,
                score_device=score_device,
                show_progress=show_progress,
                bank_dir=bank_dir,
            )

        return self._retrieve_bank_interleaved(
            queries=queries,
            bank=bank,
            corpus_ids=corpus_ids,
            top_k=top_k,
            score_device=score_device,
            show_progress=show_progress,
        )

    def retrieve_from_e2e_index(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        index,
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str = "cpu",
        show_progress: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        """Score queries against a pre-built global E2E inverted index (no corpus re-encode)."""
        cfg = self.config
        n_queries = len(queries)
        force_cpu = (
            not cfg.index_then_gpu
            or score_device == "cpu"
            or not str(score_device).startswith("cuda")
            or not torch.cuda.is_available()
        )
        t_retrieve = time.perf_counter()
        heaps = self._score_queries_on_index(
            queries,
            index=index,
            corpus_ids=corpus_ids,
            top_k=top_k,
            global_offset=0,
            force_cpu_index=force_cpu,
            show_progress=show_progress,
            progress_desc="E2E retrieve [global index]",
        )
        retrieve_seconds = time.perf_counter() - t_retrieve
        self.last_retrieval_timing = RetrievalTimingStats(
            index_seconds=0.0,
            retrieve_seconds=retrieve_seconds,
            score_seconds=retrieve_seconds,
            n_queries=n_queries,
            n_index_builds=0,
        )
        return [_heap_to_ranked(h) for h in heaps]

    def _retrieve_bank_global_two_phase(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        bank: SparseEmbeddingBank,
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str,
        show_progress: bool,
        bank_dir: Path | None = None,
    ) -> List[List[Tuple[str, float]]]:
        """Phase 1: global index over all shards; phase 2: score all queries."""
        cfg = self.config
        n_queries = len(queries)
        t_wall = time.perf_counter()
        use_gpu = (
            cfg.index_then_gpu
            and str(score_device).startswith("cuda")
            and torch.cuda.is_available()
        )
        if str(score_device).startswith("cuda") and not torch.cuda.is_available():
            use_gpu = False
            score_device = "cpu"

        cache_dir = cfg.index_cache_dir
        if cache_dir is None and bank_dir is not None:
            cache_dir = default_cache_dir(bank_dir)

        t_index = time.perf_counter()
        built_fresh = True
        if bank_dir is not None:
            global_index, latent_remap, build_stats, built_fresh = (
                load_or_build_global_index_from_bank(
                    bank,
                    bank_dir=bank_dir,
                    block_size=cfg.block_size,
                    reorder_mode=cfg.latent_reorder_mode,
                    cooc_sample_rate=cfg.cooc_sample_rate,
                    index_cache_dir=cache_dir,
                    force_rebuild_index=cfg.force_rebuild_index,
                    save_index_cache=cfg.save_index_cache,
                    show_progress=show_progress,
                )
            )
        else:
            from .global_index import build_global_index_from_bank

            global_index, latent_remap, build_stats = build_global_index_from_bank(
                bank,
                block_size=cfg.block_size,
                reorder_mode=cfg.latent_reorder_mode,
                cooc_sample_rate=cfg.cooc_sample_rate,
                show_progress=show_progress,
            )
        index_seconds = time.perf_counter() - t_index

        if show_progress:
            logger.info(
                "Global index ready: %d postings, %d active latents, reorder=%s%s",
                build_stats.n_postings,
                build_stats.n_latents_active,
                build_stats.reorder_mode,
                " (from cache)" if not built_fresh else "",
            )
            if use_gpu:
                logger.info(
                    "Phase 2: index top-%d per query, then GPU rerank when pool <= %d "
                    "(no corpus shards).",
                    top_k,
                    cfg.gpu_maxsim_max_candidates,
                )
            else:
                accum = cfg.index_accum_device
                if accum in ("hybrid", "auto", "cuda"):
                    logger.info(
                        "Phase 2: index top-%d per query, index_accum_device=%s "
                        "(gpu_hot_budget_gb=%.1f).",
                        top_k,
                        accum,
                        cfg.gpu_hot_latent_budget_gb,
                    )
                elif cfg.index_two_phase:
                    logger.info(
                        "Phase 2: synthetic two-phase index top-%d per query "
                        "(coarse_topk=%d pool=%d).",
                        top_k,
                        cfg.coarse_topk,
                        cfg.two_phase_pool_size,
                    )
                else:
                    logger.info("Phase 2: index top-%d per query.", top_k)

        remapped_queries = [
            remap_sparse_embeddings(q, latent_remap) for q in queries
        ]

        t_score = time.perf_counter()
        if use_gpu and bank_dir is not None:
            heaps = self._score_queries_global_gpu(
                remapped_queries,
                bank=bank,
                index=global_index,
                latent_remap=latent_remap,
                corpus_ids=corpus_ids,
                top_k=top_k,
                score_device=score_device,
                show_progress=show_progress,
            )
        else:
            heaps = self._score_queries_on_index(
                remapped_queries,
                index=global_index,
                corpus_ids=corpus_ids,
                top_k=top_k,
                global_offset=0,
                force_cpu_index=True,
                show_progress=show_progress,
                progress_desc="Global retrieve [phase 2, block index]",
            )
        score_seconds = time.perf_counter() - t_score

        wall_seconds = time.perf_counter() - t_wall
        self.last_retrieval_timing = RetrievalTimingStats(
            index_seconds=index_seconds,
            retrieve_seconds=wall_seconds,
            score_seconds=score_seconds,
            n_queries=n_queries,
            n_index_builds=0 if not built_fresh else build_stats.n_shards_ingested,
        )
        if show_progress:
            self.last_retrieval_timing.log_summary(
                logger,
                prefix="Retrieval timing [global two-phase]",
            )
        return [_heap_to_ranked(h) for h in heaps]

    def _retrieve_bank_interleaved(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        bank: SparseEmbeddingBank,
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str,
        show_progress: bool,
    ) -> List[List[Tuple[str, float]]]:
        """Legacy: build index per shard, score queries, repeat."""
        cfg = self.config
        use_gpu = (
            cfg.index_then_gpu
            and str(score_device).startswith("cuda")
            and torch.cuda.is_available()
        )
        n_queries = len(queries)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]
        t_retrieve = time.perf_counter()
        index_seconds = 0.0
        score_seconds = 0.0
        n_index_builds = 0

        shard_iter = bank.iter_corpus_shards()
        if show_progress:
            label = "GPU candidates" if use_gpu else "CPU index"
            shard_iter = tqdm(
                shard_iter,
                total=bank.n_corpus_shards,
                desc=f"Exact retrieve [per-shard interleaved, {label}]",
            )

        for coo, global_start, n_docs in shard_iter:
            t_index = time.perf_counter()
            index = build_chunk_index_from_flat_coo(
                coo,
                n_docs=n_docs,
                tokens_per_doc=bank.doc_tokens,
                n_latents=bank.n_latents,
                block_size=cfg.block_size,
            )
            index_seconds += time.perf_counter() - t_index
            n_index_builds += 1
            chunk = None
            if use_gpu:
                t_mat = time.perf_counter()
                chunk = coo_chunk_to_doc_sparse_list(
                    coo,
                    n_docs=n_docs,
                    doc_tokens=bank.doc_tokens,
                    n_latents=bank.n_latents,
                    topk=bank.topk,
                )
                score_seconds += time.perf_counter() - t_mat
            t_score = time.perf_counter()
            self._score_chunk_index_first(
                queries=queries,
                chunk=chunk,
                index=index,
                corpus_ids=corpus_ids,
                heaps=heaps,
                top_k=top_k,
                score_device=score_device if use_gpu else "cpu",
                global_offset=int(global_start),
                force_cpu_index=not use_gpu,
            )
            score_seconds += time.perf_counter() - t_score

        retrieve_seconds = time.perf_counter() - t_retrieve
        self.last_retrieval_timing = RetrievalTimingStats(
            index_seconds=index_seconds,
            retrieve_seconds=retrieve_seconds,
            score_seconds=score_seconds,
            n_queries=n_queries,
            n_index_builds=n_index_builds,
        )
        return [_heap_to_ranked(h) for h in heaps]

    def _score_queries_on_index(
        self,
        queries: Sequence[SparseTokenEmbeddings],
        *,
        index,
        corpus_ids: Sequence[str],
        top_k: int,
        global_offset: int,
        force_cpu_index: bool,
        show_progress: bool,
        progress_desc: str,
    ) -> List[List[Tuple[float, str]]]:
        cfg = self.config
        n_queries = len(queries)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]
        q_ranges = range(0, n_queries, cfg.query_batch_size)
        if show_progress:
            q_ranges = tqdm(
                q_ranges,
                total=(n_queries + cfg.query_batch_size - 1) // cfg.query_batch_size,
                desc=progress_desc,
            )
        for q_start in q_ranges:
            q_end = min(q_start + cfg.query_batch_size, n_queries)
            batch_queries = queries[q_start:q_end]
            min_scores = self._heap_thresholds(heaps[q_start:q_end], top_k)
            batch_out = batch_exact_maxsim_via_index(
                batch_queries,
                index,
                top_docs=top_k,
                min_scores=min_scores,
                use_doc_latent_pruning=cfg.use_doc_latent_pruning,
                index_candidate_pool=cfg.index_candidate_pool,
                query_latent_top_k=cfg.query_latent_top_k,
                index_two_phase=cfg.index_two_phase,
                two_phase_pool_size=cfg.two_phase_pool_size,
                coarse_topk=cfg.coarse_topk,
                use_vectorized=cfg.use_vectorized_index,
                index_parallel_workers=cfg.index_parallel_workers,
                index_accum_device=cfg.index_accum_device,
                gpu_hot_budget_gb=cfg.gpu_hot_latent_budget_gb,
            )
            for local_qi, (doc_local, scores) in enumerate(batch_out):
                self._merge_heap_entries(
                    heaps[q_start + local_qi],
                    doc_local,
                    scores,
                    corpus_ids,
                    global_offset,
                    top_k,
                )
        return heaps

    def _score_queries_global_gpu(
        self,
        queries: Sequence[SparseTokenEmbeddings],
        *,
        bank: SparseEmbeddingBank,
        index,
        latent_remap: np.ndarray,
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str,
        show_progress: bool,
    ) -> List[List[Tuple[float, str]]]:
        """Phase 2: index top-pool exact scores, then GPU rerank when pool is small enough."""
        cfg = self.config
        n_queries = len(queries)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]
        device = score_device
        use_gpu = True
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            use_gpu = False

        q_ranges = range(0, n_queries, cfg.query_batch_size)
        if show_progress:
            q_ranges = tqdm(
                q_ranges,
                total=(n_queries + cfg.query_batch_size - 1) // cfg.query_batch_size,
                desc="Global retrieve [phase 2, index pool + GPU rerank]",
            )

        for q_start in q_ranges:
            q_end = min(q_start + cfg.query_batch_size, n_queries)
            for local_qi, query in enumerate(queries[q_start:q_end]):
                qi = q_start + local_qi
                min_sc = self._heap_thresholds(heaps[qi : qi + 1], top_k)[0]
                doc_local, scores = self._score_one_query_index_then_gpu(
                    query=query,
                    index=index,
                    n_latents=bank.n_latents,
                    doc_tokens=bank.doc_tokens,
                    topk=bank.topk,
                    pool_k=top_k,
                    min_score=min_sc,
                    device=device,
                    use_gpu=use_gpu,
                )
                if doc_local.size == 0:
                    continue
                keep = scores > min_sc
                if not np.any(keep):
                    continue
                self._merge_heap_entries(
                    heaps[qi],
                    doc_local[keep],
                    scores[keep],
                    corpus_ids,
                    0,
                    top_k,
                )
        return heaps

    def _score_one_query_index_then_gpu(
        self,
        *,
        query: SparseTokenEmbeddings,
        index,
        n_latents: int,
        doc_tokens: int,
        topk: int,
        pool_k: int,
        min_score: float,
        device: str,
        use_gpu: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Index top-pool_k exact MaxSim, optional GPU rerank on that pool."""
        cfg = self.config
        doc_local, scores = exact_maxsim_via_index(
            query,
            index,
            top_docs=pool_k,
            min_score=min_score,
            use_doc_latent_pruning=cfg.use_doc_latent_pruning,
            index_candidate_pool=cfg.index_candidate_pool,
            query_latent_top_k=cfg.query_latent_top_k,
            index_two_phase=cfg.index_two_phase,
            two_phase_pool_size=cfg.two_phase_pool_size,
            coarse_topk=cfg.coarse_topk,
            use_vectorized=cfg.use_vectorized_index,
            index_parallel_workers=cfg.index_parallel_workers,
            index_accum_device=cfg.index_accum_device,
            gpu_hot_budget_gb=cfg.gpu_hot_latent_budget_gb,
        )
        if doc_local.size == 0:
            return doc_local, scores
        if (
            not use_gpu
            or not cfg.gpu_rerank_index_pool
            or doc_local.size > cfg.gpu_maxsim_max_candidates
        ):
            return doc_local, scores

        docs = materialize_sparse_docs_from_index(
            index,
            doc_local,
            doc_tokens=doc_tokens,
            topk=topk,
        )
        _sub_idx, scores = maxsim_query_vs_doc_indices(
            query,
            docs,
            np.arange(len(docs), dtype=np.int64),
            device=device,
            batch_size=cfg.fine_batch_size,
            block_size=cfg.block_size,
            n_latents=n_latents,
        )
        return doc_local, scores

    def _heap_thresholds(self, heaps: List[List[Tuple[float, str]]], top_k: int) -> List[float]:
        out: List[float] = []
        for heap in heaps:
            if len(heap) >= top_k:
                out.append(float(heap[0][0]))
            else:
                out.append(0.0)
        return out

    def _merge_heap_entries(
        self,
        heap: List[Tuple[float, str]],
        doc_local: np.ndarray,
        scores: np.ndarray,
        corpus_ids: Sequence[str],
        global_offset: int,
        top_k: int,
    ) -> None:
        for doc_loc, score in zip(doc_local, scores):
            sc = float(score)
            if sc <= 0.0:
                continue
            global_idx = int(global_offset + doc_loc)
            cid = corpus_ids[global_idx]
            entry = (sc, cid)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif sc > heap[0][0]:
                heapq.heapreplace(heap, entry)

    def _score_chunk_index_first(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        chunk: Sequence[SparseTokenEmbeddings] | None,
        index,
        corpus_ids: Sequence[str],
        heaps: List[List[Tuple[float, str]]],
        top_k: int,
        score_device: str,
        global_offset: int,
        force_cpu_index: bool,
    ) -> None:
        cfg = self.config
        n_queries = len(queries)
        device = score_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            force_cpu_index = True

        for q_start in range(0, n_queries, cfg.query_batch_size):
            q_end = min(q_start + cfg.query_batch_size, n_queries)
            batch_queries = queries[q_start:q_end]
            min_scores = self._heap_thresholds(
                heaps[q_start:q_end], top_k
            )

            if force_cpu_index:
                batch_out = batch_exact_maxsim_via_index(
                    batch_queries,
                    index,
                    top_docs=top_k,
                    min_scores=min_scores,
                    use_doc_latent_pruning=cfg.use_doc_latent_pruning,
                    index_candidate_pool=cfg.index_candidate_pool,
                    query_latent_top_k=cfg.query_latent_top_k,
                    index_two_phase=cfg.index_two_phase,
                    two_phase_pool_size=cfg.two_phase_pool_size,
                    coarse_topk=cfg.coarse_topk,
                    use_vectorized=cfg.use_vectorized_index,
                    index_parallel_workers=cfg.index_parallel_workers,
                    index_accum_device=cfg.index_accum_device,
                    gpu_hot_budget_gb=cfg.gpu_hot_latent_budget_gb,
                )
                for local_qi, (doc_local, scores) in enumerate(batch_out):
                    self._merge_heap_entries(
                        heaps[q_start + local_qi],
                        doc_local,
                        scores,
                        corpus_ids,
                        global_offset,
                        top_k,
                    )
                continue

            # GPU: index top-k, then torch.sparse rerank when candidate count fits on GPU.
            for local_qi, query in enumerate(batch_queries):
                qi = q_start + local_qi
                min_sc = min_scores[local_qi]
                if chunk is None:
                    raise RuntimeError("chunk required for GPU candidate MaxSim")
                doc_local, scores = exact_maxsim_via_index(
                    query,
                    index,
                    top_docs=top_k,
                    min_score=min_sc,
                    use_doc_latent_pruning=cfg.use_doc_latent_pruning,
                    index_candidate_pool=cfg.index_candidate_pool,
            query_latent_top_k=cfg.query_latent_top_k,
            index_two_phase=cfg.index_two_phase,
            two_phase_pool_size=cfg.two_phase_pool_size,
            coarse_topk=cfg.coarse_topk,
                    use_vectorized=cfg.use_vectorized_index,
                    index_parallel_workers=cfg.index_parallel_workers,
                    index_accum_device=cfg.index_accum_device,
                    gpu_hot_budget_gb=cfg.gpu_hot_latent_budget_gb,
                )
                if doc_local.size == 0:
                    continue
                if (
                    cfg.gpu_rerank_index_pool
                    and doc_local.size <= cfg.gpu_maxsim_max_candidates
                ):
                    n_lat = chunk[0].n_latents
                    doc_local, scores = maxsim_query_vs_doc_indices(
                        query,
                        chunk,
                        doc_local,
                        device=device,
                        batch_size=cfg.fine_batch_size,
                        block_size=cfg.block_size,
                        n_latents=n_lat,
                    )
                    keep = scores > min_sc
                    doc_local = doc_local[keep]
                    scores = scores[keep]
                self._merge_heap_entries(
                    heaps[qi],
                    doc_local,
                    scores,
                    corpus_ids,
                    global_offset,
                    top_k,
                )

    def _retrieve_index_first(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        corpus: Sequence[SparseTokenEmbeddings],
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str,
        show_progress: bool,
        global_offset: int = 0,
        force_cpu_index: bool = False,
    ) -> List[List[Tuple[str, float]]]:
        cfg = self.config
        n_queries = len(queries)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]
        use_gpu = (
            not force_cpu_index
            and cfg.index_then_gpu
            and str(score_device).startswith("cuda")
            and torch.cuda.is_available()
        )

        chunk_starts = range(0, len(corpus), cfg.corpus_chunk_size)
        if show_progress:
            label = "GPU candidates" if use_gpu else "CPU index"
            chunk_starts = tqdm(
                chunk_starts,
                desc=f"Exact retrieve [index-first, {label}]",
            )

        t_retrieve = time.perf_counter()
        index_seconds = 0.0
        score_seconds = 0.0
        n_index_builds = 0

        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + cfg.corpus_chunk_size, len(corpus))
            chunk = corpus[chunk_start:chunk_end]
            if not chunk:
                continue
            t_index = time.perf_counter()
            index = build_chunk_index(
                chunk,
                n_latents=chunk[0].n_latents,
                block_size=cfg.block_size,
            )
            index_seconds += time.perf_counter() - t_index
            n_index_builds += 1
            t_score = time.perf_counter()
            self._score_chunk_index_first(
                queries=queries,
                chunk=chunk if use_gpu else None,
                index=index,
                corpus_ids=corpus_ids,
                heaps=heaps,
                top_k=top_k,
                score_device=score_device,
                global_offset=int(chunk_start) + global_offset,
                force_cpu_index=not use_gpu,
            )
            score_seconds += time.perf_counter() - t_score

        retrieve_seconds = time.perf_counter() - t_retrieve
        self.last_retrieval_timing = RetrievalTimingStats(
            index_seconds=index_seconds,
            retrieve_seconds=retrieve_seconds,
            score_seconds=score_seconds,
            n_queries=n_queries,
            n_index_builds=n_index_builds,
        )
        return [_heap_to_ranked(h) for h in heaps]

    def _retrieve_bank_torch(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        bank: SparseEmbeddingBank,
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str,
        show_progress: bool,
    ) -> List[List[Tuple[str, float]]]:
        """Legacy brute-force path over COO shards (slow; debugging only)."""
        cfg = self.config
        device = score_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        n_queries = len(queries)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]
        t_retrieve = time.perf_counter()

        shard_iter = bank.iter_corpus_shards()
        if show_progress:
            label = "GPU" if device.startswith("cuda") else "CPU"
            shard_iter = tqdm(
                shard_iter,
                total=bank.n_corpus_shards,
                desc=f"Exact retrieve [{label} torch.sparse brute, COO shards]",
            )

        for coo, global_start, n_docs in shard_iter:
            chunk = coo_chunk_to_doc_sparse_list(
                coo,
                n_docs=n_docs,
                doc_tokens=bank.doc_tokens,
                n_latents=bank.n_latents,
                topk=bank.topk,
            )
            for qi, query in enumerate(queries):
                scores = maxsim_query_vs_documents(
                    query,
                    chunk,
                    device=device,
                    batch_size=cfg.fine_batch_size,
                )
                self._merge_heap_entries(
                    heaps[qi],
                    np.arange(len(chunk), dtype=np.int64),
                    scores,
                    corpus_ids,
                    int(global_start),
                    top_k,
                )

        retrieve_seconds = time.perf_counter() - t_retrieve
        self.last_retrieval_timing = RetrievalTimingStats(
            index_seconds=0.0,
            retrieve_seconds=retrieve_seconds,
            score_seconds=retrieve_seconds,
            n_queries=n_queries,
            n_index_builds=0,
        )
        return [_heap_to_ranked(h) for h in heaps]

    def _retrieve_torch_sparse(
        self,
        *,
        queries: Sequence[SparseTokenEmbeddings],
        corpus: Sequence[SparseTokenEmbeddings],
        corpus_ids: Sequence[str],
        top_k: int,
        score_device: str,
        show_progress: bool,
    ) -> List[List[Tuple[str, float]]]:
        """Legacy: score every query–document pair via torch.sparse.mm."""
        logger.warning(
            "Brute-force torch.sparse MaxSim is enabled (use_index_first=False). "
            "This is very slow on large corpora."
        )
        cfg = self.config
        device = score_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        n_queries = len(queries)
        heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]
        t_retrieve = time.perf_counter()

        chunk_starts = range(0, len(corpus), cfg.corpus_chunk_size)
        if show_progress:
            label = "GPU" if device.startswith("cuda") else "CPU"
            chunk_starts = tqdm(
                chunk_starts,
                desc=f"Exact retrieve [{label} torch.sparse brute]",
            )

        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + cfg.corpus_chunk_size, len(corpus))
            chunk = corpus[chunk_start:chunk_end]
            if not chunk:
                continue

            for qi, query in enumerate(queries):
                scores = maxsim_query_vs_documents(
                    query,
                    chunk,
                    device=device,
                    batch_size=cfg.fine_batch_size,
                )
                self._merge_heap_entries(
                    heaps[qi],
                    np.arange(len(chunk), dtype=np.int64),
                    scores,
                    corpus_ids,
                    int(chunk_start),
                    top_k,
                )

        retrieve_seconds = time.perf_counter() - t_retrieve
        self.last_retrieval_timing = RetrievalTimingStats(
            index_seconds=0.0,
            retrieve_seconds=retrieve_seconds,
            score_seconds=retrieve_seconds,
            n_queries=n_queries,
            n_index_builds=0,
        )
        return [_heap_to_ranked(h) for h in heaps]


def _heap_to_ranked(heap: List[Tuple[float, str]]) -> List[Tuple[str, float]]:
    return [(cid, sc) for sc, cid in sorted(heap, key=lambda x: x[0], reverse=True)]


def verify_exact_equals_torch(
    query: SparseTokenEmbeddings,
    documents: Sequence[SparseTokenEmbeddings],
    *,
    n_latents: int,
    block_size: int = 512,
    atol: float = 1e-4,
    device: str = "cpu",
) -> bool:
    """Check inverted-index MaxSim matches torch.sparse on a tiny set."""
    index = build_chunk_index(documents, n_latents=n_latents, block_size=block_size)
    idx_scores = exact_maxsim_via_index(query, index, top_docs=0)
    idx_map = {int(d): float(s) for d, s in zip(idx_scores[0], idx_scores[1])}
    torch_scores = maxsim_query_vs_documents(
        query, documents, device=device, block_size=block_size, n_latents=n_latents
    )
    for i, s in enumerate(torch_scores):
        if abs(idx_map.get(i, 0.0) - float(s)) > atol:
            return False
    return True
