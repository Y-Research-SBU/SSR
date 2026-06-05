"""Block-partitioned inverted index for sparse latent dimensions."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .compact_postings import (
    CompactPostings,
    Posting,
    build_doc_latent_max,
    compact_from_latent_major_coo,
    postings_dict_to_compact,
    query_latent_weight_sums,
    segment_max_by_doc,
)
from .index_accumulate_fast import accumulate_maxsim_fast, sum_per_qt_fast
from .sparse_repr import SparseTokenEmbeddings


@dataclass
class QueryBlockPlan:
    """Query nonzeros grouped by latent block (skip blocks with no query activation)."""

    n_blocks: int
    active_block_ids: Tuple[int, ...]
    # block_id -> latent -> [(query_token_idx, weighted_value), ...]
    entries_by_block: Tuple[Dict[int, List[Tuple[int, float]]], ...]


def build_query_block_plan(
    query: SparseTokenEmbeddings,
    *,
    block_size: int,
    n_latents: int,
    query_token_weights: np.ndarray | None = None,
) -> QueryBlockPlan:
    n_blocks = (n_latents + block_size - 1) // block_size
    entries: List[Dict[int, List[Tuple[int, float]]]] = [
        {} for _ in range(n_blocks)
    ]
    active: List[int] = []
    for qt in range(query.n_tokens):
        w_q = 1.0 if query_token_weights is None else float(query_token_weights[qt])
        for j in range(query.k):
            latent = int(query.indices[qt, j])
            if latent < 0:
                break
            bid = latent // block_size
            v_q = float(query.values[qt, j]) * w_q
            bucket = entries[bid].setdefault(latent, [])
            bucket.append((qt, v_q))
    active_ids = tuple(bid for bid, d in enumerate(entries) if d)
    return QueryBlockPlan(
        n_blocks=n_blocks,
        active_block_ids=active_ids,
        entries_by_block=tuple(entries),
    )


@dataclass
class BlockInvertedIndex:
    """Inverted index: latent_id -> postings (doc_idx, token_idx, value).

    Postings are stored in columnar :class:`CompactPostings` (~10 B/row in RAM).
    ``latent_max`` holds per-latent max activation in the corpus (for cheap bounds).
    """

    n_latents: int
    block_size: int
    compact: CompactPostings = field(default_factory=CompactPostings.empty)
    n_docs: int = 0
    # Cached latent ids per block (corpus postings only).
    block_latents: List[List[int]] = field(default_factory=list)
    # O(1) latent_id -> compact row index (-1 if absent).
    latent_row_index: np.ndarray | None = None
    # float32[n_latents], max posting value per latent (0 if inactive).
    latent_max: np.ndarray | None = None
    # Per-(latent, doc) max value for safe score upper bounds (smaller than token postings).
    doc_latent_max: CompactPostings | None = None
    # float16[n_blocks, n_docs]: per-block max activation per doc (~64 MiB @ 1M docs).
    block_doc_max: np.ndarray | None = None
    # Coarse postings (e.g. top-8 per doc token) for two-phase retrieval (~25% of full).
    compact_coarse: CompactPostings | None = None
    coarse_topk: int = 8
    # Optional GPU postings for largest latents (hybrid accumulate).
    gpu_hot_cache: object | None = None

    @property
    def n_blocks(self) -> int:
        return (self.n_latents + self.block_size - 1) // self.block_size

    def block_id(self, latent: int) -> int:
        return latent // self.block_size

    def build_from_corpus(
        self,
        corpus: Sequence[SparseTokenEmbeddings],
        *,
        sort_postings: bool = True,
    ) -> None:
        del sort_postings  # compact layout is always sorted per latent
        self.n_docs = len(corpus)
        postings: Dict[int, List[Posting]] = {}
        for doc_idx, sparse in enumerate(corpus):
            for t in range(sparse.n_tokens):
                for j in range(sparse.k):
                    latent = int(sparse.indices[t, j])
                    if latent < 0:
                        break
                    val = float(sparse.values[t, j])
                    if val <= 0.0:
                        continue
                    postings.setdefault(latent, []).append(
                        Posting(doc_idx=doc_idx, token_idx=t, value=val)
                    )
        self.compact = postings_dict_to_compact(postings)
        self.finalize_block_latents()

    def iter_block_latents(self, block: int) -> List[int]:
        if self.block_latents:
            return self.block_latents[block]
        lo = block * self.block_size
        hi = min(lo + self.block_size, self.n_latents)
        lids = self.compact.latent_ids
        mask = (lids >= lo) & (lids < hi)
        return lids[mask].astype(int).tolist()

    def finalize_block_latents(self) -> None:
        """Cache sorted latent ids per block and latent -> row index."""
        self.block_latents = [self.iter_block_latents(b) for b in range(self.n_blocks)]
        lids = self.compact.latent_ids
        if self.n_latents > 0 and lids.size > 0:
            row = np.full(self.n_latents, -1, dtype=np.int32)
            row[lids] = np.arange(lids.shape[0], dtype=np.int32)
            self.latent_row_index = row
        else:
            self.latent_row_index = None

    def memory_bytes_estimate(self) -> int:
        return self.compact.nbytes()


def ensure_doc_latent_max(
    index: BlockInvertedIndex,
    *,
    show_progress: bool = False,
) -> CompactPostings:
    """Build ``doc_latent_max`` on demand (for safe upper-bound pruning only)."""
    dlm = index.doc_latent_max
    if dlm is not None and dlm.n_postings > 0:
        return dlm
    index.doc_latent_max = build_doc_latent_max(
        index.compact, show_progress=show_progress
    )
    return index.doc_latent_max


def build_chunk_index(
    corpus_chunk: Sequence[SparseTokenEmbeddings],
    *,
    n_latents: int,
    block_size: int = 512,
) -> BlockInvertedIndex:
    index = BlockInvertedIndex(n_latents=n_latents, block_size=block_size)
    index.build_from_corpus(corpus_chunk)
    return index


def build_chunk_index_from_flat_coo(
    coo: torch.Tensor,
    *,
    n_docs: int,
    tokens_per_doc: int,
    n_latents: int,
    block_size: int = 512,
) -> BlockInvertedIndex:
    """Build inverted index from ``(n_docs * tokens_per_doc, n_latents)`` sparse COO."""
    index = BlockInvertedIndex(n_latents=n_latents, block_size=block_size)
    index.n_docs = n_docs
    coo = coo.coalesce()
    rows = coo.indices()[0].cpu().numpy()
    cols = coo.indices()[1].cpu().numpy()
    vals = coo.values().cpu().numpy()

    pos = vals > 0.0
    rows, cols, vals = rows[pos], cols[pos], vals[pos]
    doc_idx = rows // tokens_per_doc
    token_idx = rows % tokens_per_doc

    order = np.argsort(cols, kind="mergesort")
    cols = cols[order]
    doc_idx = doc_idx[order]
    token_idx = token_idx[order]
    vals = vals[order]

    index.compact = compact_from_latent_major_coo(cols, doc_idx, token_idx, vals)
    index.finalize_block_latents()
    return index


@dataclass
class TokenPairAccumulator:
    """Accumulate dot-product contributions per (doc_idx, doc_token) pair."""

    scores: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def add(self, doc_idx: int, token_idx: int, delta: float) -> None:
        key = (doc_idx, token_idx)
        self.scores[key] = self.scores.get(key, 0.0) + delta

    def doc_scores_max_over_tokens(self) -> Dict[int, float]:
        per_doc: Dict[int, float] = {}
        for (doc_idx, _), val in self.scores.items():
            prev = per_doc.get(doc_idx)
            if prev is None or val > prev:
                per_doc[doc_idx] = val
        return per_doc


def exact_maxsim_via_index(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    *,
    top_docs: int = 0,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
    use_doc_latent_pruning: bool = False,
    index_candidate_pool: int = 0,
    query_latent_top_k: int = 0,
    index_two_phase: bool = False,
    two_phase_pool_size: int = 16_384,
    coarse_topk: int = 8,
    scan_postings: CompactPostings | None = None,
    use_vectorized: bool = True,
    index_parallel_workers: int = 0,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact MaxSim over a corpus chunk via inverted index (CPU, no approximation).

    Mathematically identical to dense ColBERT MaxSim when embeddings are the
    same sparse SAE token vectors (nonzeros only at top-k latent dimensions):

        score(q, d) = sum_t max_s dot(q_t, d_s)

    Only intersecting latent dimensions contribute to each dot product.

    Parameters
    ----------
    top_docs
        If > 0, return only the top-N documents by score (for heap merging).
        If <= 0, return scores for every document in the chunk index.
    index_two_phase
        Coarse postings + coarse query top-k, then exact MaxSim on a candidate pool.
    """
    if index_two_phase:
        from .index_pruning import two_phase_maxsim_via_index

        return two_phase_maxsim_via_index(
            query,
            index,
            top_docs=top_docs,
            min_score=min_score,
            query_token_weights=query_token_weights,
            pool_size=two_phase_pool_size,
            coarse_topk=coarse_topk,
            query_latent_top_k=query_latent_top_k,
            use_vectorized=use_vectorized,
            index_parallel_workers=index_parallel_workers,
            index_accum_device=index_accum_device,
            cuda_device=cuda_device,
            gpu_hot_budget_gb=gpu_hot_budget_gb,
        )
    return _maxsim_via_index_impl(
        query,
        index,
        top_docs=top_docs,
        min_score=min_score,
        query_token_weights=query_token_weights,
        use_doc_latent_pruning=use_doc_latent_pruning,
        index_candidate_pool=index_candidate_pool,
        query_latent_top_k=query_latent_top_k,
        index_two_phase=False,
        two_phase_pool_size=16_384,
        coarse_topk=8,
        scan_postings=scan_postings,
        use_vectorized=use_vectorized,
        index_parallel_workers=index_parallel_workers,
        index_accum_device=index_accum_device,
        cuda_device=cuda_device,
        gpu_hot_budget_gb=gpu_hot_budget_gb,
    )


def batch_exact_maxsim_via_index(
    queries: Sequence[SparseTokenEmbeddings],
    index: BlockInvertedIndex,
    *,
    top_docs: int = 0,
    min_scores: Sequence[float] | None = None,
    use_doc_latent_pruning: bool = False,
    index_candidate_pool: int = 0,
    query_latent_top_k: int = 0,
    index_two_phase: bool = False,
    two_phase_pool_size: int = 16_384,
    coarse_topk: int = 8,
    use_vectorized: bool = True,
    index_parallel_workers: int = 0,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Score a query batch against one chunk index (shared inverted index)."""
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for qi, query in enumerate(queries):
        ms = 0.0 if min_scores is None else float(min_scores[qi])
        out.append(
            exact_maxsim_via_index(
                query,
                index,
                top_docs=top_docs,
                min_score=ms,
                use_doc_latent_pruning=use_doc_latent_pruning,
                index_candidate_pool=index_candidate_pool,
                query_latent_top_k=query_latent_top_k,
                index_two_phase=index_two_phase,
                two_phase_pool_size=two_phase_pool_size,
                coarse_topk=coarse_topk,
                use_vectorized=use_vectorized,
                index_parallel_workers=index_parallel_workers,
                index_accum_device=index_accum_device,
                cuda_device=cuda_device,
                gpu_hot_budget_gb=gpu_hot_budget_gb,
            )
        )
    return out


def coarse_maxsim_via_index(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    *,
    top_docs: int,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias: pruned pipeline uses the same index math (often with top_docs < |chunk|)."""
    return _maxsim_via_index_impl(
        query,
        index,
        top_docs=top_docs,
        min_score=min_score,
        query_token_weights=query_token_weights,
    )


def query_candidate_doc_ids(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
) -> np.ndarray:
    """Doc indices with at least one query–corpus latent overlap (necessary for score > 0)."""
    if query.n_tokens == 0:
        return np.array([], dtype=np.int64)
    plan = build_query_block_plan(
        query, block_size=index.block_size, n_latents=index.n_latents
    )
    seen: set[int] = set()
    for block_id in plan.active_block_ids:
        q_latents = plan.entries_by_block[block_id]
        for latent in index.iter_block_latents(block_id):
            if latent not in q_latents:
                continue
            for doc_id, _, _ in index.compact.iter_latent_rows(latent):
                seen.add(doc_id)
    if not seen:
        return np.array([], dtype=np.int64)
    return np.fromiter(seen, dtype=np.int64, count=len(seen))


def _query_latent_segments(
    plan: QueryBlockPlan,
    compact: CompactPostings,
    *,
    latent_row_index: np.ndarray | None = None,
) -> list[tuple[list[tuple[int, float]], int, int]]:
    """Pre-resolve (q_entries, slice_start, slice_end) for each query-active latent."""
    segments: list[tuple[list[tuple[int, float]], int, int]] = []
    offsets = compact.offsets
    for block_id in plan.active_block_ids:
        q_latents = plan.entries_by_block[block_id]
        for latent, q_entries in q_latents.items():
            if latent_row_index is not None:
                li = int(latent_row_index[int(latent)])
            else:
                li = compact.latent_index(int(latent))
            if li < 0:
                continue
            s, e = int(offsets[li]), int(offsets[li + 1])
            if e > s:
                segments.append((q_entries, s, e))
    return segments


def _accumulate_maxsim_vectorized(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    plan: QueryBlockPlan,
    per_qt: np.ndarray | None,
    *,
    parallel_workers: int = 0,
    use_fast_accumulate: bool = True,
    doc_total: np.ndarray | None = None,
    doc_to_slot: np.ndarray | None = None,
    pool_candidate_doc_ids: np.ndarray | None = None,
    scan_postings: CompactPostings | None = None,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> None:
    """NumPy/Numba MaxSim accumulate: only query-active latents."""
    compact = scan_postings if scan_postings is not None else index.compact
    if int(index.n_docs) <= 0:
        return

    if use_fast_accumulate:
        from .index_accumulate_fast import accumulate_maxsim_from_plan

        accumulate_maxsim_from_plan(
            plan,
            index,
            per_qt,
            doc_total=doc_total,
            doc_to_slot=doc_to_slot,
            pool_candidate_doc_ids=pool_candidate_doc_ids,
            n_query_tokens=query.n_tokens,
            parallel_workers=parallel_workers,
            index_accum_device=index_accum_device,
            cuda_device=cuda_device,
            gpu_hot_budget_gb=gpu_hot_budget_gb,
            scan_postings=compact,
        )
        return

    segments = _query_latent_segments(plan, compact, latent_row_index=index.latent_row_index)
    if not segments:
        return

    assert per_qt is not None
    doc_idx = compact.doc_idx
    values = compact.values
    n_qt = int(per_qt.shape[0])
    n_docs = int(index.n_docs)
    _accumulate_segment_chunk(segments, doc_idx, values, n_qt, n_docs, out=per_qt)


def _accumulate_segment_chunk(
    segments: list[tuple[list[tuple[int, float]], int, int]],
    doc_idx: np.ndarray,
    values: np.ndarray,
    n_qt: int,
    n_docs: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Process a chunk of latent segments into a local per-qt max buffer."""
    local = out if out is not None else np.zeros((n_qt, n_docs), dtype=np.float32)
    if out is None:
        local.fill(0.0)
    for q_entries, s, e in segments:
        docs = doc_idx[s:e]
        vals = values[s:e]
        if docs.size == 0:
            continue
        out_docs, max_vals = segment_max_by_doc(docs, vals)
        for qt, v_q in q_entries:
            if qt < 0 or qt >= n_qt:
                continue
            scaled = max_vals * np.float32(v_q)
            if out_docs.size == 0:
                continue
            row = local[qt]
            row[out_docs] = np.maximum(row[out_docs], scaled)
    return local


def _accumulate_maxsim_from_index(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    plan: QueryBlockPlan,
    per_query_token: List[Dict[int, float]],
    *,
    top_docs: int = 0,
    min_score: float = 0.0,
    use_doc_latent_pruning: bool = False,
    use_vectorized: bool = True,
    per_qt_array: np.ndarray | None = None,
    parallel_workers: int = 0,
    use_fast_accumulate: bool = True,
    doc_total: np.ndarray | None = None,
    doc_to_slot: np.ndarray | None = None,
    pool_candidate_doc_ids: np.ndarray | None = None,
    scan_postings: CompactPostings | None = None,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> None:
    """Block-first MaxSim: per (query_token, doc) keep max(v_q * val) over doc tokens."""
    if use_vectorized and (per_qt_array is not None or doc_total is not None):
        _accumulate_maxsim_vectorized(
            query,
            index,
            plan,
            per_qt_array,
            parallel_workers=parallel_workers,
            use_fast_accumulate=use_fast_accumulate,
            doc_total=doc_total,
            doc_to_slot=doc_to_slot,
            pool_candidate_doc_ids=pool_candidate_doc_ids,
            scan_postings=scan_postings,
            index_accum_device=index_accum_device,
            cuda_device=cuda_device,
            gpu_hot_budget_gb=gpu_hot_budget_gb,
        )
        return

    scan = scan_postings if scan_postings is not None else index.compact
    if use_doc_latent_pruning:
        ensure_doc_latent_max(index)
        _accumulate_maxsim_pruned(
            query,
            index,
            plan,
            per_query_token,
            top_docs=top_docs,
            min_score=min_score,
        )
        return

    for block_id in plan.active_block_ids:
        q_latents = plan.entries_by_block[block_id]
        for latent, q_entries in q_latents.items():
            for qt, v_q in q_entries:
                acc = per_query_token[qt]
                for doc_id, _tok_id, val in scan.iter_latent_rows(int(latent)):
                    contrib = v_q * val
                    prev = acc.get(doc_id)
                    if prev is None or contrib > prev:
                        acc[doc_id] = contrib


def _latent_doc_max_slices(
    doc_latent_max: CompactPostings,
    latent_weights: Dict[int, float],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Per-query-latent sorted (doc_idx, max_value) views into ``doc_latent_max``."""
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for latent in latent_weights:
        i = doc_latent_max.latent_index(latent)
        if i < 0:
            continue
        s, e = int(doc_latent_max.offsets[i]), int(doc_latent_max.offsets[i + 1])
        if e > s:
            out[latent] = (doc_latent_max.doc_idx[s:e], doc_latent_max.values[s:e])
    return out


def _doc_score_upper_bound_fast(
    doc_id: int,
    latent_weights: Dict[int, float],
    latent_slices: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> float:
    total = 0.0
    for latent, v_sum in latent_weights.items():
        pair = latent_slices.get(latent)
        if pair is None:
            continue
        docs, vals = pair
        pos = int(np.searchsorted(docs, doc_id))
        if pos < int(docs.shape[0]) and int(docs[pos]) == doc_id:
            total += v_sum * float(vals[pos])
    return total


def _accumulate_maxsim_pruned(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    plan: QueryBlockPlan,
    per_query_token: List[Dict[int, float]],
    *,
    top_docs: int,
    min_score: float,
) -> None:
    """Skip docs whose per-doc MaxSim upper bound cannot beat ``min_score`` / top-heap."""
    doc_latent_max = ensure_doc_latent_max(index)

    pruned: set[int] = set()
    doc_ub_cache: Dict[int, float] = {}
    doc_partial: Dict[int, float] = {}
    latent_weights = query_latent_weight_sums(query)
    latent_slices = _latent_doc_max_slices(doc_latent_max, latent_weights)
    threshold = float(min_score)
    min_heap: List[Tuple[float, int]] = []

    use_dynamic_threshold = top_docs > 0

    def _maybe_raise_threshold(doc_id: int, partial: float) -> None:
        nonlocal threshold
        if not use_dynamic_threshold:
            return
        if len(min_heap) < top_docs:
            heapq.heappush(min_heap, (partial, doc_id))
            if len(min_heap) >= top_docs:
                threshold = max(min_score, float(min_heap[0][0]))
            return
        if partial <= min_heap[0][0]:
            return
        heapq.heapreplace(min_heap, (partial, doc_id))
        threshold = max(min_score, float(min_heap[0][0]))

    for block_id in plan.active_block_ids:
        q_latents = plan.entries_by_block[block_id]
        for latent in index.iter_block_latents(block_id):
            q_entries = q_latents.get(latent)
            if not q_entries:
                continue
            for qt, v_q in q_entries:
                acc = per_query_token[qt]
                for doc_id, _tok_id, val in index.compact.iter_latent_rows(latent):
                    if doc_id in pruned:
                        continue
                    if doc_id not in doc_ub_cache:
                        ub = _doc_score_upper_bound_fast(
                            doc_id, latent_weights, latent_slices
                        )
                        doc_ub_cache[doc_id] = ub
                        if ub <= threshold:
                            pruned.add(doc_id)
                            continue
                    contrib = v_q * val
                    prev = acc.get(doc_id)
                    if prev is None or contrib > prev:
                        delta = contrib if prev is None else contrib - prev
                        acc[doc_id] = contrib
                        if use_dynamic_threshold:
                            partial = doc_partial.get(doc_id, 0.0) + delta
                            doc_partial[doc_id] = partial
                            _maybe_raise_threshold(doc_id, partial)


def _maxsim_via_index_impl_unblocked(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    *,
    top_docs: int,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference: iterate every query latent (no block skipping)."""
    per_query_token: List[Dict[int, float]] = [
        {} for _ in range(query.n_tokens)
    ]
    for qt in range(query.n_tokens):
        acc = per_query_token[qt]
        w_q = 1.0 if query_token_weights is None else float(query_token_weights[qt])
        for j in range(query.k):
            latent = int(query.indices[qt, j])
            if latent < 0:
                break
            v_q = float(query.values[qt, j]) * w_q
            for doc_id, _tok_id, val in index.compact.iter_latent_rows(latent):
                contrib = v_q * val
                prev = acc.get(doc_id)
                if prev is None or contrib > prev:
                    acc[doc_id] = contrib
    return _finalize_maxsim_from_qt_dicts(
        per_query_token, top_docs=top_docs, min_score=min_score
    )


def _finalize_maxsim_from_qt_array(
    per_qt: np.ndarray,
    *,
    top_docs: int,
    min_score: float,
    doc_total: np.ndarray | None = None,
    candidate_doc_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum per-query-token maxima (dense rows) and return top_docs."""
    if per_qt.size == 0 and doc_total is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if doc_total is not None:
        total = doc_total
    else:
        total = sum_per_qt_fast(per_qt).astype(np.float64, copy=False)
    if top_docs > 0:
        k = min(int(top_docs), int(total.shape[0]))
        pick = np.argpartition(total, -k)[-k:]
        scores = total[pick].astype(np.float32, copy=False)
        if min_score > 0.0:
            keep = scores > float(min_score)
            pick = pick[keep]
            scores = scores[keep]
        else:
            keep = scores > 0.0
            pick = pick[keep]
            scores = scores[keep]
        if pick.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        pick = pick[order]
        scores = scores[order]
        if candidate_doc_ids is not None:
            return candidate_doc_ids[pick].astype(np.int64, copy=False), scores
        return pick.astype(np.int64, copy=False), scores

    if min_score > 0.0:
        mask = total > float(min_score)
    else:
        mask = total > 0.0
    doc_ids = np.nonzero(mask)[0].astype(np.int64, copy=False)
    if doc_ids.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    scores = total[doc_ids].astype(np.float32, copy=False)
    order = np.argsort(-scores, kind="stable")
    return doc_ids[order], scores[order]


def _finalize_maxsim_from_qt_dicts(
    per_query_token: List[Dict[int, float]],
    *,
    top_docs: int,
    min_score: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum per-query-token maxima into doc scores; return top_docs via heap."""
    doc_totals: Dict[int, float] = {}
    for qt_dict in per_query_token:
        if not qt_dict:
            continue
        for doc_idx, val in qt_dict.items():
            doc_totals[doc_idx] = doc_totals.get(doc_idx, 0.0) + val

    if not doc_totals:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    if min_score > 0.0:
        doc_totals = {d: s for d, s in doc_totals.items() if s > min_score}
        if not doc_totals:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    if top_docs > 0 and len(doc_totals) > top_docs:
        items = heapq.nlargest(top_docs, doc_totals.items(), key=lambda x: x[1])
    else:
        items = sorted(doc_totals.items(), key=lambda x: x[1], reverse=True)
    doc_ids = np.array([d for d, _ in items], dtype=np.int64)
    scores = np.array([s for _, s in items], dtype=np.float32)
    return doc_ids, scores


def _maxsim_via_index_impl(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    *,
    top_docs: int,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
    use_doc_latent_pruning: bool = False,
    index_candidate_pool: int = 0,
    query_latent_top_k: int = 0,
    index_two_phase: bool = False,
    two_phase_pool_size: int = 16_384,
    coarse_topk: int = 8,
    scan_postings: CompactPostings | None = None,
    use_vectorized: bool = True,
    index_parallel_workers: int = 0,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    use_fast_accumulate = use_vectorized
    postings = scan_postings if scan_postings is not None else index.compact
    plan = build_query_block_plan(
        query,
        block_size=index.block_size,
        n_latents=index.n_latents,
        query_token_weights=query_token_weights,
    )
    if query_latent_top_k > 0:
        from .index_pruning import filter_plan_top_latents

        plan = filter_plan_top_latents(
            plan, query, top_k_latents=int(query_latent_top_k)
        )
    if use_vectorized and index.n_docs > 0:
        n_docs = int(index.n_docs)
        use_doc_total = use_fast_accumulate and top_docs > 0
        candidate_doc_ids: np.ndarray | None = None
        doc_to_slot: np.ndarray | None = None
        pool_size = int(index_candidate_pool)
        if pool_size > 0 and pool_size < n_docs:
            from .index_pruning import block_upper_bound_scores, select_candidate_pool

            ub = block_upper_bound_scores(plan, index, query)
            candidate_doc_ids, doc_to_slot = select_candidate_pool(
                ub, pool_size=pool_size, min_score=min_score
            )
            if candidate_doc_ids.size == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            n_docs = int(candidate_doc_ids.shape[0])
        if use_doc_total:
            if index_accum_device in ("cuda", "auto", "hybrid"):
                from .index_accumulate_gpu import (
                    finalize_topk_from_doc_total_torch,
                    finalize_topk_from_qt_scores_gpu,
                    try_accumulate_gpu_from_plan,
                )

                gpu_out = try_accumulate_gpu_from_plan(
                    plan,
                    index,
                    n_qt=query.n_tokens,
                    doc_total=None,
                    index_accum_device=index_accum_device,  # type: ignore[arg-type]
                    cuda_device=cuda_device,
                    gpu_hot_budget_gb=gpu_hot_budget_gb,
                )
                if gpu_out is not None:
                    _tag, accum_gpu = gpu_out
                    if int(accum_gpu.ndim) == 1:
                        return finalize_topk_from_doc_total_torch(
                            accum_gpu,
                            top_docs=top_docs,
                            min_score=min_score,
                        )
                    return finalize_topk_from_qt_scores_gpu(
                        accum_gpu,
                        top_docs=top_docs,
                        min_score=min_score,
                    )

            from .index_accumulate_fast import _get_stamp_state

            if doc_to_slot is not None:
                doc_total = np.zeros(n_docs, dtype=np.float32)
            else:
                _qt, _qg, doc_total, _gen = _get_stamp_state(query.n_tokens, n_docs)
            _accumulate_maxsim_from_index(
                query,
                index,
                plan,
                [],
                top_docs=top_docs,
                min_score=min_score,
                use_doc_latent_pruning=False,
                use_vectorized=True,
                per_qt_array=None,
                parallel_workers=index_parallel_workers,
                use_fast_accumulate=True,
                doc_total=doc_total,
                doc_to_slot=doc_to_slot,
                pool_candidate_doc_ids=candidate_doc_ids,
                scan_postings=postings,
                index_accum_device="cpu",
                cuda_device=cuda_device,
            )
            return _finalize_maxsim_from_qt_array(
                np.zeros((0, 0), dtype=np.float32),
                top_docs=top_docs,
                min_score=min_score,
                doc_total=doc_total,
                candidate_doc_ids=candidate_doc_ids,
            )
        per_qt = np.zeros((query.n_tokens, n_docs), dtype=np.float32)
        _accumulate_maxsim_from_index(
            query,
            index,
            plan,
            [],
            top_docs=top_docs,
            min_score=min_score,
            use_doc_latent_pruning=False,
            use_vectorized=True,
            per_qt_array=per_qt,
            parallel_workers=index_parallel_workers,
            use_fast_accumulate=use_fast_accumulate,
            scan_postings=postings,
        )
        return _finalize_maxsim_from_qt_array(
            per_qt,
            top_docs=top_docs,
            min_score=min_score,
        )

    per_query_token: List[Dict[int, float]] = [
        {} for _ in range(query.n_tokens)
    ]
    _accumulate_maxsim_from_index(
        query,
        index,
        plan,
        per_query_token,
        top_docs=top_docs,
        min_score=min_score,
        use_doc_latent_pruning=use_doc_latent_pruning,
        use_vectorized=False,
    )
    return _finalize_maxsim_from_qt_dicts(
        per_query_token, top_docs=top_docs, min_score=min_score
    )


def maxsim_index_results_equal(
    query: SparseTokenEmbeddings,
    index: BlockInvertedIndex,
    *,
    top_docs: int = 0,
    min_score: float = 0.0,
    atol: float = 1e-5,
) -> bool:
    """True if block-first and unblocked index scoring agree."""
    a = _maxsim_via_index_impl(
        query, index, top_docs=top_docs, min_score=min_score
    )
    b = _maxsim_via_index_impl_unblocked(
        query, index, top_docs=top_docs, min_score=min_score
    )
    if a[0].shape != b[0].shape:
        return False
    if a[0].size == 0:
        return True
    order = np.argsort(a[0])
    bo = np.argsort(b[0])
    return np.array_equal(a[0][order], b[0][bo]) and np.allclose(
        a[1][order], b[1][bo], atol=atol, rtol=0
    )


def batch_coarse_scores(
    queries: Sequence[SparseTokenEmbeddings],
    index: BlockInvertedIndex,
    *,
    top_docs_per_query: int,
    min_scores: Sequence[float] | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [
        coarse_maxsim_via_index(
            q,
            index,
            top_docs=top_docs_per_query,
            min_score=0.0 if min_scores is None else float(min_scores[qi]),
        )
        for qi, q in enumerate(queries)
    ]
