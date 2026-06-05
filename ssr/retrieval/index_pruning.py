"""Index-side pruning helpers (block upper bounds + candidate pools)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np

from .compact_postings import query_latent_weight_sums

# Tuned on synthetic_1m (1M docs): top-100 overlap mean ~0.98, min ~0.96 vs exact MaxSim.
SYNTHETIC_INDEX_FAST_LATENT_TOP_K: int = 832
# Two-phase: coarse top-8 index + pool rerank with full top-32 (tune pool for overlap).
SYNTHETIC_INDEX_TWO_PHASE_POOL: int = 32_768
SYNTHETIC_INDEX_TWO_PHASE_COARSE_TOPK: int = 8
# GPU two-phase defaults (synthetic_1m: ~30ms/query with hybrid coarse + pool cand-merge).
SYNTHETIC_GPU_TWO_PHASE_POOL: int = 5_000
SYNTHETIC_GPU_TWO_PHASE_LATENT_TOP_K: int = 512


@dataclass(frozen=True)
class SyntheticIndexFastSettings:
    """Pruned index scoring preset for the synthetic efficiency benchmark."""

    query_latent_top_k: int = SYNTHETIC_INDEX_FAST_LATENT_TOP_K
    index_candidate_pool: int = 0


def apply_synthetic_index_fast(settings: SyntheticIndexFastSettings | None = None) -> SyntheticIndexFastSettings:
    """Return the active fast-path settings (for logging / tests)."""
    return settings or SyntheticIndexFastSettings()


def ranked_doc_overlap_fraction(
    ref_doc_ids: Sequence[int],
    pruned_doc_ids: Sequence[int],
    k: int,
) -> float:
    """Fraction of exact top-``k`` doc ids that appear in the pruned top-``k`` list."""
    k = int(k)
    if k <= 0:
        return 1.0
    ref = {int(x) for x in ref_doc_ids[:k]}
    if not ref:
        return 1.0
    pr = {int(x) for x in pruned_doc_ids[:k]}
    return len(ref & pr) / float(len(ref))

if TYPE_CHECKING:
    from .inverted_index import BlockInvertedIndex, QueryBlockPlan
    from .sparse_repr import SparseTokenEmbeddings


def build_block_doc_max(
    compact,
    *,
    n_docs: int,
    n_latents: int,
    block_size: int,
    show_progress: bool = False,
) -> np.ndarray:
    """Per (block, doc) max activation over latents in the block.

    Shape ``(n_blocks, n_docs)`` float16. Storage ~ ``2 * n_blocks * n_docs`` bytes
    (e.g. 64 MiB for 1M docs, block_size=512, n_latents=16384).
    """
    from tqdm import tqdm

    from .compact_postings import _collapse_segment_doc_max

    n_blocks = (int(n_latents) + int(block_size) - 1) // int(block_size)
    out = np.zeros((n_blocks, int(n_docs)), dtype=np.float16)
    lids = compact.latent_ids
    offsets = compact.offsets
    n_lat = int(lids.shape[0])
    latent_iter: range | tqdm = range(n_lat)
    if show_progress:
        latent_iter = tqdm(latent_iter, desc="Global index [block_doc_max]", unit="latent")
    for i in latent_iter:
        latent = int(lids[i])
        bid = latent // int(block_size)
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        d_part, v_part = _collapse_segment_doc_max(compact.doc_idx[s:e], compact.values[s:e])
        row = out[bid]
        for j in range(int(d_part.shape[0])):
            d = int(d_part[j])
            v = float(v_part[j])
            prev = float(row[d])
            if v > prev:
                row[d] = np.float16(v)
    return out


def ensure_block_doc_max(
    index: "BlockInvertedIndex",
    *,
    show_progress: bool = False,
) -> np.ndarray:
    bdm = getattr(index, "block_doc_max", None)
    if bdm is not None and bdm.shape == (index.n_blocks, int(index.n_docs)):
        return bdm
    index.block_doc_max = build_block_doc_max(
        index.compact,
        n_docs=int(index.n_docs),
        n_latents=int(index.n_latents),
        block_size=int(index.block_size),
        show_progress=show_progress,
    )
    return index.block_doc_max


def block_upper_bound_scores(
    plan: "QueryBlockPlan",
    index: "BlockInvertedIndex",
    query: "SparseTokenEmbeddings",
) -> np.ndarray:
    """Cheap per-doc upper bound: sum_b W_b * max_{l in b} M(d, l)."""
    bdm = ensure_block_doc_max(index)
    n_docs = int(index.n_docs)
    ub = np.zeros(n_docs, dtype=np.float32)
    weights = query_latent_weight_sums(query)
    for block_id in plan.active_block_ids:
        w_b = 0.0
        for latent in plan.entries_by_block[block_id]:
            w_b += float(weights.get(int(latent), 0.0))
        if w_b <= 0.0:
            continue
        ub += np.float32(w_b) * bdm[int(block_id)].astype(np.float32, copy=False)
    return ub


def filter_plan_top_latents(
    plan: "QueryBlockPlan",
    query: "SparseTokenEmbeddings",
    *,
    top_k_latents: int,
) -> "QueryBlockPlan":
    """Keep only the ``top_k_latents`` highest total-weight latents (lossy)."""
    from .inverted_index import QueryBlockPlan

    if top_k_latents <= 0:
        return plan
    weights = query_latent_weight_sums(query)
    if len(weights) <= top_k_latents:
        return plan
    keep = set(
        sorted(weights.keys(), key=lambda lid: weights[lid], reverse=True)[
            :top_k_latents
        ]
    )
    new_entries: list[dict[int, list]] = [dict() for _ in range(plan.n_blocks)]
    active: list[int] = []
    for block_id in plan.active_block_ids:
        src = plan.entries_by_block[block_id]
        dst: dict[int, list] = {}
        for latent, items in src.items():
            if int(latent) in keep:
                dst[int(latent)] = items
        if dst:
            new_entries[block_id] = dst
            active.append(block_id)
    return QueryBlockPlan(
        n_blocks=plan.n_blocks,
        active_block_ids=tuple(active),
        entries_by_block=tuple(new_entries),
    )


def select_candidate_pool(
    ub: np.ndarray,
    *,
    pool_size: int,
    min_score: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (candidate_doc_ids, doc_to_slot) with ``doc_to_slot[doc]=slot`` or -1."""
    n_docs = int(ub.shape[0])
    pool_size = int(pool_size)
    if pool_size <= 0 or pool_size >= n_docs:
        doc_ids = np.arange(n_docs, dtype=np.int32)
        doc_to_slot = doc_ids.astype(np.int32, copy=False)
        return doc_ids, doc_to_slot

    if min_score > 0.0:
        mask = ub > float(min_score)
        if not np.any(mask):
            return np.array([], dtype=np.int32), np.full(n_docs, -1, dtype=np.int32)
        eligible = np.nonzero(mask)[0]
        if eligible.size <= pool_size:
            doc_ids = eligible.astype(np.int32, copy=False)
            doc_to_slot = np.full(n_docs, -1, dtype=np.int32)
            doc_to_slot[doc_ids] = np.arange(doc_ids.shape[0], dtype=np.int32)
            return doc_ids, doc_to_slot

    k = min(pool_size, n_docs)
    pick = np.argpartition(ub, -k)[-k:]
    order = pick[np.argsort(-ub[pick], kind="stable")]
    doc_ids = order.astype(np.int32, copy=False)
    doc_to_slot = np.full(n_docs, -1, dtype=np.int32)
    doc_to_slot[doc_ids] = np.arange(doc_ids.shape[0], dtype=np.int32)
    return doc_ids, doc_to_slot


def ensure_compact_coarse(
    index: "BlockInvertedIndex",
    *,
    coarse_topk: int | None = None,
    show_progress: bool = False,
) -> "CompactPostings":
    from .compact_postings import CompactPostings, downsample_compact_per_token_topk

    k = int(coarse_topk if coarse_topk is not None else index.coarse_topk)
    coarse = index.compact_coarse
    if coarse is not None and coarse.n_postings > 0:
        return coarse
    index.compact_coarse = downsample_compact_per_token_topk(
        index.compact, topk=k, show_progress=show_progress
    )
    index.coarse_topk = k
    return index.compact_coarse


def build_doc_to_slot_from_candidates(
    candidate_doc_ids: np.ndarray,
    n_docs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map global doc ids to dense pool slots."""
    doc_ids = np.asarray(candidate_doc_ids, dtype=np.int32).reshape(-1)
    n_pool = int(doc_ids.shape[0])
    if n_pool <= 0:
        return doc_ids, np.full(int(n_docs), -1, dtype=np.int32)
    doc_to_slot = np.full(int(n_docs), -1, dtype=np.int32)
    doc_to_slot[doc_ids] = np.arange(n_pool, dtype=np.int32)
    return doc_ids, doc_to_slot


def two_phase_maxsim_via_index(
    query: "SparseTokenEmbeddings",
    index: "BlockInvertedIndex",
    *,
    top_docs: int,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
    pool_size: int = 16_384,
    coarse_topk: int = 8,
    query_latent_top_k: int = 0,
    use_vectorized: bool = True,
    index_parallel_workers: int = 0,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse MaxSim (high sparsity) → candidate pool → exact MaxSim on pool."""
    from .inverted_index import _maxsim_via_index_impl
    from .sparse_repr import prune_sparse_rows

    n_docs = int(index.n_docs)
    pool_size = int(pool_size)
    if pool_size <= 0 or pool_size >= n_docs:
        return _maxsim_via_index_impl(
            query,
            index,
            top_docs=top_docs,
            min_score=min_score,
            query_token_weights=query_token_weights,
            query_latent_top_k=query_latent_top_k,
            use_vectorized=use_vectorized,
            index_parallel_workers=index_parallel_workers,
            index_accum_device=index_accum_device,
            cuda_device=cuda_device,
            gpu_hot_budget_gb=gpu_hot_budget_gb,
        )

    coarse = ensure_compact_coarse(index, coarse_topk=coarse_topk)
    coarse_q = prune_sparse_rows(query, int(coarse_topk))
    coarse_ids, _ = _maxsim_via_index_impl(
        coarse_q,
        index,
        top_docs=pool_size,
        min_score=min_score,
        query_token_weights=query_token_weights,
        use_vectorized=use_vectorized,
        index_parallel_workers=index_parallel_workers,
        index_accum_device=index_accum_device,
        cuda_device=cuda_device,
        gpu_hot_budget_gb=gpu_hot_budget_gb,
        scan_postings=coarse,
    )
    if coarse_ids.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    candidate_doc_ids, doc_to_slot = build_doc_to_slot_from_candidates(
        coarse_ids, n_docs
    )
    n_pool = int(candidate_doc_ids.shape[0])
    from .index_accumulate_fast import _get_stamp_state

    if use_vectorized and top_docs > 0:
        _qt, _qg, doc_total, _gen = _get_stamp_state(query.n_tokens, n_pool)
    else:
        doc_total = None

    plan = None
    from .inverted_index import build_query_block_plan

    plan = build_query_block_plan(
        query,
        block_size=index.block_size,
        n_latents=index.n_latents,
        query_token_weights=query_token_weights,
    )
    phase2_latent_top_k = int(query_latent_top_k)
    accum = str(index_accum_device).lower()
    if phase2_latent_top_k <= 0 and accum in ("hybrid", "cuda", "auto"):
        phase2_latent_top_k = SYNTHETIC_GPU_TWO_PHASE_LATENT_TOP_K
    if phase2_latent_top_k > 0:
        plan = filter_plan_top_latents(
            plan, query, top_k_latents=phase2_latent_top_k
        )

    from .inverted_index import _accumulate_maxsim_from_index, _finalize_maxsim_from_qt_array

    if doc_total is None:
        per_qt = np.zeros((query.n_tokens, n_pool), dtype=np.float32)
        _accumulate_maxsim_from_index(
            query,
            index,
            plan,
            [],
            top_docs=top_docs,
            min_score=min_score,
            use_vectorized=True,
            per_qt_array=per_qt,
            parallel_workers=index_parallel_workers,
            use_fast_accumulate=True,
            scan_postings=index.compact,
        )
        return _finalize_maxsim_from_qt_array(
            per_qt,
            top_docs=top_docs,
            min_score=min_score,
            candidate_doc_ids=candidate_doc_ids,
        )

    _accumulate_maxsim_from_index(
        query,
        index,
        plan,
        [],
        top_docs=top_docs,
        min_score=min_score,
        use_vectorized=True,
        per_qt_array=None,
        parallel_workers=index_parallel_workers,
        use_fast_accumulate=True,
        doc_total=doc_total,
        doc_to_slot=doc_to_slot,
        pool_candidate_doc_ids=candidate_doc_ids,
        scan_postings=index.compact,
        index_accum_device=index_accum_device,
        cuda_device=cuda_device,
        gpu_hot_budget_gb=gpu_hot_budget_gb,
    )
    return _finalize_maxsim_from_qt_array(
        np.zeros((0, 0), dtype=np.float32),
        top_docs=top_docs,
        min_score=min_score,
        doc_total=doc_total,
        candidate_doc_ids=candidate_doc_ids,
    )


