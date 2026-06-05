"""Global corpus-wide inverted index (build all shards, then retrieve)."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .compact_postings import (
    CompactPostings,
    compact_from_latent_major_coo,
    compute_latent_max,
    merge_compact,
    remap_compact,
)
from .global_index_cache import (
    GlobalIndexBuildStats,
    cache_artifact_path,
    default_cache_dir,
    load_global_index_cache,
    save_global_index_cache,
)
from .inverted_index import BlockInvertedIndex
from .sparse_repr import SparseTokenEmbeddings
from .sparse_tensors import SparseEmbeddingBank

logger = logging.getLogger(__name__)

ReorderMode = Literal["none", "frequency", "cooc"]


@dataclass
class GlobalIndexBuildState:
    """Incremental CPU-side postings merge state (shared by bank and streaming builds)."""

    n_latents: int
    block_size: int = 512
    reorder_mode: ReorderMode = "frequency"
    cooc_sample_rate: float = 0.01
    coarse_topk: int = 8
    compact: CompactPostings = field(default_factory=CompactPostings.empty)
    compact_coarse: CompactPostings = field(default_factory=CompactPostings.empty)
    latent_freq: Counter[int] = field(default_factory=Counter)
    pair_counts: Counter[Tuple[int, int]] | None = None
    n_ingest_batches: int = 0

    def __post_init__(self) -> None:
        if self.reorder_mode == "cooc" and self.pair_counts is None:
            self.pair_counts = Counter()

    def ingest_coo(
        self,
        coo: torch.Tensor,
        *,
        global_doc_start: int,
        tokens_per_doc: int,
    ) -> None:
        shard_compact, shard_coarse = _shard_compacts_from_coo(
            coo,
            global_doc_start=int(global_doc_start),
            tokens_per_doc=int(tokens_per_doc),
            latent_freq=self.latent_freq,
            pair_counts=self.pair_counts,
            cooc_sample_rate=self.cooc_sample_rate,
            coarse_topk=self.coarse_topk,
        )
        self.compact = merge_compact(self.compact, shard_compact)
        self.compact_coarse = merge_compact(self.compact_coarse, shard_coarse)
        self.n_ingest_batches += 1

    def finalize(
        self,
        *,
        n_docs: int,
        show_progress: bool = True,
    ) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats]:
        return finalize_global_index_from_state(
            self,
            n_docs=n_docs,
            show_progress=show_progress,
        )


def finalize_global_index_from_state(
    state: GlobalIndexBuildState,
    *,
    n_docs: int,
    show_progress: bool = True,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats]:
    latent_remap = compute_latent_block_remap(
        n_latents=state.n_latents,
        block_size=state.block_size,
        latent_freq=state.latent_freq,
        pair_counts=state.pair_counts,
        mode=state.reorder_mode,
        show_progress=show_progress,
    )
    compact = state.compact
    compact_coarse = state.compact_coarse
    if state.reorder_mode != "none":
        if show_progress:
            with tqdm(total=2, desc="Global index [remap postings]") as pbar:
                compact = remap_compact(compact, latent_remap)
                pbar.update(1)
                compact_coarse = remap_compact(compact_coarse, latent_remap)
                pbar.update(1)
        else:
            compact = remap_compact(compact, latent_remap)
            compact_coarse = remap_compact(compact_coarse, latent_remap)

    index = BlockInvertedIndex(n_latents=state.n_latents, block_size=state.block_size)
    index.n_docs = int(n_docs)
    index.compact = compact
    index.latent_max = compute_latent_max(compact, state.n_latents)
    index.finalize_block_latents()
    index.coarse_topk = int(state.coarse_topk)
    index.compact_coarse = compact_coarse

    stats = GlobalIndexBuildStats(
        n_shards_ingested=int(state.n_ingest_batches),
        n_postings=compact.n_postings,
        n_latents_active=len(state.latent_freq),
        reorder_mode=state.reorder_mode,
    )
    return index, latent_remap, stats


def remap_sparse_embeddings(
    sparse: SparseTokenEmbeddings,
    latent_remap: np.ndarray,
) -> SparseTokenEmbeddings:
    """Apply corpus latent permutation to query/document sparse rows."""
    out = sparse.indices.copy()
    for t in range(out.shape[0]):
        for j in range(out.shape[1]):
            c = int(out[t, j])
            if c < 0:
                break
            out[t, j] = int(latent_remap[c])
    return SparseTokenEmbeddings(
        indices=out,
        values=sparse.values.copy(),
        n_latents=sparse.n_latents,
    )


def _filter_topk_per_coo_row(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep top-``topk`` values per COO row (doc slot × token slot)."""
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    row_change = np.concatenate([[0], np.where(np.diff(rows) != 0)[0] + 1, [len(rows)]])
    keep = np.zeros(len(rows), dtype=bool)
    for start, end in zip(row_change[:-1], row_change[1:]):
        n = end - start
        if n <= topk:
            keep[start:end] = True
            continue
        pick = np.argpartition(vals[start:end], -topk)[-topk:] + start
        keep[pick] = True
    return rows[keep], cols[keep], vals[keep]


def _compact_from_coo_arrays(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    *,
    global_doc_start: int,
    tokens_per_doc: int,
    latent_freq: Counter[int],
) -> CompactPostings:
    doc_idx = (global_doc_start + rows // tokens_per_doc).astype(np.int32, copy=False)
    token_idx = (rows % tokens_per_doc).astype(np.int16, copy=False)
    order = np.argsort(cols, kind="mergesort")
    cols = cols[order]
    doc_idx = doc_idx[order]
    token_idx = token_idx[order]
    vals = vals[order]
    latent_change = np.concatenate([[0], np.where(np.diff(cols) != 0)[0] + 1, [len(cols)]])
    for start, end in zip(latent_change[:-1], latent_change[1:]):
        latent_freq[int(cols[start])] += end - start
    return compact_from_latent_major_coo(cols, doc_idx, token_idx, vals)


def _shard_compacts_from_coo(
    coo: torch.Tensor,
    *,
    global_doc_start: int,
    tokens_per_doc: int,
    latent_freq: Counter[int],
    pair_counts: Counter[Tuple[int, int]] | None,
    cooc_sample_rate: float,
    coarse_topk: int = 0,
) -> tuple[CompactPostings, CompactPostings]:
    """Full compact plus optional coarse (per-token top-``coarse_topk``) compact."""
    coo = coo.coalesce()
    rows = coo.indices()[0].cpu().numpy()
    cols = coo.indices()[1].cpu().numpy()
    vals = coo.values().cpu().numpy().astype(np.float32, copy=False)

    pos = vals > 0.0
    if not np.any(pos):
        empty = CompactPostings.empty()
        return empty, empty
    rows = rows[pos]
    cols = cols[pos]
    vals = vals[pos]

    if pair_counts is not None:
        _accumulate_cooc_pairs(
            rows,
            cols,
            pair_counts,
            cooc_sample_rate=cooc_sample_rate,
        )

    full = _compact_from_coo_arrays(
        rows,
        cols,
        vals,
        global_doc_start=global_doc_start,
        tokens_per_doc=tokens_per_doc,
        latent_freq=latent_freq,
    )
    if coarse_topk <= 0:
        return full, CompactPostings.empty()

    rows_c, cols_c, vals_c = _filter_topk_per_coo_row(rows, cols, vals, int(coarse_topk))
    if rows_c.size == 0:
        return full, CompactPostings.empty()
    coarse_freq: Counter[int] = Counter()
    coarse = _compact_from_coo_arrays(
        rows_c,
        cols_c,
        vals_c,
        global_doc_start=global_doc_start,
        tokens_per_doc=tokens_per_doc,
        latent_freq=coarse_freq,
    )
    return full, coarse


def _shard_compact_from_coo(
    coo: torch.Tensor,
    *,
    global_doc_start: int,
    tokens_per_doc: int,
    latent_freq: Counter[int],
    pair_counts: Counter[Tuple[int, int]] | None,
    cooc_sample_rate: float,
) -> CompactPostings:
    """Build one shard's compact postings (latent-major, vectorized)."""
    full, _ = _shard_compacts_from_coo(
        coo,
        global_doc_start=global_doc_start,
        tokens_per_doc=tokens_per_doc,
        latent_freq=latent_freq,
        pair_counts=pair_counts,
        cooc_sample_rate=cooc_sample_rate,
        coarse_topk=0,
    )
    return full


def _accumulate_cooc_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    pair_counts: Counter[Tuple[int, int]],
    *,
    cooc_sample_rate: float,
) -> None:
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    cols = cols[order]
    row_change = np.concatenate([[0], np.where(np.diff(rows) != 0)[0] + 1, [len(rows)]])
    rng = np.random.default_rng(0)
    for start, end in zip(row_change[:-1], row_change[1:]):
        row_cols = cols[start:end]
        if row_cols.size <= 1:
            continue
        if cooc_sample_rate < 1.0 and rng.random() >= cooc_sample_rate:
            continue
        uniq = np.unique(row_cols)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = int(uniq[i]), int(uniq[j])
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] += 1


def compute_latent_block_remap(
    *,
    n_latents: int,
    block_size: int,
    latent_freq: Counter[int],
    pair_counts: Counter[Tuple[int, int]] | None,
    mode: ReorderMode,
    show_progress: bool = True,
) -> np.ndarray:
    """Map original latent id -> reordered id (similar/co-active latents share a block)."""
    if mode == "none":
        return np.arange(n_latents, dtype=np.int32)

    active = sorted(latent_freq.keys(), key=lambda l: -latent_freq[l])
    if not active:
        return np.arange(n_latents, dtype=np.int32)

    remap = np.full(n_latents, -1, dtype=np.int32)
    new_id = 0
    unassigned = set(active)

    def cooc_score(candidate: int, block: List[int]) -> int:
        if not pair_counts or not block:
            return latent_freq.get(candidate, 0)
        return sum(
            pair_counts.get(
                (candidate, b) if candidate < b else (b, candidate),
                0,
            )
            for b in block
        )

    n_blocks_est = max(1, (len(active) + block_size - 1) // block_size)
    block_iter = tqdm(
        total=n_blocks_est,
        desc="Global index [reorder latents]",
        disable=not show_progress,
        unit="block",
    )
    try:
        while unassigned and new_id < n_latents:
            seed = max(unassigned, key=lambda l: latent_freq.get(l, 0))
            block: List[int] = [seed]
            unassigned.remove(seed)
            while len(block) < block_size and unassigned:
                if mode == "cooc" and pair_counts:
                    nxt = max(unassigned, key=lambda l: cooc_score(l, block))
                else:
                    nxt = max(unassigned, key=lambda l: latent_freq.get(l, 0))
                block.append(nxt)
                unassigned.remove(nxt)
            for old in block:
                remap[old] = new_id
                new_id += 1
            block_iter.update(1)
    finally:
        block_iter.close()

    for old in range(n_latents):
        if remap[old] < 0:
            remap[old] = old
    return remap


def build_global_index_from_bank(
    bank: SparseEmbeddingBank,
    *,
    block_size: int = 512,
    reorder_mode: ReorderMode = "frequency",
    cooc_sample_rate: float = 0.01,
    show_progress: bool = True,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats]:
    """Phase 1: ingest every corpus shard into one global inverted index."""
    state = GlobalIndexBuildState(
        n_latents=bank.n_latents,
        block_size=block_size,
        reorder_mode=reorder_mode,
        cooc_sample_rate=cooc_sample_rate,
    )
    shard_iter = bank.iter_corpus_shards()
    if show_progress:
        shard_iter = tqdm(
            shard_iter,
            total=bank.n_corpus_shards,
            desc="Global index [ingest shards]",
        )
    for coo, global_start, _n_docs in shard_iter:
        state.ingest_coo(
            coo,
            global_doc_start=int(global_start),
            tokens_per_doc=bank.doc_tokens,
        )
    return state.finalize(n_docs=bank.n_docs, show_progress=show_progress)


def load_or_build_global_index_from_bank(
    bank: SparseEmbeddingBank,
    *,
    bank_dir: Path,
    block_size: int = 512,
    reorder_mode: ReorderMode = "frequency",
    cooc_sample_rate: float = 0.01,
    index_cache_dir: Path | None = None,
    force_rebuild_index: bool = False,
    save_index_cache: bool = True,
    show_progress: bool = True,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats, bool]:
    """Return ``(index, latent_remap, stats, built_fresh)``."""
    cache_dir = index_cache_dir or default_cache_dir(bank_dir)
    cache_path = cache_artifact_path(
        cache_dir, block_size=block_size, reorder_mode=reorder_mode
    )

    if not force_rebuild_index:
        loaded = load_global_index_cache(
            cache_path,
            bank_dir=bank_dir,
            block_size=block_size,
            reorder_mode=reorder_mode,
            show_progress=show_progress,
        )
        if loaded is not None:
            return (*loaded, False)

    index, latent_remap, stats = build_global_index_from_bank(
        bank,
        block_size=block_size,
        reorder_mode=reorder_mode,
        cooc_sample_rate=cooc_sample_rate,
        show_progress=show_progress,
    )
    if save_index_cache:
        save_global_index_cache(
            cache_path,
            index=index,
            latent_remap=latent_remap,
            stats=stats,
            bank_dir=bank_dir,
            block_size=block_size,
            reorder_mode=reorder_mode,
            show_progress=show_progress,
        )
    return index, latent_remap, stats, True


def persist_global_index_cache(
    *,
    path: Path,
    index: BlockInvertedIndex,
    latent_remap: np.ndarray,
    stats: GlobalIndexBuildStats,
    bank_dir: Path,
    block_size: int,
    reorder_mode: str,
    show_progress: bool = True,
) -> None:
    save_global_index_cache(
        path,
        index=index,
        latent_remap=latent_remap,
        stats=stats,
        bank_dir=bank_dir,
        block_size=block_size,
        reorder_mode=reorder_mode,
        show_progress=show_progress,
    )
