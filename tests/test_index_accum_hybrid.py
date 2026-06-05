"""Hybrid GPU hot-latent + CPU cold accumulate must match CPU Numba."""

from __future__ import annotations

import numpy as np
import torch

from ssr.retrieval.inverted_index import build_chunk_index, exact_maxsim_via_index
from ssr.retrieval.sparse_repr import SparseTokenEmbeddings


def _tiny_corpus() -> list[SparseTokenEmbeddings]:
    return [
        SparseTokenEmbeddings(
            indices=np.array([[0, 1, -1], [0, 2, -1]], dtype=np.int64),
            values=np.array([[1.0, 0.5, 0.0], [0.8, 0.3, 0.0]], dtype=np.float32),
            n_latents=8,
        ),
        SparseTokenEmbeddings(
            indices=np.array([[1, 3, -1], [2, -1, -1]], dtype=np.int64),
            values=np.array([[0.9, 0.4, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32),
            n_latents=8,
        ),
    ]


def test_packed_work_qt_aligns_with_work_seg():
    from ssr.retrieval.index_accumulate_fast import build_maxsim_work_from_plan
    from ssr.retrieval.inverted_index import build_query_block_plan

    index = build_chunk_index(_tiny_corpus(), n_latents=8, block_size=4)
    query = _tiny_corpus()[0]
    plan = build_query_block_plan(
        query, block_size=index.block_size, n_latents=index.n_latents
    )
    work = build_maxsim_work_from_plan(plan, index, query.n_tokens)
    assert work is not None
    work_seg, work_vq, work_qt_item = work[6], work[7], work[8]
    assert work_seg.shape[0] == work_qt_item.shape[0] == work_vq.shape[0]


def test_hybrid_matches_cpu_topk():
    if not torch.cuda.is_available():
        return
    index = build_chunk_index(_tiny_corpus(), n_latents=8, block_size=4)
    query = _tiny_corpus()[0]
    ref = exact_maxsim_via_index(
        query, index, top_docs=2, use_vectorized=True, index_accum_device="cpu"
    )
    hybrid = exact_maxsim_via_index(
        query,
        index,
        top_docs=2,
        use_vectorized=True,
        index_accum_device="hybrid",
        gpu_hot_budget_gb=1.0,
    )
    np.testing.assert_array_equal(np.sort(ref[0]), np.sort(hybrid[0]))
    np.testing.assert_allclose(
        np.sort(ref[1])[::-1], np.sort(hybrid[1])[::-1], rtol=0, atol=1e-4
    )
