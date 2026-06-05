"""Tests for columnar compact inverted index."""

from __future__ import annotations

import numpy as np

from ssr.retrieval.compact_postings import (
    CompactPostings,
    compact_from_latent_major_coo,
    merge_compact,
    remap_compact,
)
from ssr.retrieval.inverted_index import (
    BlockInvertedIndex,
    build_chunk_index_from_flat_coo,
    exact_maxsim_via_index,
    maxsim_index_results_equal,
    query_candidate_doc_ids,
)
from ssr.retrieval.sparse_repr import SparseTokenEmbeddings


def test_merge_and_remap_roundtrip():
    a = compact_from_latent_major_coo(
        np.array([1, 1, 2], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int16),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    b = compact_from_latent_major_coo(
        np.array([1, 3], dtype=np.int32),
        np.array([2, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int16),
        np.array([4.0, 5.0], dtype=np.float32),
    )
    m = merge_compact(a, b)
    assert m.n_postings == 5
    assert m.latent_index(1) >= 0
    remap = np.arange(4, dtype=np.int32)
    remap[1], remap[2] = 2, 1
    r = remap_compact(m, remap)
    assert r.n_postings == 5


def test_chunk_index_maxsim_matches_block_first():
    n_latents = 128
    n_docs = 8
    tokens = 4
    rows, cols, vals = [], [], []
    for d in range(n_docs):
        for t in range(tokens):
            row = d * tokens + t
            rows.extend([row, row])
            cols.extend([t * 3, t * 3 + 1])
            vals.extend([1.0, 0.5])
    import torch

    coo = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.int64),
        torch.tensor(vals, dtype=torch.float32),
        (n_docs * tokens, n_latents),
    ).coalesce()
    index = build_chunk_index_from_flat_coo(
        coo, n_docs=n_docs, tokens_per_doc=tokens, n_latents=n_latents, block_size=32
    )
    assert index.compact.n_postings > 0
    assert index.memory_bytes_estimate() < index.compact.n_postings * 40

    q = SparseTokenEmbeddings(
        indices=np.array([[0, 3, -1], [1, 4, -1]], dtype=np.int64),
        values=np.array([[1.0, 0.5, 0], [0.8, 0.2, 0]], dtype=np.float32),
        n_latents=n_latents,
    )
    assert maxsim_index_results_equal(q, index)
    cand = query_candidate_doc_ids(q, index)
    assert cand.size > 0
    doc_ids, scores = exact_maxsim_via_index(q, index, top_docs=3)
    assert doc_ids.size > 0
    assert np.all(scores > 0)
