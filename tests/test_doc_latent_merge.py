"""Incremental ``merge_doc_latent_max`` matches full ``build_doc_latent_max``."""

from __future__ import annotations

import numpy as np

from ssr.retrieval.compact_postings import (
    build_doc_latent_max,
    compact_from_latent_major_coo,
    merge_compact,
    merge_doc_latent_max,
)


def _shard_compact(*, doc_start: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 4, size=n, dtype=np.int32)
    docs = (doc_start + rng.integers(0, 6, size=n)).astype(np.int32)
    toks = np.zeros(n, dtype=np.int16)
    vals = rng.random(n, dtype=np.float32)
    order = np.lexsort((toks, docs, cols))
    return compact_from_latent_major_coo(
        cols[order], docs[order], toks[order], vals[order]
    )


def test_merge_doc_latent_max_matches_full_build():
    a = _shard_compact(doc_start=0, n=80, seed=1)
    b = _shard_compact(doc_start=10_000, n=60, seed=2)
    merged_postings = merge_compact(a, b)
    inc = merge_doc_latent_max(
        build_doc_latent_max(a), build_doc_latent_max(b)
    )
    full = build_doc_latent_max(merged_postings)
    assert inc.n_postings == full.n_postings
    np.testing.assert_array_equal(inc.latent_ids, full.latent_ids)
    np.testing.assert_array_equal(inc.doc_idx, full.doc_idx)
    np.testing.assert_allclose(inc.values, full.values)
