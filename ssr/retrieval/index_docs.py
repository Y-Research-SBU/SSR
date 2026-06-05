"""Materialize per-document sparse embeddings from a global compact index."""

from __future__ import annotations

import numpy as np

from .inverted_index import BlockInvertedIndex
from .sparse_repr import SparseTokenEmbeddings


def materialize_sparse_docs_from_index(
    index: BlockInvertedIndex,
    global_doc_ids: np.ndarray,
    *,
    doc_tokens: int,
    topk: int,
) -> list[SparseTokenEmbeddings]:
    """Rebuild ``SparseTokenEmbeddings`` for ``doc_ids`` using index postings only.

    Postings use the same latent ids as the loaded global index (already remapped
    when ``latent_reorder`` was applied at build time). No corpus shards required.
    """
    if global_doc_ids.size == 0:
        return []

    ids = np.asarray(global_doc_ids, dtype=np.int32).reshape(-1)
    n_docs = int(ids.shape[0])
    n_latents = index.n_latents
    compact = index.compact

    indices = np.full((n_docs, doc_tokens, topk), -1, dtype=np.int32)
    values = np.zeros((n_docs, doc_tokens, topk), dtype=np.float32)
    counts = np.zeros((n_docs, doc_tokens), dtype=np.int32)

    wanted_sorted = np.sort(ids)
    pos_of = {int(d): i for i, d in enumerate(ids)}

    for li in range(int(compact.latent_ids.shape[0])):
        s, e = int(compact.offsets[li]), int(compact.offsets[li + 1])
        if e <= s:
            continue
        latent = int(compact.latent_ids[li])
        docs = compact.doc_idx[s:e]
        toks = compact.token_idx[s:e]
        vals = compact.values[s:e]

        wi = 0
        j = 0
        seg_len = e - s
        n_w = int(wanted_sorted.shape[0])
        while j < seg_len and wi < n_w:
            d = int(docs[j])
            w = int(wanted_sorted[wi])
            if d < w:
                j += 1
            elif d > w:
                wi += 1
            else:
                di = pos_of[d]
                t = int(toks[j])
                if 0 <= t < doc_tokens:
                    slot = int(counts[di, t])
                    if slot < topk:
                        indices[di, t, slot] = latent
                        values[di, t, slot] = float(vals[j])
                        counts[di, t] += 1
                    elif float(vals[j]) > float(np.min(values[di, t])):
                        # Rare: more than topk entries on one token; keep largest.
                        min_slot = int(np.argmin(values[di, t]))
                        if float(vals[j]) > float(values[di, t, min_slot]):
                            indices[di, t, min_slot] = latent
                            values[di, t, min_slot] = float(vals[j])
                j += 1

    out: list[SparseTokenEmbeddings] = []
    for di in range(n_docs):
        out.append(
            SparseTokenEmbeddings(
                indices=indices[di],
                values=values[di],
                n_latents=n_latents,
            )
        )
    return out
