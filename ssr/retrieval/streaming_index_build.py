"""End-to-end index build: GPU encode → CPU streaming inverted-index merge → on-disk cache."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .doc_id_map import DocIdMap
from .global_index import GlobalIndexBuildState, ReorderMode
from .global_index_cache import (
    GlobalIndexBuildStats,
    cache_artifact_path,
    corpus_fingerprint,
    save_global_index_cache,
)
from .inverted_index import BlockInvertedIndex
from .mteb_io import iter_corpus_jsonl_batches
from .sparse_repr import SparseTokenEmbeddings, batch_dense_to_sparse
from .sparse_tensors import sparse_list_to_flat_coo

logger = logging.getLogger(__name__)


def encode_corpus_batch(
    model,
    texts: Sequence[str],
    *,
    n_latents: int,
    topk: int,
    device: str,
    cls_encoder=None,
) -> list[SparseTokenEmbeddings]:
    """Run SSR document encoding on GPU; return CPU sparse rows."""
    if cls_encoder is not None:
        return cls_encoder.encode_with_token_sae(
            model,
            texts,
            is_query=False,
            token_n_latents=n_latents,
            token_topk=topk,
        )
    with torch.inference_mode():
        embs = model.encode(
            list(texts),
            batch_size=1,
            is_query=False,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    if not isinstance(embs, list):
        embs = [embs]
    return batch_dense_to_sparse(embs, n_latents=n_latents, topk=topk)


def build_global_index_streaming(
    model,
    corpus_batches: Iterator[tuple[Sequence[str], Sequence[str]]],
    *,
    doc_tokens: int,
    n_latents: int,
    topk: int,
    encode_device: str,
    doc_id_map: DocIdMap | None = None,
    block_size: int = 512,
    reorder_mode: ReorderMode = "frequency",
    cooc_sample_rate: float = 0.01,
    coarse_topk: int = 8,
    show_progress: bool = True,
    total_docs: int | None = None,
    empty_cache_every: int = 0,
    cls_encoder=None,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats, DocIdMap]:
    """Encode corpus batches on GPU and merge postings on CPU.

    ``corpus_batches`` yields ``(external_doc_ids, texts)`` chunks. String ids are
    mapped to contiguous int doc indices via ``DocIdMap``.
    """
    id_map = doc_id_map or DocIdMap()
    state = GlobalIndexBuildState(
        n_latents=int(n_latents),
        block_size=int(block_size),
        reorder_mode=reorder_mode,
        cooc_sample_rate=cooc_sample_rate,
        coarse_topk=int(coarse_topk),
    )

    pbar = (
        tqdm(total=total_docs, desc="E2E index [encode+ingest]", unit="doc")
        if show_progress and total_docs and total_docs > 0
        else None
    )

    n_batches = 0
    for external_ids, texts in corpus_batches:
        if not texts:
            continue
        n_batches += 1
        sparse_docs = encode_corpus_batch(
            model,
            texts,
            n_latents=(
                int(n_latents) - int(cls_encoder.n_latents)
                if cls_encoder is not None
                else int(n_latents)
            ),
            topk=topk,
            device=encode_device,
            cls_encoder=cls_encoder,
        )
        del texts

        _gids, global_start = id_map.global_starts_for_batch(external_ids)
        coo = sparse_list_to_flat_coo(
            sparse_docs,
            doc_tokens=int(doc_tokens),
        )
        del sparse_docs

        state.ingest_coo(
            coo,
            global_doc_start=int(global_start),
            tokens_per_doc=int(doc_tokens),
        )
        del coo

        if pbar is not None:
            pbar.update(len(external_ids))
        if empty_cache_every > 0 and n_batches % empty_cache_every == 0:
            if str(encode_device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    if pbar is not None:
        pbar.close()

    if id_map.n_docs == 0:
        raise ValueError("No documents ingested into index")

    index, latent_remap, stats = state.finalize(
        n_docs=id_map.n_docs,
        show_progress=show_progress,
    )
    return index, latent_remap, stats, id_map


def build_mteb_corpus_index_e2e(
    model,
    *,
    corpus_path: Path,
    model_path: Path,
    index_cache_dir: Path,
    encode_device: str,
    block_size: int = 512,
    reorder_mode: ReorderMode = "frequency",
    cooc_sample_rate: float = 0.01,
    encode_batch_size: int = 8,
    max_docs: int = 0,
    show_progress: bool = True,
    empty_cache_every: int = 4,
    cls_encoder=None,
    cls_sae_path: Path | None = None,
    cls_topk: int | None = None,
) -> tuple[Path, DocIdMap, GlobalIndexBuildStats]:
    """Stream ``corpus.jsonl`` → encode on GPU → CPU index → save cache + doc_id_map."""
    doc_tokens = int(getattr(model, "document_length", None) or 180)
    n_latents = int(model.sae_module.n_latents)
    topk = int(model.sae_module.topk)
    if cls_encoder is not None:
        doc_tokens += 1
        n_latents += int(cls_encoder.n_latents)

    total_docs = None
    if max_docs <= 0:
        from .mteb_io import count_corpus_jsonl

        total_docs = count_corpus_jsonl(corpus_path)

    batches = iter_corpus_jsonl_batches(
        corpus_path,
        batch_size=int(encode_batch_size),
        max_docs=max_docs if max_docs > 0 else None,
    )
    effective_total = max_docs if max_docs > 0 else total_docs

    index, latent_remap, stats, id_map = build_global_index_streaming(
        model,
        batches,
        doc_tokens=doc_tokens,
        n_latents=n_latents,
        topk=topk,
        encode_device=encode_device,
        block_size=block_size,
        reorder_mode=reorder_mode,
        cooc_sample_rate=cooc_sample_rate,
        show_progress=show_progress,
        total_docs=effective_total,
        empty_cache_every=empty_cache_every,
        cls_encoder=cls_encoder,
    )

    slug_root = corpus_path.parent
    cache_dir = index_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact = cache_artifact_path(
        cache_dir,
        block_size=block_size,
        reorder_mode=reorder_mode,
    )
    fp = corpus_fingerprint(
        corpus_path=corpus_path,
        model_path=model_path,
        n_docs=id_map.n_docs,
        doc_tokens=doc_tokens,
        n_latents=n_latents,
        topk=topk,
        cls_sae_path=cls_sae_path,
        cls_topk=cls_topk,
    )
    save_global_index_cache(
        artifact,
        index=index,
        latent_remap=latent_remap,
        stats=stats,
        bank_dir=slug_root,
        block_size=block_size,
        reorder_mode=reorder_mode,
        corpus_fingerprint=fp,
        show_progress=show_progress,
    )
    id_map_path = artifact / "doc_id_map.json"
    id_map.save(id_map_path)
    logger.info(
        "Saved e2e index: %s (%d docs, %d postings), doc_id_map=%s",
        artifact,
        id_map.n_docs,
        stats.n_postings,
        id_map_path,
    )
    return artifact, id_map, stats


def load_mteb_e2e_index(
    *,
    index_cache_dir: Path,
    corpus_path: Path,
    model_path: Path,
    doc_tokens: int,
    n_latents: int,
    topk: int,
    block_size: int = 512,
    reorder_mode: str = "frequency",
    show_progress: bool = True,
    cls_sae_path: Path | None = None,
    cls_topk: int | None = None,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats, DocIdMap] | None:
    """Load index + doc id map written by :func:`build_mteb_corpus_index_e2e`."""
    from .global_index_cache import load_global_index_cache

    cache_dir = index_cache_dir.resolve()
    artifact = cache_artifact_path(
        cache_dir,
        block_size=block_size,
        reorder_mode=reorder_mode,
    )
    fp = corpus_fingerprint(
        corpus_path=corpus_path,
        model_path=model_path,
        n_docs=0,
        doc_tokens=doc_tokens,
        n_latents=n_latents,
        topk=topk,
        cls_sae_path=cls_sae_path,
        cls_topk=cls_topk,
    )
    id_map_path = artifact / "doc_id_map.json"
    if not id_map_path.is_file():
        return None
    id_map = DocIdMap.load(id_map_path)
    fp["n_docs"] = id_map.n_docs
    loaded = load_global_index_cache(
        artifact,
        bank_dir=corpus_path.parent,
        block_size=block_size,
        reorder_mode=reorder_mode,
        show_progress=show_progress,
        expected_corpus_fingerprint=fp,
    )
    if loaded is None:
        return None
    index, latent_remap, stats = loaded
    return index, latent_remap, stats, id_map
