"""Persist / load global inverted indexes (E2E and shard banks)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .compact_postings import (
    CompactPostings,
    build_doc_latent_max,
    compute_latent_max,
    postings_dict_to_compact,
)
from .inverted_index import BlockInvertedIndex

logger = logging.getLogger(__name__)

# v2: directory + np.savez (avoids pickle 4GiB limit on huge posting arrays)
CACHE_FORMAT_VERSION = 2
_PICKLE_PROTOCOL = 5


@dataclass
class GlobalIndexBuildStats:
    n_shards_ingested: int = 0
    n_postings: int = 0
    n_latents_active: int = 0
    reorder_mode: str = "frequency"


def default_cache_dir(bank_dir: Path) -> Path:
    return bank_dir.resolve() / ".global_index_cache"


def cache_artifact_path(
    cache_dir: Path,
    *,
    block_size: int,
    reorder_mode: str,
) -> Path:
    """Directory path for the v2 on-disk cache bundle."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"global_index_v{CACHE_FORMAT_VERSION}_bs{block_size}_{reorder_mode}"


def legacy_cache_artifact_path(
    cache_dir: Path,
    *,
    block_size: int,
    reorder_mode: str,
) -> Path:
    """Legacy single-file ``.pt`` cache (v1)."""
    return cache_dir / f"global_index_v1_bs{block_size}_{reorder_mode}.pt"


def corpus_fingerprint(
    *,
    corpus_path: Path,
    model_path: Path,
    n_docs: int,
    doc_tokens: int,
    n_latents: int,
    topk: int,
    cls_sae_path: Path | None = None,
    cls_topk: int | None = None,
) -> dict[str, Any]:
    """Fingerprint for MTEB / JSONL streaming index builds (no pre-materialized bank)."""
    corpus_path = corpus_path.resolve()
    model_path = model_path.resolve()
    c_stat = corpus_path.stat()
    m_stat = model_path.stat()
    fp = {
        "kind": "corpus_jsonl",
        "corpus_path": str(corpus_path),
        "corpus_mtime_ns": int(c_stat.st_mtime_ns),
        "corpus_size": int(c_stat.st_size),
        "model_path": str(model_path),
        "model_mtime_ns": int(m_stat.st_mtime_ns),
        "n_docs": int(n_docs),
        "doc_tokens": int(doc_tokens),
        "n_latents": int(n_latents),
        "topk": int(topk),
    }
    if cls_sae_path is not None:
        cls_path = cls_sae_path.resolve()
        cls_stat = cls_path.stat()
        fp.update(
            {
                "cls_sae_path": str(cls_path),
                "cls_sae_mtime_ns": int(cls_stat.st_mtime_ns),
                "cls_topk": int(cls_topk) if cls_topk is not None else None,
            }
        )
    return fp


def bank_fingerprint(bank_dir: Path) -> dict[str, Any]:
    meta_path = bank_dir.resolve() / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    keys = (
        "version",
        "n_docs",
        "n_queries",
        "doc_tokens",
        "query_tokens",
        "n_latents",
        "topk",
        "shard_size_docs",
        "seed",
        "n_corpus_shards",
    )
    return {k: meta[k] for k in keys if k in meta}


def _validate_cache_meta(
    meta: dict[str, Any],
    *,
    bank_dir: Path | None,
    block_size: int,
    reorder_mode: str,
    path: Path,
    expected_corpus_fingerprint: dict[str, Any] | None = None,
) -> bool:
    if int(meta.get("format_version", 0)) != CACHE_FORMAT_VERSION:
        logger.warning("Cache format mismatch at %s; rebuilding index.", path)
        return False
    if expected_corpus_fingerprint is not None:
        if meta.get("corpus_fingerprint") != expected_corpus_fingerprint:
            logger.warning("Corpus fingerprint mismatch for %s; rebuilding index.", path)
            return False
    elif "corpus_fingerprint" in meta:
        pass
    elif bank_dir is not None:
        if meta.get("bank_fingerprint") != bank_fingerprint(bank_dir):
            logger.warning("Bank fingerprint mismatch for %s; rebuilding index.", path)
            return False
    else:
        logger.warning("No fingerprint to validate cache at %s", path)
        return False
    if int(meta["block_size"]) != block_size or meta["reorder_mode"] != reorder_mode:
        logger.warning(
            "Cache params mismatch (want bs=%s mode=%s); rebuilding index.",
            block_size,
            reorder_mode,
        )
        return False
    return True


def _index_from_payload(
    meta: dict[str, Any],
    compact_arrays: dict[str, np.ndarray],
    *,
    block_size: int,
    show_progress: bool,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats]:
    index = BlockInvertedIndex(
        n_latents=int(meta["n_latents"]),
        block_size=block_size,
    )
    index.n_docs = int(meta["n_docs"])
    index.compact = CompactPostings.from_arrays_dict(compact_arrays)
    n_latents = int(meta["n_latents"])
    if "latent_max" in compact_arrays:
        index.latent_max = np.asarray(compact_arrays["latent_max"], dtype=np.float32)
    else:
        index.latent_max = compute_latent_max(index.compact, n_latents)
    if "block_doc_max" in compact_arrays:
        index.block_doc_max = np.asarray(compact_arrays["block_doc_max"], dtype=np.float16)
    if "coarse_latent_ids" in compact_arrays:
        index.compact_coarse = CompactPostings.from_arrays_dict(
            {
                "latent_ids": compact_arrays["coarse_latent_ids"],
                "offsets": compact_arrays["coarse_offsets"],
                "doc_idx": compact_arrays["coarse_doc_idx"],
                "token_idx": compact_arrays["coarse_token_idx"],
                "values": compact_arrays["coarse_values"],
            }
        )
        index.coarse_topk = int(meta.get("coarse_topk", 8))
    if "doc_latent_max_offsets" in compact_arrays:
        index.doc_latent_max = CompactPostings.from_arrays_dict(
            {
                "latent_ids": compact_arrays["doc_latent_max_latent_ids"],
                "offsets": compact_arrays["doc_latent_max_offsets"],
                "doc_idx": compact_arrays["doc_latent_max_doc_idx"],
                "token_idx": compact_arrays.get(
                    "doc_latent_max_token_idx",
                    np.array([], dtype=np.int16),
                ),
                "values": compact_arrays["doc_latent_max_values"],
            }
        )
    index.block_latents = meta["block_latents"]
    if not index.block_latents:
        index.finalize_block_latents()

    latent_remap = np.asarray(meta["latent_remap"], dtype=np.int32)
    st = meta["stats"]
    stats = GlobalIndexBuildStats(
        n_shards_ingested=int(st["n_shards_ingested"]),
        n_postings=int(st["n_postings"]),
        n_latents_active=int(st["n_latents_active"]),
        reorder_mode=str(st["reorder_mode"]),
    )
    return index, latent_remap, stats


def save_global_index_cache(
    path: Path,
    *,
    index: BlockInvertedIndex,
    latent_remap: np.ndarray,
    stats: GlobalIndexBuildStats,
    bank_dir: Path,
    block_size: int,
    reorder_mode: str,
    corpus_fingerprint: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> None:
    """Write v2 cache directory ``path/`` (meta.json + postings.npz)."""
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving global index cache to %s", path)
    arrays_path = path / "postings.npz"
    latent_max = index.latent_max
    if latent_max is None:
        latent_max = compute_latent_max(index.compact, index.n_latents)
    compact_arrays = index.compact.to_arrays_dict()
    compact_arrays["values"] = np.asarray(
        compact_arrays["values"], dtype=np.float16
    )
    save_arrays = {
        **compact_arrays,
        "latent_remap": np.asarray(latent_remap, dtype=np.int32),
        "latent_max": np.asarray(latent_max, dtype=np.float32),
    }
    if index.block_doc_max is None:
        from .index_pruning import ensure_block_doc_max

        index.block_doc_max = ensure_block_doc_max(index, show_progress=show_progress)
    save_arrays["block_doc_max"] = np.asarray(index.block_doc_max, dtype=np.float16)
    coarse = index.compact_coarse
    if coarse is None or coarse.n_postings == 0:
        from .compact_postings import downsample_compact_per_token_topk

        coarse = downsample_compact_per_token_topk(
            index.compact, topk=int(index.coarse_topk or 8), show_progress=show_progress
        )
        index.compact_coarse = coarse
    coarse_vals = np.asarray(coarse.values, dtype=np.float16)
    save_arrays.update(
        {
            "coarse_latent_ids": coarse.latent_ids,
            "coarse_offsets": coarse.offsets,
            "coarse_doc_idx": coarse.doc_idx,
            "coarse_token_idx": coarse.token_idx,
            "coarse_values": coarse_vals,
        }
    )
    if index.doc_latent_max is not None and index.doc_latent_max.n_postings > 0:
        dlm = index.doc_latent_max
        save_arrays.update(
            {
                "doc_latent_max_latent_ids": dlm.latent_ids,
                "doc_latent_max_offsets": dlm.offsets,
                "doc_latent_max_doc_idx": dlm.doc_idx,
                "doc_latent_max_values": dlm.values,
            }
        )
    if show_progress:
        with tqdm(total=1, desc="Global index cache [write arrays]") as pbar:
            np.savez(arrays_path, **save_arrays)
            pbar.update(1)
    else:
        np.savez(arrays_path, **save_arrays)

    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "bank_fingerprint": bank_fingerprint(bank_dir) if corpus_fingerprint is None else None,
        "corpus_fingerprint": corpus_fingerprint,
        "block_size": block_size,
        "reorder_mode": reorder_mode,
        "n_latents": index.n_latents,
        "n_docs": index.n_docs,
        "stats": {
            "n_shards_ingested": stats.n_shards_ingested,
            "n_postings": stats.n_postings,
            "n_latents_active": stats.n_latents_active,
            "reorder_mode": stats.reorder_mode,
        },
        "block_latents": index.block_latents,
        "coarse_topk": int(index.coarse_topk or 8),
        "n_coarse_postings": int(index.compact_coarse.n_postings)
        if index.compact_coarse is not None
        else 0,
    }
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info("Global index cache saved (%d postings)", stats.n_postings)


def _load_v2_cache_dir(
    path: Path,
    *,
    bank_dir: Path | None,
    block_size: int,
    reorder_mode: str,
    show_progress: bool,
    expected_corpus_fingerprint: dict[str, Any] | None = None,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats] | None:
    meta_path = path / "meta.json"
    arrays_path = path / "postings.npz"
    if not meta_path.is_file() or not arrays_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if not _validate_cache_meta(
        meta,
        bank_dir=bank_dir,
        block_size=block_size,
        reorder_mode=reorder_mode,
        path=path,
        expected_corpus_fingerprint=expected_corpus_fingerprint,
    ):
        return None

    _NPZ_KEYS = (
        "latent_ids",
        "offsets",
        "doc_idx",
        "token_idx",
        "values",
        "latent_max",
        "latent_remap",
        "block_doc_max",
        "coarse_latent_ids",
        "coarse_offsets",
        "coarse_doc_idx",
        "coarse_token_idx",
        "coarse_values",
    )

    if show_progress:
        with tqdm(total=1, desc="Global index cache [read arrays]") as pbar:
            with np.load(arrays_path) as z:
                compact = {k: z[k] for k in z.files if k in _NPZ_KEYS}
            pbar.update(1)
    else:
        with np.load(arrays_path) as z:
            compact = {k: z[k] for k in z.files if k in _NPZ_KEYS}

    latent_remap = np.asarray(compact.pop("latent_remap"), dtype=np.int32)

    meta["latent_remap"] = latent_remap
    index, latent_remap, stats = _index_from_payload(
        meta, compact, block_size=block_size, show_progress=show_progress
    )
    logger.info(
        "Loaded global index cache: %d postings, reorder=%s",
        stats.n_postings,
        stats.reorder_mode,
    )
    return index, latent_remap, stats


def _load_v1_pt_cache(
    path: Path,
    *,
    bank_dir: Path,
    block_size: int,
    reorder_mode: str,
    show_progress: bool,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats] | None:
    if not path.is_file():
        return None
    logger.info("Loading legacy global index cache from %s", path)
    try:
        if show_progress:
            with tqdm(total=1, desc="Global index cache [read file]") as pbar:
                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                    pickle_protocol=_PICKLE_PROTOCOL,
                )
                pbar.update(1)
        else:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                pickle_protocol=_PICKLE_PROTOCOL,
            )
    except Exception as exc:
        logger.warning("Failed to load legacy cache %s (%s); rebuilding index.", path, exc)
        return None

    if int(payload.get("format_version", 0)) != 1:
        logger.warning("Unexpected legacy cache version at %s; rebuilding index.", path)
        return None
    meta = {
        "format_version": 1,
        "bank_fingerprint": payload["bank_fingerprint"],
        "block_size": payload["block_size"],
        "reorder_mode": payload["reorder_mode"],
        "n_latents": payload["n_latents"],
        "n_docs": payload["n_docs"],
        "latent_remap": payload["latent_remap"],
        "stats": payload["stats"],
        "block_latents": payload["block_latents"],
    }
    if not _validate_cache_meta(
        meta, bank_dir=bank_dir, block_size=block_size, reorder_mode=reorder_mode, path=path
    ):
        return None
    # Normalize meta for _index_from_payload (expects v2-shaped meta dict)
    meta["format_version"] = CACHE_FORMAT_VERSION
    legacy_postings = payload["postings"]
    if isinstance(legacy_postings, dict) and legacy_postings.get("latent_ids") is not None:
        compact_arrays = legacy_postings
    else:
        compact_arrays = postings_dict_to_compact(
            legacy_postings, show_progress=show_progress
        ).to_arrays_dict()
    index, latent_remap, stats = _index_from_payload(
        meta,
        compact_arrays,
        block_size=block_size,
        show_progress=show_progress,
    )
    logger.info(
        "Loaded legacy global index cache: %d postings, reorder=%s",
        stats.n_postings,
        stats.reorder_mode,
    )
    return index, latent_remap, stats


def load_global_index_cache(
    path: Path,
    *,
    bank_dir: Path | None = None,
    block_size: int = 512,
    reorder_mode: str = "frequency",
    show_progress: bool = True,
    expected_corpus_fingerprint: dict[str, Any] | None = None,
) -> tuple[BlockInvertedIndex, np.ndarray, GlobalIndexBuildStats] | None:
    """Load v2 cache directory, or fall back to legacy v1 ``.pt`` next to it."""
    if path.is_dir():
        loaded = _load_v2_cache_dir(
            path,
            bank_dir=bank_dir,
            block_size=block_size,
            reorder_mode=reorder_mode,
            show_progress=show_progress,
            expected_corpus_fingerprint=expected_corpus_fingerprint,
        )
        if loaded is not None:
            return loaded

    legacy = legacy_cache_artifact_path(
        path.parent if path.is_dir() else path.parent,
        block_size=block_size,
        reorder_mode=reorder_mode,
    )
    return _load_v1_pt_cache(
        legacy,
        bank_dir=bank_dir,
        block_size=block_size,
        reorder_mode=reorder_mode,
        show_progress=show_progress,
    )
