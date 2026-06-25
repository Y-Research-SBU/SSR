#!/usr/bin/env python3
"""Evaluate SSR on MTEB BEIR-style retrieval data with sparse MaxSim."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ssr import _bootstrap  # noqa: F401

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prepare_mteb_eval import MTEB_EVAL_DATASETS  # noqa: E402
from ssr.model import SSR  # noqa: E402

from .device_utils import describe_hardware, resolve_encode_device, resolve_score_device
from .cls_sae import CLSSparseEncoder
from .exact_retriever import ExactSparseMaxSimRetriever
from .timing_stats import RetrievalTimingStats
from .mteb_io import iter_dataset_slugs, load_mteb_split
from .metrics import RetrievalMetrics, compute_retrieval_metrics
from .retriever import RetrieverConfig, build_retriever
from .sparse_repr import load_sparse_corpus, prune_sparse_rows, save_sparse_corpus
from .streaming_index_build import build_mteb_corpus_index_e2e, load_mteb_e2e_index

logger = logging.getLogger(__name__)


def resolve_model_checkpoint(model_path: Path) -> Path:
    """Prefer ``.../final`` when the user passes a training run directory."""
    path = model_path.resolve()
    if (path / "config.json").is_file() or (path / "model.safetensors").is_file():
        return path
    final = path / "final"
    if final.is_dir() and (final / "config.json").is_file():
        return final
    return path


def resolve_per_slug_index_cache_dir(base: Path, slug: str, *, multi_dataset: bool) -> Path:
    """Per-dataset cache root: ``base`` for a single slug, else ``base/slug``."""
    base = base.resolve()
    if not multi_dataset:
        return base
    return base / slug


def load_ssr(model_path: Path, device: str) -> SSR:
    model = SSR(model_name_or_path=str(model_path), device=device)
    model.eval()
    return model


def cache_suffix(model, cls_encoder: CLSSparseEncoder | None = None) -> str:
    token_part = f"{int(model.sae_module.topk)}k"
    if cls_encoder is None:
        return token_part
    return f"{token_part}_cls{int(cls_encoder.topk)}k"


def load_cls_encoder_from_args(
    args: argparse.Namespace,
    *,
    device: str,
) -> CLSSparseEncoder | None:
    if args.cls_sae_path is None:
        return None
    cls_encoder = CLSSparseEncoder.from_path(
        args.cls_sae_path,
        device=device,
        topk=args.cls_topk,
    )
    logger.info(
        "Loaded CLS SAE: %s (n_latents=%d, topk=%d)",
        args.cls_sae_path,
        cls_encoder.n_latents,
        cls_encoder.topk,
    )
    return cls_encoder


def is_msmarco_slug(slug: str) -> bool:
    normalized = slug.lower().replace("_", "-")
    return normalized in {"msmarco", "msmarco-passage", "ms-marco", "ms-marco-passage"}


def metrics_for_dataset(args: argparse.Namespace, slug: str) -> RetrievalMetrics:
    default_mrr = [10] if is_msmarco_slug(slug) else []
    default_ndcg = [] if is_msmarco_slug(slug) else [10]
    return RetrievalMetrics(
        accuracy_at_k=tuple(sorted(set(args.recall_k))),
        precision_recall_at_k=tuple(sorted(set(args.recall_k))),
        mrr_at_k=tuple(args.mrr_k if args.mrr_k is not None else default_mrr),
        ndcg_at_k=tuple(args.ndcg_k if args.ndcg_k is not None else default_ndcg),
        map_at_k=tuple(args.map_k),
    )


def apply_variant_defaults(args: argparse.Namespace) -> None:
    """Map public method names to the lower-level retrieval flags."""
    if args.variant is None:
        return
    variant = args.variant.lower()
    if variant == "ssr":
        args.mode = "exact"
        args.index_two_phase = False
        if args.cls_sae_path is not None:
            raise ValueError("--variant ssr is token-only; drop --cls-sae-path or use ssr-cls")
    elif variant == "ssr-cls":
        args.mode = "exact"
        args.index_two_phase = False
        if args.cls_sae_path is None:
            raise ValueError("--variant ssr-cls requires --cls-sae-path")
    elif variant == "ssr++":
        if args.corpus_backend == "e2e-index":
            args.mode = "exact"
            args.index_two_phase = True
        else:
            args.mode = "pruned"
    else:
        raise ValueError(f"Unknown --variant: {args.variant}")


def method_name(args: argparse.Namespace) -> str:
    if args.cls_sae_path is not None and (
        args.mode == "pruned" or getattr(args, "index_two_phase", False)
    ):
        return "SSR-CLS++"
    if args.cls_sae_path is not None:
        return "SSR-CLS"
    if args.mode == "pruned" or getattr(args, "index_two_phase", False):
        return "SSR++"
    return "SSR"


def encode_corpus_with_cache(
    model,
    split,
    *,
    cache_path: Path | None,
    retriever,
    force_reencode: bool,
    cls_encoder: CLSSparseEncoder | None = None,
):
    n_latents = model.sae_module.n_latents
    if cache_path and cache_path.is_file() and not force_reencode:
        logger.info("Loading cached corpus embeddings: %s", cache_path)
        doc_ids, sparse = load_sparse_corpus(cache_path)
        if doc_ids != split.corpus_ids:
            logger.warning(
                "Cached doc id count (%d) != corpus (%d); re-encoding.",
                len(doc_ids),
                len(split.corpus_ids),
            )
        else:
            return sparse

    logger.info("Encoding corpus (%d docs) ...", len(split.corpus_texts))
    sparse = retriever.encode_corpus(
        model,
        split.corpus_texts,
        n_latents=n_latents,
        device=str(next(model.parameters()).device),
        cls_encoder=cls_encoder,
    )
    if cache_path:
        save_sparse_corpus(cache_path, split.corpus_ids, sparse)
        logger.info("Saved corpus cache to %s", cache_path)
    return sparse


def evaluate_split(
    model,
    split,
    *,
    retriever,
    encode_device: str,
    score_device: str,
    metrics: RetrievalMetrics,
    cache_dir: Path | None,
    force_reencode: bool,
    top_k: int,
    cls_encoder: CLSSparseEncoder | None = None,
) -> Dict[str, float]:
    n_latents = model.sae_module.n_latents
    k_final = model.sae_module.topk
    retriever.config.final_topk = k_final

    cache_path = None
    if cache_dir:
        cache_path = (
            cache_dir
            / split.slug
            / f"corpus_{cache_suffix(model, cls_encoder)}_{split.split}.npz"
        )

    corpus_sparse = encode_corpus_with_cache(
        model,
        split,
        cache_path=cache_path,
        retriever=retriever,
        force_reencode=force_reencode,
        cls_encoder=cls_encoder,
    )

    logger.info("Encoding queries (%d) ...", len(split.query_texts))
    queries_sparse = retriever.encode_queries(
        model,
        split.query_texts,
        n_latents=n_latents,
        device=encode_device,
        cls_encoder=cls_encoder,
    )

    retrieval_timing: RetrievalTimingStats | None = None
    if isinstance(retriever, ExactSparseMaxSimRetriever):
        ranked = retriever.retrieve(
            queries=queries_sparse,
            corpus=corpus_sparse,
            corpus_ids=split.corpus_ids,
            top_k=top_k,
            score_device=score_device,
            show_progress=True,
        )
        retrieval_timing = retriever.last_retrieval_timing
    else:
        prune_k = retriever.config.prune_topk
        if prune_k and prune_k < k_final:
            query_prune = [prune_sparse_rows(q, prune_k) for q in queries_sparse]
        else:
            query_prune = queries_sparse
        query_fine = queries_sparse if retriever.config.use_fine_rerank else None
        ranked = retriever.retrieve(
            query_sparse=query_prune,
            query_sparse_fine=query_fine,
            corpus_sparse=corpus_sparse,
            corpus_ids=split.corpus_ids,
            top_k=top_k,
            device=score_device,
        )

    ir_scores = compute_retrieval_metrics(
        query_ids=split.query_ids,
        ranked_results=ranked,
        relevant_docs=split.relevant_docs,
        metrics=metrics,
    )
    if retrieval_timing is not None:
        retrieval_timing.log_summary(logger, prefix=f"Retrieval timing [{split.slug}]")
        ir_scores.update(retrieval_timing.as_dict())
    return ir_scores


def cache_corpus_embeddings_only(
    model,
    split,
    *,
    retriever,
    cache_dir: Path,
    force_reencode: bool,
    cls_encoder: CLSSparseEncoder | None = None,
) -> Dict[str, Any]:
    """Encode and save corpus sparse embeddings to ``cache_dir`` (no query retrieval)."""
    cache_path = (
        cache_dir
        / split.slug
        / f"corpus_{cache_suffix(model, cls_encoder)}_{split.split}.npz"
    )
    encode_corpus_with_cache(
        model,
        split,
        cache_path=cache_path,
        retriever=retriever,
        force_reencode=force_reencode,
        cls_encoder=cls_encoder,
    )
    return {
        "cache_path": str(cache_path),
        "n_docs": len(split.corpus_ids),
        "topk": int(model.sae_module.topk),
        "cls_topk": int(cls_encoder.topk) if cls_encoder is not None else None,
    }


def evaluate_split_e2e(
    model,
    split,
    *,
    retriever: ExactSparseMaxSimRetriever,
    encode_device: str,
    score_device: str,
    metrics: RetrievalMetrics,
    data_dir: Path,
    index_cache_dir: Path,
    model_path: Path,
    top_k: int,
    block_size: int,
    latent_reorder: str,
    cls_encoder: CLSSparseEncoder | None = None,
    cls_sae_path: Path | None = None,
) -> Dict[str, float]:
    """MTEB IR metrics using a pre-built E2E global index (queries encoded only)."""
    token_n_latents = model.sae_module.n_latents
    index_n_latents = token_n_latents
    k_final = model.sae_module.topk
    doc_tokens = int(getattr(model, "document_length", None) or 180)
    if cls_encoder is not None:
        index_n_latents += int(cls_encoder.n_latents)
        doc_tokens += 1
    retriever.config.final_topk = k_final

    corpus_path = data_dir / split.slug / "corpus.jsonl"
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)

    loaded = load_mteb_e2e_index(
        index_cache_dir=index_cache_dir,
        corpus_path=corpus_path,
        model_path=model_path,
        doc_tokens=doc_tokens,
        n_latents=index_n_latents,
        topk=k_final,
        block_size=block_size,
        reorder_mode=latent_reorder,
        show_progress=True,
        cls_sae_path=cls_sae_path,
        cls_topk=cls_encoder.topk if cls_encoder is not None else None,
    )
    if loaded is None:
        raise FileNotFoundError(
            f"E2E index not found under {index_cache_dir} "
            f"(expected global_index_v2_bs{block_size}_{latent_reorder}/doc_id_map.json). "
            "Run with --task index first."
        )
    index, _latent_remap, stats, id_map = loaded
    corpus_ids = id_map.external_ids
    if len(corpus_ids) != len(split.corpus_ids):
        logger.warning(
            "E2E index n_docs=%d != split corpus_ids=%d (corpus fingerprint may differ).",
            len(corpus_ids),
            len(split.corpus_ids),
        )
    logger.info(
        "Loaded E2E index: %d docs, %d postings",
        id_map.n_docs,
        stats.n_postings,
    )

    logger.info("Encoding queries (%d) ...", len(split.query_texts))
    queries_sparse = retriever.encode_queries(
        model,
        split.query_texts,
        n_latents=token_n_latents,
        device=encode_device,
        cls_encoder=cls_encoder,
    )

    ranked = retriever.retrieve_from_e2e_index(
        queries=queries_sparse,
        index=index,
        corpus_ids=corpus_ids,
        top_k=top_k,
        score_device=score_device,
        show_progress=True,
    )
    ir_scores = compute_retrieval_metrics(
        query_ids=split.query_ids,
        ranked_results=ranked,
        relevant_docs=split.relevant_docs,
        metrics=metrics,
    )
    if retriever.last_retrieval_timing is not None:
        retriever.last_retrieval_timing.log_summary(
            logger, prefix=f"Retrieval timing [{split.slug}, e2e-index]"
        )
        ir_scores.update(retriever.last_retrieval_timing.as_dict())
    return ir_scores


def run_mteb_index_build(args: argparse.Namespace) -> Dict[str, Any]:
    """Stream-build global inverted indexes for MTEB corpora (no sparse bank on disk)."""
    apply_variant_defaults(args)
    if args.model_path is None:
        raise ValueError("--model-path is required for --task index")

    encode_device = resolve_encode_device(args.encode_device)
    model_path = resolve_model_checkpoint(args.model_path)
    logger.info(
        describe_hardware(
            encode_device=encode_device,
            score_device="cpu",
            retrieval_mode="e2e-index-build",
        )
    )
    model = load_ssr(model_path, encode_device)
    cls_encoder = load_cls_encoder_from_args(args, device=encode_device)
    logger.info("Index variant: %s", method_name(args))

    data_dir = args.data_dir.resolve()
    slugs = _resolve_dataset_slugs(args, data_dir)
    multi = len(slugs) > 1
    if args.index_cache_dir is None:
        raise ValueError("--index-cache-dir is required for --task index")

    base_cache = args.index_cache_dir.resolve()
    results: Dict[str, Any] = {}

    for slug in slugs:
        logger.info("=== E2E index build: %s ===", slug)
        corpus_path = data_dir / slug / "corpus.jsonl"
        if not corpus_path.is_file():
            raise FileNotFoundError(corpus_path)
        per_slug_cache = resolve_per_slug_index_cache_dir(
            base_cache, slug, multi_dataset=multi
        )
        if not args.force_rebuild_index:
            doc_tokens = int(getattr(model, "document_length", None) or 180)
            n_latents = int(model.sae_module.n_latents)
            if cls_encoder is not None:
                doc_tokens += 1
                n_latents += int(cls_encoder.n_latents)
            existing = load_mteb_e2e_index(
                index_cache_dir=per_slug_cache,
                corpus_path=corpus_path,
                model_path=model_path,
                doc_tokens=doc_tokens,
                n_latents=n_latents,
                topk=model.sae_module.topk,
                block_size=args.block_size,
                reorder_mode=args.latent_reorder,
                show_progress=False,
                cls_sae_path=args.cls_sae_path,
                cls_topk=cls_encoder.topk if cls_encoder is not None else None,
            )
            if existing is not None:
                _index, _remap, stats, id_map = existing
                logger.info(
                    "Skipping build (cache hit): %d docs, %d postings at %s",
                    id_map.n_docs,
                    stats.n_postings,
                    per_slug_cache,
                )
                results[slug] = {
                    "skipped": True,
                    "n_docs": id_map.n_docs,
                    "n_postings": stats.n_postings,
                    "index_cache_dir": str(per_slug_cache),
                }
                continue

        artifact, id_map, stats = build_mteb_corpus_index_e2e(
            model,
            corpus_path=corpus_path,
            model_path=model_path,
            index_cache_dir=per_slug_cache,
            encode_device=encode_device,
            block_size=args.block_size,
            reorder_mode=args.latent_reorder,
            cooc_sample_rate=args.cooc_sample_rate,
            encode_batch_size=args.encode_batch_size,
            max_docs=args.max_corpus or 0,
            empty_cache_every=args.empty_cache_every,
            cls_encoder=cls_encoder,
            cls_sae_path=args.cls_sae_path,
            cls_topk=cls_encoder.topk if cls_encoder is not None else None,
        )
        results[slug] = {
            "skipped": False,
            "artifact": str(artifact),
            "n_docs": id_map.n_docs,
            "n_postings": stats.n_postings,
        }
        logger.info("Built index: %s", artifact)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def _resolve_dataset_slugs(args: argparse.Namespace, data_dir: Path) -> List[str]:
    if getattr(args, "dataset", None):
        if args.datasets:
            raise ValueError("Use either --dataset or --datasets, not both")
        return [str(args.dataset)]
    if args.datasets:
        return list(args.datasets)
    return list(iter_dataset_slugs(data_dir, None))


def _retriever_config_from_args(args: argparse.Namespace, score_backend: str) -> RetrieverConfig:
    use_index_first = args.mode == "exact" and not getattr(args, "brute_force_torch", False)
    index_then_gpu = use_index_first and str(
        resolve_score_device(args.score_device, encode_device=args.encode_device or "cpu")
    ).startswith("cuda")
    return RetrieverConfig(
        mode=args.mode,
        corpus_chunk_size=args.corpus_chunk_size,
        block_size=args.block_size,
        prune_topk=args.prune_topk,
        n_candidates=args.n_candidates,
        encode_batch_size=args.encode_batch_size,
        query_batch_size=args.query_batch_size,
        use_fine_rerank=not args.no_fine_rerank,
        score_backend=score_backend,
        use_index_first=use_index_first,
        index_then_gpu=index_then_gpu,
        use_brute_force_torch=getattr(args, "brute_force_torch", False),
        gpu_maxsim_max_candidates=getattr(args, "gpu_maxsim_max_candidates", 16_384),
        index_two_phase=getattr(args, "index_two_phase", False),
        two_phase_pool_size=getattr(args, "two_phase_pool_size", 32_768),
        coarse_topk=getattr(args, "coarse_topk", 8),
        query_latent_top_k=getattr(args, "query_latent_top_k", 0),
        index_accum_device=getattr(args, "index_accum_device", "cpu"),
        gpu_hot_latent_budget_gb=getattr(args, "gpu_hot_budget_gb", 8.0),
        latent_reorder_mode=args.latent_reorder,
        cooc_sample_rate=args.cooc_sample_rate,
        index_cache_dir=args.index_cache_dir.resolve() if args.index_cache_dir else None,
        force_rebuild_index=args.force_rebuild_index,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    all_slugs = sorted(MTEB_EVAL_DATASETS.keys())
    parser = argparse.ArgumentParser(
        description=(
            "SSR MTEB sparse MaxSim: --task index (E2E inverted index) or "
            "retrieval (IR metrics; sparse-cache or e2e-index)."
        )
    )
    parser.add_argument(
        "--task",
        choices=("index", "retrieval"),
        default="retrieval",
        help="index: stream corpus→encode→global index; retrieval: score queries (default).",
    )
    parser.add_argument(
        "--variant",
        choices=("ssr", "ssr-cls", "ssr++"),
        default=None,
        help=(
            "Public method preset: ssr=token-only exact, ssr-cls=token+[CLS] exact "
            "(requires --cls-sae-path), ssr++=pruned/two-phase retrieval."
        ),
    )
    parser.add_argument(
        "--corpus-backend",
        choices=("sparse-cache", "e2e-index"),
        default="sparse-cache",
        help=(
            "[retrieval] sparse-cache: encode/load sparse corpus npz; "
            "e2e-index: load pre-built global index (--index-cache-dir, exact mode)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help="Single dataset slug (alternative to one entry in --datasets).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="SSR checkpoint path (directory with config.json or .../final).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/mteb"),
        help="Root directory of prepared MTEB datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=f"Dataset slugs to evaluate (default: all under data-dir).",
    )
    parser.add_argument("--split", default="test", help="Query/qrels split name.")
    parser.add_argument(
        "--device",
        "--encode-device",
        dest="encode_device",
        default=None,
        help="Model encoding device: cuda, cpu, cuda:0 (default: cuda if available).",
    )
    parser.add_argument(
        "--score-device",
        default=None,
        choices=("auto", "cpu", "cuda", "index"),
        help=(
            "MaxSim scoring device. exact+index=CPU inverted index (fast, exact); "
            "exact+cuda/cpu=torch.sparse; pruned: cuda/cpu for fine rerank."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("pruned", "exact"),
        default="pruned",
        help=(
            "pruned: multi-stage (query top-k prune + candidate cap + rerank). "
            "exact: full sparse MaxSim, identical to dense MaxSim on SAE vectors."
        ),
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES index.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write per-dataset metrics JSON to this path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/mteb_sparse"),
        help="[sparse-cache] Directory for cached sparse corpus embeddings.",
    )
    parser.add_argument(
        "--index-cache-dir",
        type=Path,
        default=None,
        help=(
            "E2E global index root. Single dataset: cache lives directly here; "
            "multiple datasets: one subdir per slug."
        ),
    )
    parser.add_argument(
        "--latent-reorder",
        choices=("none", "frequency", "cooc"),
        default="frequency",
        help="Latent reorder for E2E index build/load (must match at retrieval).",
    )
    parser.add_argument(
        "--cooc-sample-rate",
        type=float,
        default=0.01,
        help="Co-occurrence sample rate when --latent-reorder=cooc.",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=4,
        help="[index] torch.cuda.empty_cache() every N encode batches (0=off).",
    )
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="[index] Rebuild even if a matching E2E cache exists.",
    )
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument(
        "--encode-corpus-only",
        action="store_true",
        help=(
            "[sparse-cache] Only encode corpus and write npz under --cache-dir; "
            "skip query retrieval and IR metrics."
        ),
    )
    parser.add_argument("--no-fine-rerank", action="store_true")
    parser.add_argument("--corpus-chunk-size", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--prune-topk", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Final retrieval depth (also drives metric heap size).",
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        nargs="+",
        default=None,
        help="nDCG cutoffs. Default: [10] for non-MSMARCO datasets, disabled for MSMARCO.",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=[1, 10, 100],
    )
    parser.add_argument(
        "--mrr-k",
        type=int,
        nargs="+",
        default=None,
        help="MRR cutoffs. Default: [10] for MSMARCO datasets, disabled otherwise.",
    )
    parser.add_argument(
        "--map-k",
        type=int,
        nargs="+",
        default=[100],
    )
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=None,
        help="Limit corpus size (debug).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit query count (debug).",
    )
    parser.add_argument(
        "--brute-force-torch",
        action="store_true",
        help="[exact] Legacy all-pairs torch.sparse.mm (slow; disables index-first).",
    )
    parser.add_argument(
        "--gpu-maxsim-max-candidates",
        type=int,
        default=16_384,
        help="[exact+cuda] Fall back to CPU index if a query has more overlap candidates.",
    )
    parser.add_argument(
        "--index-two-phase",
        action="store_true",
        help="[exact+e2e-index] Coarse postings + candidate-pool exact MaxSim.",
    )
    parser.add_argument("--two-phase-pool-size", type=int, default=32_768)
    parser.add_argument("--coarse-topk", type=int, default=8)
    parser.add_argument(
        "--index-accum-device",
        choices=("cpu", "cuda", "auto", "hybrid"),
        default="cpu",
        help="Posting accumulate device during index scoring.",
    )
    parser.add_argument(
        "--query-latent-top-k",
        type=int,
        default=0,
        help="Cap query latents per token (0=all; lossy if too small).",
    )
    parser.add_argument("--gpu-hot-budget-gb", type=float, default=8.0)
    parser.add_argument(
        "--cls-sae-path",
        type=Path,
        default=None,
        help="Optional separate SAE checkpoint for [CLS] embeddings during retrieval/indexing.",
    )
    parser.add_argument(
        "--cls-topk",
        type=int,
        default=None,
        help="Top-K for --cls-sae-path (default: CLS SAE checkpoint topk).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def run_mteb_eval(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.model_path is None:
        raise ValueError("--model-path is required")

    apply_variant_defaults(args)

    if args.task == "index":
        return run_mteb_index_build(args)

    if args.corpus_backend == "e2e-index":
        if args.mode != "exact":
            raise ValueError("--corpus-backend e2e-index requires --mode exact")
        if args.index_cache_dir is None:
            raise ValueError("--index-cache-dir is required for --corpus-backend e2e-index")
    if args.encode_corpus_only and args.corpus_backend != "sparse-cache":
        raise ValueError("--encode-corpus-only requires --corpus-backend sparse-cache")

    encode_device = resolve_encode_device(args.encode_device)
    score_device = resolve_score_device(args.score_device, encode_device=encode_device)

    if args.mode == "exact":
        score_backend = args.score_device or "index"
        if score_backend == "auto":
            score_backend = "index"
    else:
        score_backend = args.score_device or "auto"

    logger.info(
        describe_hardware(
            encode_device=encode_device,
            score_device=score_device,
            retrieval_mode=f"{args.mode}+{args.corpus_backend}",
        )
    )

    model_path = resolve_model_checkpoint(args.model_path)
    model = load_ssr(model_path, encode_device)
    cls_encoder = load_cls_encoder_from_args(args, device=encode_device)
    retriever = build_retriever(_retriever_config_from_args(args, score_backend))
    logger.info("Evaluation variant: %s", method_name(args))

    data_dir = args.data_dir.resolve()
    slugs = _resolve_dataset_slugs(args, data_dir)
    multi_index = len(slugs) > 1
    base_index_cache = args.index_cache_dir.resolve() if args.index_cache_dir else None

    all_results: Dict[str, Dict[str, float]] = {}
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    if cache_dir and args.corpus_backend == "sparse-cache":
        cache_dir.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        logger.info("=== Evaluating %s (%s) ===", slug, MTEB_EVAL_DATASETS.get(slug, slug))
        metrics = metrics_for_dataset(args, slug)
        eval_top_k = max(metrics.all_k(), args.top_k)
        logger.info(
            "Metrics for %s: mrr@%s ndcg@%s",
            slug,
            list(metrics.mrr_at_k),
            list(metrics.ndcg_at_k),
        )
        split = load_mteb_split(data_dir, slug, args.split)
        if args.max_corpus:
            split = type(split)(
                slug=split.slug,
                split=split.split,
                corpus_ids=split.corpus_ids[: args.max_corpus],
                corpus_texts=split.corpus_texts[: args.max_corpus],
                query_ids=split.query_ids,
                query_texts=split.query_texts,
                relevant_docs=split.relevant_docs,
            )
        if args.max_queries:
            qids = split.query_ids[: args.max_queries]
            split = type(split)(
                slug=split.slug,
                split=split.split,
                corpus_ids=split.corpus_ids,
                corpus_texts=split.corpus_texts,
                query_ids=qids,
                query_texts=[split.query_texts[split.query_ids.index(q)] for q in qids],
                relevant_docs={q: split.relevant_docs[q] for q in qids if q in split.relevant_docs},
            )
        if args.encode_corpus_only:
            if cache_dir is None:
                raise ValueError("--cache-dir is required for --encode-corpus-only")
            info = cache_corpus_embeddings_only(
                model,
                split,
                retriever=retriever,
                cache_dir=cache_dir,
                force_reencode=args.force_reencode,
                cls_encoder=cls_encoder,
            )
            all_results[slug] = info
            logger.info("Cached corpus embeddings: %s", info["cache_path"])
            continue
        if args.corpus_backend == "e2e-index":
            if not isinstance(retriever, ExactSparseMaxSimRetriever):
                raise ValueError("e2e-index retrieval requires ExactSparseMaxSimRetriever")
            per_slug_cache = resolve_per_slug_index_cache_dir(
                base_index_cache, slug, multi_dataset=multi_index
            )
            scores = evaluate_split_e2e(
                model,
                split,
                retriever=retriever,
                encode_device=encode_device,
                score_device=score_device,
                metrics=metrics,
                data_dir=data_dir,
                index_cache_dir=per_slug_cache,
                model_path=model_path,
                top_k=eval_top_k,
                block_size=args.block_size,
                latent_reorder=args.latent_reorder,
                cls_encoder=cls_encoder,
                cls_sae_path=args.cls_sae_path,
            )
        else:
            scores = evaluate_split(
                model,
                split,
                retriever=retriever,
                encode_device=encode_device,
                score_device=score_device,
                metrics=metrics,
                cache_dir=cache_dir,
                force_reencode=args.force_reencode,
                top_k=eval_top_k,
                cls_encoder=cls_encoder,
            )
        all_results[slug] = scores
        for name, val in sorted(scores.items()):
            logger.info("  %s: %.4f", name, val)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Wrote metrics to %s", args.output_json)

    return all_results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_mteb_eval(args)


if __name__ == "__main__":
    main()
