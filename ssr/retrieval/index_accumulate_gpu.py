"""GPU MaxSim posting accumulation (hybrid hot-cache + CPU cold; small-corpus CUDA stamp)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch

from .gpu_hot_latent_cache import ensure_gpu_hot_cache
from .index_accumulate_fast import (
    _accumulate_stamp_qt_only,
    _bump_generation,
    _get_stamp_state,
    build_maxsim_work_from_plan,
    filter_cold_work,
    filter_work_by_candidate_range,
)

logger = logging.getLogger(__name__)

IndexAccumDevice = Literal["cpu", "cuda", "auto", "hybrid"]

_GPU_QT_STATE: dict[tuple[int, int, str], torch.Tensor] = {}
_GPU_STAMP_STATE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}

_NUMBA_OK = False
_CUDA_STAMP_OK = False
try:
    from numba import njit, prange

    _NUMBA_OK = True
except ImportError:
    njit = lambda *a, **k: (lambda f: f)  # type: ignore[misc]
    prange = range  # type: ignore[misc, assignment]

try:
    from numba import cuda as numba_cuda

    _CUDA_STAMP_OK = bool(_NUMBA_OK and numba_cuda.is_available())
except ImportError:
    numba_cuda = None  # type: ignore[assignment]


if _CUDA_STAMP_OK:

    @numba_cuda.jit(device=True)
    def _dev_apply_contrib_flat(
        qt_scores,
        qt_gen,
        doc_total,
        flat_idx,
        doc_d,
        contrib,
        generation,
    ):
        if qt_gen[flat_idx] != generation:
            qt_gen[flat_idx] = generation
            qt_scores[flat_idx] = contrib
            doc_total[doc_d] += contrib
        elif contrib > qt_scores[flat_idx]:
            doc_total[doc_d] += contrib - qt_scores[flat_idx]
            qt_scores[flat_idx] = contrib

    @numba_cuda.jit(device=True)
    def _dev_update_segment_qt_flat(
        qt_scores,
        qt_gen,
        doc_total,
        doc_idx,
        values,
        s,
        e,
        v_q,
        qt,
        n_docs,
        generation,
    ):
        n = e - s
        if n <= 0:
            return
        d0 = doc_idx[s]
        vmax = values[s] * v_q
        i = s + 1
        while i < e:
            d = doc_idx[i]
            v = values[i] * v_q
            if d == d0:
                if v > vmax:
                    vmax = v
            else:
                _dev_apply_contrib_flat(
                    qt_scores,
                    qt_gen,
                    doc_total,
                    qt * n_docs + d0,
                    d0,
                    vmax,
                    generation,
                )
                d0 = d
                vmax = v
            i += 1
        _dev_apply_contrib_flat(
            qt_scores,
            qt_gen,
            doc_total,
            qt * n_docs + d0,
            d0,
            vmax,
            generation,
        )

    @numba_cuda.jit
    def _kernel_stamp_qt_work(
        qt_scores,
        qt_gen,
        doc_total,
        doc_idx,
        values,
        seg_start,
        seg_end,
        work_qt_offsets,
        work_seg,
        work_vq,
        generation,
        n_qt,
        n_docs,
    ):
        qt = numba_cuda.grid(1)
        if qt >= n_qt:
            return
        wb = work_qt_offsets[qt]
        we = work_qt_offsets[qt + 1]
        for wi in range(wb, we):
            si = work_seg[wi]
            vq = work_vq[wi]
            s = seg_start[si]
            e = seg_end[si]
            _dev_update_segment_qt_flat(
                qt_scores,
                qt_gen,
                doc_total,
                doc_idx,
                values,
                s,
                e,
                vq,
                qt,
                n_docs,
                generation,
            )


def resolve_index_accum_device(requested: IndexAccumDevice) -> str:
    if requested == "cpu":
        return "cpu"
    # ``cuda`` is a deprecated alias for hybrid (flat scatter removed).
    if requested in ("cuda", "hybrid", "auto"):
        return "hybrid" if torch.cuda.is_available() else "cpu"
    return "hybrid" if torch.cuda.is_available() else "cpu"


def _resolve_torch_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    if device.startswith("cuda:"):
        return torch.device(device)
    return torch.device(device)


def _get_qt_scores_buffer(n_qt: int, n_docs: int, device: torch.device) -> torch.Tensor:
    key = (n_qt, n_docs, str(device))
    buf = _GPU_QT_STATE.get(key)
    if buf is None:
        buf = torch.zeros((n_qt, n_docs), dtype=torch.float32, device=device)
        _GPU_QT_STATE[key] = buf
    return buf


def _get_gpu_stamp_state(
    n_qt: int, n_docs: int, torch_device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    key = (n_qt, n_docs, str(torch_device))
    st = _GPU_STAMP_STATE.get(key)
    if st is None:
        qt_scores = torch.zeros(
            (n_qt, n_docs), dtype=torch.float32, device=torch_device
        )
        qt_gen = torch.zeros((n_qt, n_docs), dtype=torch.int32, device=torch_device)
        doc_total = torch.zeros(n_docs, dtype=torch.float32, device=torch_device)
        st = (qt_scores, qt_gen, doc_total, 1)
        _GPU_STAMP_STATE[key] = st
    return st


def _bump_gpu_generation(qt_gen: torch.Tensor, generation: int) -> int:
    generation += 1
    if generation > 2_000_000_000:
        qt_gen.zero_()
        generation = 1
    return generation


def accumulate_maxsim_gpu_stamp_from_plan(
    plan,
    index,
    *,
    n_qt: int,
    device: str = "cuda",
    budget_bytes: int | None = None,
) -> tuple[str, torch.Tensor] | None:
    """Generation-stamp MaxSim on GPU (no per-query zero of qt_scores)."""
    if not _CUDA_STAMP_OK:
        return None
    from .gpu_compact_cache import ensure_gpu_compact_postings

    if not ensure_gpu_compact_postings(
        index, device=device, max_bytes=budget_bytes
    ):
        return None

    work = build_maxsim_work_from_plan(plan, index, n_qt)
    if work is None:
        torch_device = _resolve_torch_device(device)
        _qs, _qg, doc_total, _gen = _get_gpu_stamp_state(
            n_qt, int(index.n_docs), torch_device
        )
        doc_total.zero_()
        return "cuda-stamp-empty", doc_total

    (
        seg_start,
        seg_end,
        _seg_qt_offsets,
        _seg_qt,
        _seg_vq,
        work_qt_offsets,
        work_seg,
        work_vq,
        _work_qt_item,
        _item_offsets,
        _seg_compact_row,
    ) = work
    n_docs = int(index.n_docs)
    torch_device = _resolve_torch_device(device)
    qt_scores, qt_gen, doc_total, generation = _get_gpu_stamp_state(
        n_qt, n_docs, torch_device
    )
    generation = _bump_gpu_generation(qt_gen, generation)
    doc_total.zero_()
    _GPU_STAMP_STATE[(n_qt, n_docs, str(torch_device))] = (
        qt_scores,
        qt_gen,
        doc_total,
        generation,
    )

    gpu_doc = index._gpu_compact_doc_idx
    gpu_val = index._gpu_compact_values

    from numba import cuda as numba_cuda

    qt_flat = qt_scores.view(-1)
    gen_flat = qt_gen.view(-1)
    d_qt_scores = numba_cuda.as_cuda_array(qt_flat)
    d_qt_gen = numba_cuda.as_cuda_array(gen_flat)
    d_doc_total = numba_cuda.as_cuda_array(doc_total)
    d_doc = numba_cuda.as_cuda_array(gpu_doc)
    d_val = numba_cuda.as_cuda_array(gpu_val)

    h_seg_start = np.ascontiguousarray(seg_start)
    h_seg_end = np.ascontiguousarray(seg_end)
    h_wq_off = np.ascontiguousarray(work_qt_offsets)
    h_work_seg = np.ascontiguousarray(work_seg)
    h_work_vq = np.ascontiguousarray(work_vq)
    d_seg_start = numba_cuda.to_device(h_seg_start)
    d_seg_end = numba_cuda.to_device(h_seg_end)
    d_wq_off = numba_cuda.to_device(h_wq_off)
    d_work_seg = numba_cuda.to_device(h_work_seg)
    d_work_vq = numba_cuda.to_device(h_work_vq)

    _kernel_stamp_qt_work[n_qt, 1](
        d_qt_scores,
        d_qt_gen,
        d_doc_total,
        d_doc,
        d_val,
        d_seg_start,
        d_seg_end,
        d_wq_off,
        d_work_seg,
        d_work_vq,
        generation,
        n_qt,
        n_docs,
    )
    numba_cuda.synchronize()
    tag = f"cuda-stamp-{n_qt}x{n_docs}"
    return tag, doc_total


def finalize_topk_from_doc_total_torch(
    doc_total: torch.Tensor,
    *,
    top_docs: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(doc_total.shape[0])
    k = min(int(top_docs), n)
    vals, pick = torch.topk(doc_total, k, largest=True, sorted=False)
    vals_np = vals.detach().cpu().numpy()
    pick_np = pick.detach().cpu().numpy()
    if min_score > 0.0:
        keep = vals_np > float(min_score)
    else:
        keep = vals_np > 0.0
    pick_np = pick_np[keep]
    vals_np = vals_np[keep]
    if pick_np.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    order = np.argsort(-vals_np, kind="stable")
    return pick_np[order].astype(np.int64, copy=False), vals_np[order]


def finalize_topk_from_qt_scores_gpu(
    qt_scores: torch.Tensor,
    *,
    top_docs: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = qt_scores.sum(dim=0, dtype=torch.float32)
    n = int(total.shape[0])
    k = min(int(top_docs), n)
    vals, pick = torch.topk(total, k, largest=True, sorted=False)
    vals_np = vals.detach().cpu().numpy()
    pick_np = pick.detach().cpu().numpy()
    if min_score > 0.0:
        keep = vals_np > float(min_score)
    else:
        keep = vals_np > 0.0
    pick_np = pick_np[keep]
    vals_np = vals_np[keep]
    if pick_np.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    order = np.argsort(-vals_np, kind="stable")
    return pick_np[order].astype(np.int64, copy=False), vals_np[order]


def finalize_topk_from_doc_total_gpu(
    doc_total: np.ndarray,
    *,
    top_docs: int,
    min_score: float,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray] | None:
    if not torch.cuda.is_available() or top_docs <= 0:
        return None
    torch_device = _resolve_torch_device(device)
    total = torch.from_numpy(doc_total).to(device=torch_device, dtype=torch.float32)
    n = int(total.shape[0])
    k = min(int(top_docs), n)
    vals, pick = torch.topk(total, k, largest=True, sorted=False)
    vals_np = vals.detach().cpu().numpy()
    pick_np = pick.detach().cpu().numpy()
    if min_score > 0.0:
        keep = vals_np > float(min_score)
    else:
        keep = vals_np > 0.0
    pick_np = pick_np[keep]
    vals_np = vals_np[keep]
    if pick_np.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    order = np.argsort(-vals_np, kind="stable")
    return pick_np[order].astype(np.int64, copy=False), vals_np[order]


def _scatter_hot_work_batched(
    qt_scores: torch.Tensor,
    cache,
    *,
    seg_compact_row: np.ndarray,
    work_seg: np.ndarray,
    work_qt_item: np.ndarray,
    work_vq: np.ndarray,
) -> int:
    """Batched hot scatter: one ``scatter_reduce`` per query token (avoids Python per-item loop)."""
    n_items = int(work_seg.shape[0])
    n_qt = int(qt_scores.shape[0])
    per_qt_docs: list[list[torch.Tensor]] = [[] for _ in range(n_qt)]
    per_qt_vals: list[list[torch.Tensor]] = [[] for _ in range(n_qt)]
    n_hot = 0
    for wi in range(n_items):
        si = int(work_seg[wi])
        li = int(seg_compact_row[si])
        if not cache.hot_row(li):
            continue
        qt = int(work_qt_item[wi])
        if qt < 0 or qt >= n_qt:
            continue
        n_hot += 1
        vq = float(work_vq[wi])
        g0 = int(cache.gpu_row_start[li])
        g1 = int(cache.gpu_row_end[li])
        if g1 <= g0:
            continue
        per_qt_docs[qt].append(cache.gpu_doc_idx[g0:g1])
        per_qt_vals[qt].append(cache.gpu_values[g0:g1].mul(vq))
    for qt in range(n_qt):
        if not per_qt_docs[qt]:
            continue
        docs = torch.cat(per_qt_docs[qt]).to(dtype=torch.int64)
        vals = torch.cat(per_qt_vals[qt])
        qt_scores[qt].scatter_reduce_(
            0, docs, vals, reduce="amax", include_self=True
        )
    return n_hot


def accumulate_maxsim_hybrid_from_plan(
    plan,
    index,
    *,
    n_qt: int,
    device: str = "cuda",
    budget_bytes: int = 8 * 1024**3,
    qt_scores_out: torch.Tensor | None = None,
) -> tuple[str, torch.Tensor] | None:
    """Hot latents on GPU (cached postings), cold latents on CPU Numba → ``doc_total``."""
    scan = index.compact
    work = build_maxsim_work_from_plan(
        plan, index, n_qt, scan_postings=scan, with_item_offsets=True
    )
    torch_device = _resolve_torch_device(device)
    n_docs = int(index.n_docs)
    qt_scores = qt_scores_out if qt_scores_out is not None else _get_qt_scores_buffer(
        n_qt, n_docs, torch_device
    )
    qt_scores.zero_()

    doc_total = torch.zeros(n_docs, dtype=torch.float32, device=torch_device)
    if work is None:
        return "hybrid-empty", doc_total

    cache = ensure_gpu_hot_cache(
        index,
        budget_bytes=budget_bytes,
        device=device,
    )
    if cache is None:
        return None

    (
        seg_start,
        seg_end,
        _seg_qt_offsets,
        _seg_qt,
        _seg_vq,
        _work_qt_offsets,
        work_seg,
        work_vq,
        work_qt_item,
        _item_offsets,
        seg_compact_row,
    ) = work

    n_hot_items = _scatter_hot_work_batched(
        qt_scores,
        cache,
        seg_compact_row=seg_compact_row,
        work_seg=work_seg,
        work_qt_item=work_qt_item,
        work_vq=work_vq,
    )

    cold = filter_cold_work(work, cache.is_hot_row)
    cold_qt_offsets, cold_seg, cold_vq = cold[2], cold[3], cold[4]
    n_cold = int(cold_seg.shape[0])

    if n_cold > 0:
        from .index_accumulate_fast import (
            _NUMBA_OK,
            _accumulate_stamp_qt_only,
            _bump_generation,
            _get_stamp_state,
            _sum_qt_stamp_to_doc_total,
        )

        if not _NUMBA_OK:
            raise RuntimeError("hybrid cold path requires numba")
        qt_scores_cpu, qt_gen_cold, _buf, generation_c = _get_stamp_state(n_qt, n_docs)
        generation_c = _bump_generation(qt_gen_cold, generation_c)
        doc_cold = np.zeros(n_docs, dtype=np.float32)
        _accumulate_stamp_qt_only(
            qt_scores_cpu,
            qt_gen_cold,
            scan.doc_idx,
            scan.values,
            seg_start,
            seg_end,
            cold_qt_offsets,
            cold_seg,
            cold_vq,
            generation_c,
        )
        _sum_qt_stamp_to_doc_total(
            qt_scores_cpu, qt_gen_cold, doc_cold, generation_c
        )
        doc_total.copy_(
            torch.from_numpy(doc_cold).to(
                device=torch_device, dtype=torch.float32, non_blocking=True
            )
        )

    if n_hot_items > 0:
        hot_total = qt_scores.sum(dim=0, dtype=torch.float32)
        doc_total.add_(hot_total)

    tag = (
        f"hybrid-hot{cache.n_hot_rows}-items{n_hot_items}"
        f"-cold{n_cold}-gb{cache.nbytes / (1024**3):.1f}"
    )
    return tag, doc_total


def _upload_doc_to_slot_gpu(
    doc_to_slot: np.ndarray,
    *,
    n_docs: int,
    device: torch.device,
) -> torch.Tensor:
    """Upload full-length ``doc_to_slot`` (length ``n_docs``) for global doc id → pool slot."""
    dts = np.asarray(doc_to_slot, dtype=np.int32)
    if int(dts.shape[0]) != int(n_docs):
        raise ValueError(
            f"doc_to_slot length {dts.shape[0]} != index n_docs {n_docs}"
        )
    return torch.from_numpy(dts).to(
        device=device, dtype=torch.int64, non_blocking=True
    )


def _scatter_hot_work_batched_pool(
    qt_scores: torch.Tensor,
    cache,
    doc_to_slot_gpu: torch.Tensor,
    *,
    seg_compact_row: np.ndarray,
    work_seg: np.ndarray,
    work_qt_item: np.ndarray,
    work_vq: np.ndarray,
) -> int:
    """Pool hot path: GPU cached postings → pool slots (one scatter per query token)."""
    n_items = int(work_seg.shape[0])
    n_qt = int(qt_scores.shape[0])
    per_qt_slots: list[list[torch.Tensor]] = [[] for _ in range(n_qt)]
    per_qt_vals: list[list[torch.Tensor]] = [[] for _ in range(n_qt)]
    n_hot = 0
    for wi in range(n_items):
        si = int(work_seg[wi])
        li = int(seg_compact_row[si])
        if not cache.hot_row(li):
            continue
        qt = int(work_qt_item[wi])
        if qt < 0 or qt >= n_qt:
            continue
        n_hot += 1
        vq = float(work_vq[wi])
        g0 = int(cache.gpu_row_start[li])
        g1 = int(cache.gpu_row_end[li])
        if g1 <= g0:
            continue
        seg_docs = cache.gpu_doc_idx[g0:g1]
        seg_vals = cache.gpu_values[g0:g1]
        if seg_docs.numel() == 0 or seg_docs.numel() != seg_vals.numel():
            continue
        docs = seg_docs.to(dtype=torch.int64)
        slots = doc_to_slot_gpu[docs]
        keep = slots >= 0
        if not keep.any():
            continue
        slots = slots[keep]
        vals = seg_vals[keep].mul(vq)
        per_qt_slots[qt].append(slots)
        per_qt_vals[qt].append(vals)
    for qt in range(n_qt):
        if not per_qt_slots[qt]:
            continue
        slots = torch.cat(per_qt_slots[qt])
        vals = torch.cat(per_qt_vals[qt])
        qt_scores[qt].scatter_reduce_(
            0, slots, vals, reduce="amax", include_self=True
        )
    return n_hot


def accumulate_maxsim_gpu_pool_hybrid_from_plan(
    plan,
    index,
    *,
    n_qt: int,
    doc_to_slot: np.ndarray,
    pool_candidate_doc_ids: np.ndarray,
    device: str = "cuda",
    budget_bytes: int = 8 * 1024**3,
    scan_postings=None,
) -> tuple[str, torch.Tensor] | None:
    """Pool MaxSim: GPU hot scatter by slot + CPU cold cand-merge (small ``n_pool``)."""
    from .compact_postings import CompactPostings

    scan: CompactPostings = (
        scan_postings if scan_postings is not None else index.compact
    )
    n_docs = int(index.n_docs)
    n_pool = int(pool_candidate_doc_ids.shape[0])
    if n_pool <= 0:
        return None
    torch_device = _resolve_torch_device(device)
    work = build_maxsim_work_from_plan(
        plan, index, n_qt, scan_postings=scan, with_item_offsets=True
    )
    doc_total_gpu = torch.zeros(n_pool, dtype=torch.float32, device=torch_device)
    if work is None:
        return "gpu-pool-empty", doc_total_gpu

    cand = np.asarray(pool_candidate_doc_ids, dtype=np.int32)
    filtered = filter_work_by_candidate_range(
        work,
        compact_doc=scan.doc_idx,
        cand_min=int(cand.min()),
        cand_max=int(cand.max()),
    )
    if filtered is None:
        return "gpu-pool-empty", doc_total_gpu
    work = filtered

    cache = ensure_gpu_hot_cache(
        index,
        budget_bytes=budget_bytes,
        device=device,
    )
    if cache is None:
        return None

    (
        seg_start,
        seg_end,
        _seg_qt_offsets,
        _seg_qt,
        _seg_vq,
        _work_qt_offsets,
        work_seg,
        work_vq,
        work_qt_item,
        _item_offsets,
        seg_compact_row,
    ) = work

    qt_scores = _get_qt_scores_buffer(n_qt, n_pool, torch_device)
    qt_scores.zero_()
    cold_vals = scan.values
    if cold_vals.dtype != np.float32:
        cold_vals = cold_vals.astype(np.float32, copy=False)

    doc_to_slot_gpu = _upload_doc_to_slot_gpu(
        doc_to_slot, n_docs=n_docs, device=torch_device
    )
    n_hot_items = _scatter_hot_work_batched_pool(
        qt_scores,
        cache,
        doc_to_slot_gpu,
        seg_compact_row=seg_compact_row,
        work_seg=work_seg,
        work_qt_item=work_qt_item,
        work_vq=work_vq,
    )

    cold = filter_cold_work(work, cache.is_hot_row)
    cold_qt_offsets, cold_seg, cold_vq = cold[2], cold[3], cold[4]
    n_cold = int(cold_seg.shape[0])

    if n_cold > 0:
        from .index_accumulate_fast import (
            _NUMBA_OK,
            _accumulate_stamp_qt_cand_merge,
            _bump_generation,
            _get_pool_stamp_state,
            _sum_qt_stamp_to_pool_total,
        )

        if not _NUMBA_OK:
            return None
        qt_scores_cpu, qt_gen, _gen = _get_pool_stamp_state(n_qt, n_pool)
        generation = _bump_generation(qt_gen, 0)
        doc_cold = np.zeros(n_pool, dtype=np.float32)
        cand_sorted = np.sort(pool_candidate_doc_ids.astype(np.int32, copy=False))
        _accumulate_stamp_qt_cand_merge(
            qt_scores_cpu,
            qt_gen,
            scan.doc_idx,
            cold_vals,
            doc_to_slot,
            cand_sorted,
            seg_start,
            seg_end,
            cold_qt_offsets,
            cold_seg,
            cold_vq,
            generation,
        )
        _sum_qt_stamp_to_pool_total(qt_scores_cpu, qt_gen, doc_cold, generation)
        doc_total_gpu.add_(
            torch.from_numpy(doc_cold).to(
                device=torch_device, dtype=torch.float32, non_blocking=True
            )
        )

    if n_hot_items > 0:
        doc_total_gpu.add_(qt_scores.sum(dim=0, dtype=torch.float32))

    tag = (
        f"gpu-pool-hot{cache.n_hot_rows}-items{n_hot_items}"
        f"-cold{n_cold}-pool{n_pool}"
    )
    return tag, doc_total_gpu


def finalize_topk_pool_from_doc_total_gpu(
    doc_total: torch.Tensor,
    candidate_doc_ids: np.ndarray,
    *,
    top_docs: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k over pool slots, map slot indices to global doc ids."""
    n_pool = int(doc_total.shape[0])
    k = min(int(top_docs), n_pool)
    vals, pick = torch.topk(doc_total, k, largest=True, sorted=False)
    vals_np = vals.detach().cpu().numpy()
    pick_np = pick.detach().cpu().numpy().astype(np.int64, copy=False)
    cand = np.asarray(candidate_doc_ids, dtype=np.int64)
    if min_score > 0.0:
        keep = vals_np > float(min_score)
    else:
        keep = vals_np > 0.0
    pick_np = pick_np[keep]
    vals_np = vals_np[keep]
    if pick_np.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    global_ids = cand[pick_np]
    order = np.argsort(-vals_np, kind="stable")
    return global_ids[order], vals_np[order]


def try_accumulate_gpu_pool_maxsim_from_plan(
    plan,
    index,
    *,
    n_qt: int,
    doc_to_slot: np.ndarray,
    pool_candidate_doc_ids: np.ndarray,
    index_accum_device: IndexAccumDevice = "auto",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
    scan_postings=None,
    top_docs: int = 0,
    min_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GPU pool two-phase phase-2; returns ``(doc_ids, scores)`` or ``None`` (CPU fallback)."""
    resolved = resolve_index_accum_device(index_accum_device)
    if resolved == "cpu" or not torch.cuda.is_available():
        return None
    dev = cuda_device if cuda_device.startswith("cuda") else "cuda"
    budget_bytes = int(max(0.5, gpu_hot_budget_gb) * (1024**3))
    try:
        out = accumulate_maxsim_gpu_pool_hybrid_from_plan(
            plan,
            index,
            n_qt=n_qt,
            doc_to_slot=doc_to_slot,
            pool_candidate_doc_ids=pool_candidate_doc_ids,
            device=dev,
            budget_bytes=budget_bytes,
            scan_postings=scan_postings,
        )
        if out is None:
            return None
        _tag, doc_total_gpu = out
        return finalize_topk_pool_from_doc_total_gpu(
            doc_total_gpu,
            pool_candidate_doc_ids,
            top_docs=top_docs,
            min_score=min_score,
        )
    except Exception as exc:
        logger.warning("GPU pool accumulate failed (%s); CPU fallback", exc)
        return None


def try_accumulate_pool_doc_total_from_plan(
    plan,
    index,
    *,
    n_qt: int,
    doc_total: np.ndarray,
    doc_to_slot: np.ndarray,
    pool_candidate_doc_ids: np.ndarray,
    index_accum_device: IndexAccumDevice = "auto",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
    scan_postings=None,
) -> str | None:
    """Fill ``doc_total`` (pool slots) via GPU hybrid pool path; ``None`` → CPU fallback."""
    resolved = resolve_index_accum_device(index_accum_device)
    if resolved == "cpu" or not torch.cuda.is_available():
        return None
    dev = cuda_device if cuda_device.startswith("cuda") else "cuda"
    budget_bytes = int(max(0.5, gpu_hot_budget_gb) * (1024**3))
    try:
        cache = ensure_gpu_hot_cache(
            index,
            budget_bytes=budget_bytes,
            device=dev,
        )
        if cache is not None and cache.hot_posting_fraction >= 0.90:
            return None
        out = accumulate_maxsim_gpu_pool_hybrid_from_plan(
            plan,
            index,
            n_qt=n_qt,
            doc_to_slot=doc_to_slot,
            pool_candidate_doc_ids=pool_candidate_doc_ids,
            device=dev,
            budget_bytes=budget_bytes,
            scan_postings=scan_postings,
        )
        if out is None:
            return None
        tag, doc_total_gpu = out
        np.copyto(
            doc_total,
            doc_total_gpu.detach().cpu().numpy(),
            casting="unsafe",
        )
        return tag
    except Exception as exc:
        logger.warning("GPU pool doc_total failed (%s); CPU fallback", exc)
        return None


def try_accumulate_gpu_from_plan(
    plan,
    index,
    *,
    n_qt: int,
    doc_total: np.ndarray | None,
    index_accum_device: IndexAccumDevice = "auto",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
) -> tuple[str, torch.Tensor] | None:
    """Return ``(tag, doc_total_or_qt_gpu)`` on success; ``None`` for CPU fallback."""
    resolved = resolve_index_accum_device(index_accum_device)
    if resolved == "cpu":
        return None
    dev = cuda_device if cuda_device.startswith("cuda") else "cuda"
    if not torch.cuda.is_available():
        return None
    budget_bytes = int(max(0.5, gpu_hot_budget_gb) * (1024**3))
    n_docs = int(index.n_docs)
    # Small corpora: optional full-GPU CUDA stamp (hybrid hot-cache is empty on tiny indexes).
    if _CUDA_STAMP_OK and n_docs < 100_000:
        compact_budget = max(budget_bytes, 22 * 1024**3)
        stamp_out = accumulate_maxsim_gpu_stamp_from_plan(
            plan,
            index,
            n_qt=n_qt,
            device=dev,
            budget_bytes=compact_budget,
        )
        if stamp_out is not None:
            return stamp_out
    try:
        hybrid_out = accumulate_maxsim_hybrid_from_plan(
            plan,
            index,
            n_qt=n_qt,
            device=dev,
            budget_bytes=budget_bytes,
        )
        if hybrid_out is None:
            return None
        return hybrid_out
    except Exception as exc:
        logger.warning("GPU index accumulate failed (%s); falling back to CPU", exc)
        return None
