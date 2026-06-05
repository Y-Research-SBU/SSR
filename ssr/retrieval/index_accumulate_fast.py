"""Fast MaxSim posting accumulation (Numba)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .compact_postings import CompactPostings
    from .inverted_index import BlockInvertedIndex

if "NUMBA_NUM_THREADS" not in os.environ:
    os.environ["NUMBA_NUM_THREADS"] = str(min(32, os.cpu_count() or 1))

_NUMBA_OK = False
try:
    from numba import njit, prange

    _NUMBA_OK = True
except ImportError:
    prange = range  # type: ignore[misc, assignment]


def resolve_index_parallel_workers(requested: int) -> int:
    """0/1 -> single qt-parallel path; >=2 enables experimental segment workers."""
    if requested <= 1:
        return 1
    return int(requested)


if _NUMBA_OK:

    @njit(cache=True, fastmath=True)
    def _apply_contrib(
        doc_total: np.ndarray,
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        d: int,
        contrib: float,
        qt: int,
        generation: int,
    ) -> None:
        if qt_gen[d] != generation:
            qt_gen[d] = generation
            qt_scores[d] = contrib
            doc_total[d] += contrib
        elif contrib > qt_scores[d]:
            doc_total[d] += contrib - qt_scores[d]
            qt_scores[d] = contrib

    @njit(cache=True, fastmath=True)
    def _update_stamp_qt_only_from_segment(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        s: int,
        e: int,
        v_q: float,
        generation: int,
    ) -> None:
        n = e - s
        if n <= 0:
            return
        d0 = int(doc_idx[s])
        vmax = np.float32(values[s]) * np.float32(v_q)
        for i in range(s + 1, e):
            d = int(doc_idx[i])
            v = np.float32(values[i]) * np.float32(v_q)
            if d == d0:
                if v > vmax:
                    vmax = v
            else:
                _apply_contrib_qt_only(qt_scores, qt_gen, d0, vmax, generation)
                d0 = d
                vmax = v
        _apply_contrib_qt_only(qt_scores, qt_gen, d0, vmax, generation)

    @njit(cache=True, fastmath=True)
    def _apply_contrib_qt_only(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        d: int,
        contrib: float,
        generation: int,
    ) -> None:
        if qt_gen[d] != generation:
            qt_gen[d] = generation
            qt_scores[d] = contrib
        elif contrib > qt_scores[d]:
            qt_scores[d] = contrib

    @njit(cache=True, fastmath=True)
    def _apply_contrib_qt_slot(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        slot: int,
        contrib: float,
        generation: int,
    ) -> None:
        if qt_gen[slot] != generation:
            qt_gen[slot] = generation
            qt_scores[slot] = contrib
        elif contrib > qt_scores[slot]:
            qt_scores[slot] = contrib

    @njit(cache=True, fastmath=True)
    def _update_stamp_qt_slot_from_segment(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        doc_to_slot: np.ndarray,
        s: int,
        e: int,
        v_q: float,
        generation: int,
    ) -> None:
        n = e - s
        if n <= 0:
            return
        d0 = int(doc_idx[s])
        slot0 = int(doc_to_slot[d0])
        vmax = np.float32(values[s]) * np.float32(v_q)
        for i in range(s + 1, e):
            d = int(doc_idx[i])
            v = np.float32(values[i]) * np.float32(v_q)
            if d == d0:
                if v > vmax:
                    vmax = v
            else:
                if slot0 >= 0:
                    _apply_contrib_qt_slot(qt_scores, qt_gen, slot0, vmax, generation)
                d0 = d
                slot0 = int(doc_to_slot[d0])
                vmax = v
        if slot0 >= 0:
            _apply_contrib_qt_slot(qt_scores, qt_gen, slot0, vmax, generation)

    @njit(cache=True, fastmath=True)
    def _update_stamp_qt_cand_merge_segment(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        doc_to_slot: np.ndarray,
        cand_docs_sorted: np.ndarray,
        s: int,
        e: int,
        v_q: float,
        generation: int,
    ) -> None:
        """Merge-join segment (sorted by doc) with sorted candidate ids; skip non-candidate runs."""
        n_cand = int(cand_docs_sorted.shape[0])
        if n_cand == 0 or e <= s:
            return
        ci = 0
        i = int(s)
        vq = np.float32(v_q)
        while i < e and ci < n_cand:
            d_seg = int(doc_idx[i])
            d_cand = int(cand_docs_sorted[ci])
            if d_cand < d_seg:
                ci += 1
            elif d_seg < d_cand:
                while i < e and int(doc_idx[i]) == d_seg:
                    i += 1
            else:
                slot0 = int(doc_to_slot[d_seg])
                if slot0 >= 0:
                    run_end = i + 1
                    while run_end < e and int(doc_idx[run_end]) == d_seg:
                        run_end += 1
                    vmax = np.float32(values[i]) * vq
                    for j in range(i + 1, run_end):
                        v = np.float32(values[j]) * vq
                        if v > vmax:
                            vmax = v
                    _apply_contrib_qt_slot(
                        qt_scores, qt_gen, slot0, vmax, generation
                    )
                    i = run_end
                else:
                    while i < e and int(doc_idx[i]) == d_seg:
                        i += 1
                ci += 1

    @njit(parallel=True, cache=True, fastmath=True)
    def _accumulate_stamp_qt_cand_merge(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        doc_to_slot: np.ndarray,
        cand_docs_sorted: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        work_qt_offsets: np.ndarray,
        work_seg: np.ndarray,
        work_vq: np.ndarray,
        generation: int,
    ) -> None:
        n_qt = int(work_qt_offsets.shape[0]) - 1
        for qt in prange(n_qt):
            row_s = qt_scores[qt]
            row_g = qt_gen[qt]
            wb = int(work_qt_offsets[qt])
            we = int(work_qt_offsets[qt + 1])
            for wi in range(wb, we):
                si = int(work_seg[wi])
                vq = float(work_vq[wi])
                s = int(seg_start[si])
                e = int(seg_end[si])
                _update_stamp_qt_cand_merge_segment(
                    row_s,
                    row_g,
                    doc_idx,
                    values,
                    doc_to_slot,
                    cand_docs_sorted,
                    s,
                    e,
                    vq,
                    generation,
                )

    @njit(parallel=True, cache=True, fastmath=True)
    def _sum_qt_stamp_to_doc_total(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_total: np.ndarray,
        generation: int,
    ) -> None:
        """Sum per-qt rows into ``doc_total`` without zeroing the qt matrix."""
        n_qt = int(qt_scores.shape[0])
        n_docs = int(qt_scores.shape[1])
        for d in prange(n_docs):
            acc = 0.0
            for qt in range(n_qt):
                if qt_gen[qt, d] == generation:
                    acc += qt_scores[qt, d]
            doc_total[d] = acc

    @njit(parallel=True, cache=True, fastmath=True)
    def _accumulate_stamp_segment_workers(
        qt_scores_w: np.ndarray,
        qt_gen_w: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        seg_qt_offsets: np.ndarray,
        seg_qt: np.ndarray,
        seg_vq: np.ndarray,
        generation: int,
        n_workers: int,
    ) -> None:
        n_seg = int(seg_start.shape[0])
        for w in prange(n_workers):
            qt_scores = qt_scores_w[w]
            qt_gen = qt_gen_w[w]
            for si in range(w, n_seg, n_workers):
                s = int(seg_start[si])
                e = int(seg_end[si])
                q0 = int(seg_qt_offsets[si])
                q1 = int(seg_qt_offsets[si + 1])
                for j in range(q0, q1):
                    qt = int(seg_qt[j])
                    vq = float(seg_vq[j])
                    _update_stamp_qt_only_from_segment(
                        qt_scores[qt],
                        qt_gen[qt],
                        doc_idx,
                        values,
                        s,
                        e,
                        vq,
                        generation,
                    )

    @njit(cache=True, fastmath=True)
    def _update_stamp_from_segment(
        doc_total: np.ndarray,
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        s: int,
        e: int,
        v_q: float,
        qt: int,
        generation: int,
    ) -> None:
        n = e - s
        if n <= 0:
            return
        d0 = int(doc_idx[s])
        vmax = np.float32(values[s]) * np.float32(v_q)
        for i in range(s + 1, e):
            d = int(doc_idx[i])
            v = np.float32(values[i]) * np.float32(v_q)
            if d == d0:
                if v > vmax:
                    vmax = v
            else:
                _apply_contrib(
                    doc_total, qt_scores, qt_gen, d0, vmax, qt, generation
                )
                d0 = d
                vmax = v
        _apply_contrib(doc_total, qt_scores, qt_gen, d0, vmax, qt, generation)

    @njit(parallel=True, cache=True, fastmath=True)
    def _accumulate_stamp_qt_slot(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        doc_to_slot: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        work_qt_offsets: np.ndarray,
        work_seg: np.ndarray,
        work_vq: np.ndarray,
        generation: int,
    ) -> None:
        n_qt = int(work_qt_offsets.shape[0]) - 1
        for qt in prange(n_qt):
            row_s = qt_scores[qt]
            row_g = qt_gen[qt]
            wb = int(work_qt_offsets[qt])
            we = int(work_qt_offsets[qt + 1])
            for wi in range(wb, we):
                si = int(work_seg[wi])
                vq = float(work_vq[wi])
                s = int(seg_start[si])
                e = int(seg_end[si])
                _update_stamp_qt_slot_from_segment(
                    row_s,
                    row_g,
                    doc_idx,
                    values,
                    doc_to_slot,
                    s,
                    e,
                    vq,
                    generation,
                )

    @njit(parallel=True, cache=True, fastmath=True)
    def _sum_qt_stamp_to_pool_total(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_total: np.ndarray,
        generation: int,
    ) -> None:
        n_qt = int(qt_scores.shape[0])
        n_pool = int(doc_total.shape[0])
        for slot in prange(n_pool):
            acc = 0.0
            for qt in range(n_qt):
                if qt_gen[qt, slot] == generation:
                    acc += qt_scores[qt, slot]
            doc_total[slot] = acc

    @njit(parallel=True, cache=True, fastmath=True)
    def _accumulate_stamp_qt_only(
        qt_scores: np.ndarray,
        qt_gen: np.ndarray,
        doc_idx: np.ndarray,
        values: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        work_qt_offsets: np.ndarray,
        work_seg: np.ndarray,
        work_vq: np.ndarray,
        generation: int,
    ) -> None:
        """Per-qt max accumulation (parallel-safe); sum rows into ``doc_total`` in Python."""
        n_qt = int(work_qt_offsets.shape[0]) - 1
        for qt in prange(n_qt):
            row_s = qt_scores[qt]
            row_g = qt_gen[qt]
            wb = int(work_qt_offsets[qt])
            we = int(work_qt_offsets[qt + 1])
            for wi in range(wb, we):
                si = int(work_seg[wi])
                vq = float(work_vq[wi])
                s = int(seg_start[si])
                e = int(seg_end[si])
                _update_stamp_qt_only_from_segment(
                    row_s, row_g, doc_idx, values, s, e, vq, generation
                )

_STAMP_STATE: dict[
    tuple[int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray, int],
] = {}
_POOL_STAMP_STATE: dict[
    tuple[int, int],
    tuple[np.ndarray, np.ndarray, int],
] = {}
_WORKER_STATE: dict[
    tuple[int, int, int], tuple[np.ndarray, np.ndarray, int]
] = {}
_WORK_BUILD_LISTS: dict[str, list] | None = None


def _work_build_lists() -> dict[str, list]:
    global _WORK_BUILD_LISTS
    if _WORK_BUILD_LISTS is None:
        _WORK_BUILD_LISTS = {
            "seg_start": [],
            "seg_end": [],
            "seg_row": [],
            "seg_qt": [],
            "seg_vq": [],
            "seg_qt_offsets": [0],
            "qt": [],
            "si": [],
            "vq": [],
        }
    for key, buf in _WORK_BUILD_LISTS.items():
        if key == "seg_qt_offsets":
            buf.clear()
            buf.append(0)
        else:
            buf.clear()
    return _WORK_BUILD_LISTS


def _get_worker_state(
    n_workers: int, n_qt: int, n_docs: int
) -> tuple[np.ndarray, np.ndarray, int]:
    key = (n_workers, n_qt, n_docs)
    st = _WORKER_STATE.get(key)
    if st is None:
        qt_scores_w = np.zeros((n_workers, n_qt, n_docs), dtype=np.float32)
        qt_gen_w = np.zeros((n_workers, n_qt, n_docs), dtype=np.int32)
        st = (qt_scores_w, qt_gen_w, 1)
        _WORKER_STATE[key] = st
    return st


def merge_worker_qt_to_doc_total(
    qt_scores_w: np.ndarray, doc_total: np.ndarray
) -> None:
    merged = np.max(qt_scores_w, axis=0)
    np.sum(merged, axis=0, dtype=np.float32, out=doc_total)


def _get_stamp_state(
    n_qt: int, n_docs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    key = (n_qt, n_docs)
    st = _STAMP_STATE.get(key)
    if st is None:
        qt_scores = np.zeros((n_qt, n_docs), dtype=np.float32)
        qt_gen = np.zeros((n_qt, n_docs), dtype=np.int32)
        doc_total = np.zeros(n_docs, dtype=np.float32)
        st = (qt_scores, qt_gen, doc_total, 1)
        _STAMP_STATE[key] = st
    return st


def _get_pool_stamp_state(
    n_qt: int, n_pool: int
) -> tuple[np.ndarray, np.ndarray, int]:
    key = (n_qt, n_pool)
    st = _POOL_STAMP_STATE.get(key)
    if st is None:
        qt_scores = np.zeros((n_qt, n_pool), dtype=np.float32)
        qt_gen = np.zeros((n_qt, n_pool), dtype=np.int32)
        st = (qt_scores, qt_gen, 1)
        _POOL_STAMP_STATE[key] = st
    return st


def _bump_generation(gen_arr: np.ndarray, generation: int) -> int:
    generation += 1
    if generation > 2_000_000_000:
        gen_arr.fill(0)
        generation = 1
    return generation


def pack_segments_flat(
    segments: list[tuple[list[tuple[int, float]], int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Columnar segment descriptor (``seg_start`` / ``seg_end`` / per-seg qt lists)."""
    n_seg = len(segments)
    seg_start = np.empty(n_seg, dtype=np.int64)
    seg_end = np.empty(n_seg, dtype=np.int64)
    seg_qt_offsets = np.zeros(n_seg + 1, dtype=np.int64)
    qt_list: list[int] = []
    vq_list: list[float] = []
    for si, (q_entries, s, e) in enumerate(segments):
        seg_start[si] = s
        seg_end[si] = e
        seg_qt_offsets[si] = len(qt_list)
        for qt, v_q in q_entries:
            qt_list.append(int(qt))
            vq_list.append(float(v_q))
    seg_qt_offsets[n_seg] = len(qt_list)
    return (
        seg_start,
        seg_end,
        seg_qt_offsets,
        np.asarray(qt_list, dtype=np.int32),
        np.asarray(vq_list, dtype=np.float32),
    )


def _pack_work_by_qt_np(
    qt_arr: np.ndarray,
    seg_arr: np.ndarray,
    vq_arr: np.ndarray,
    n_qt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort work items by ``qt``; returned ``work_qt`` aligns with ``work_seg`` / ``work_vq``."""
    if qt_arr.size == 0:
        return (
            np.zeros(n_qt + 1, dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
        )
    order = np.argsort(qt_arr, kind="stable")
    seg_arr = seg_arr[order]
    vq_arr = vq_arr[order]
    qt_arr = qt_arr[order]
    counts = np.bincount(qt_arr, minlength=n_qt)
    work_qt_offsets = np.zeros(n_qt + 1, dtype=np.int64)
    work_qt_offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return (
        work_qt_offsets,
        seg_arr.astype(np.int32, copy=False),
        vq_arr.astype(np.float32, copy=False),
        qt_arr.astype(np.int32, copy=False),
    )


def build_maxsim_work_from_plan(
    plan,
    index,
    n_qt: int,
    *,
    scan_postings: CompactPostings | None = None,
    with_item_offsets: bool = False,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
):
    """One-pass plan -> segment + per-qt work arrays."""
    scan = scan_postings if scan_postings is not None else index.compact
    row_index = index.latent_row_index
    offsets = scan.offsets
    buf = _work_build_lists()
    seg_start_list = buf["seg_start"]
    seg_end_list = buf["seg_end"]
    seg_row_list = buf["seg_row"]
    seg_qt_list = buf["seg_qt"]
    seg_vq_list = buf["seg_vq"]
    seg_qt_offsets_list = buf["seg_qt_offsets"]
    qt_list = buf["qt"]
    si_list = buf["si"]
    vq_list = buf["vq"]
    si = 0
    for block_id in plan.active_block_ids:
        q_latents = plan.entries_by_block[block_id]
        for latent, q_entries in q_latents.items():
            if row_index is not None:
                li = int(row_index[int(latent)])
            else:
                li = scan.latent_index(int(latent))
            if li < 0:
                continue
            s, e = int(offsets[li]), int(offsets[li + 1])
            if e <= s:
                continue
            seg_start_list.append(s)
            seg_end_list.append(e)
            seg_row_list.append(li)
            for qt, v_q in q_entries:
                if 0 <= qt < n_qt:
                    seg_qt_list.append(qt)
                    seg_vq_list.append(v_q)
                    qt_list.append(qt)
                    si_list.append(si)
                    vq_list.append(v_q)
            seg_qt_offsets_list.append(len(seg_qt_list))
            si += 1
    if not seg_start_list:
        return None
    qt_arr = np.asarray(qt_list, dtype=np.int32)
    si_arr = np.asarray(si_list, dtype=np.int32)
    vq_arr = np.asarray(vq_list, dtype=np.float32)
    work_qt_offsets, work_seg, work_vq, work_qt_item = _pack_work_by_qt_np(
        qt_arr, si_arr, vq_arr, n_qt
    )
    seg_start_a = np.asarray(seg_start_list, dtype=np.int64)
    seg_end_a = np.asarray(seg_end_list, dtype=np.int64)
    seg_row_a = np.asarray(seg_row_list, dtype=np.int32)
    if with_item_offsets:
        n_items = int(work_seg.shape[0])
        item_offsets = np.zeros(n_items + 1, dtype=np.int64)
        for wi in range(n_items):
            si_i = int(work_seg[wi])
            item_offsets[wi + 1] = item_offsets[wi] + (
                int(seg_end_a[si_i]) - int(seg_start_a[si_i])
            )
    else:
        item_offsets = np.zeros(1, dtype=np.int64)
    return (
        seg_start_a,
        seg_end_a,
        np.asarray(seg_qt_offsets_list, dtype=np.int64),
        np.asarray(seg_qt_list, dtype=np.int32),
        np.asarray(seg_vq_list, dtype=np.float32),
        work_qt_offsets,
        work_seg,
        work_vq,
        work_qt_item,
        item_offsets,
        seg_row_a,
    )


def filter_work_by_candidate_range(
    work: tuple,
    compact_doc: np.ndarray,
    cand_min: int,
    cand_max: int,
) -> tuple | None:
    """Drop work items whose posting segment cannot intersect ``[cand_min, cand_max]``."""
    (
        seg_start,
        seg_end,
        _seg_qt_offsets,
        _seg_qt,
        _seg_vq,
        work_qt_offsets,
        work_seg,
        work_vq,
        work_qt_item,
        _item_offsets,
        _seg_compact_row,
    ) = work
    n_items = int(work_seg.shape[0])
    kept: list[int] = []
    for wi in range(n_items):
        si = int(work_seg[wi])
        s, e = int(seg_start[si]), int(seg_end[si])
        if e <= s:
            continue
        d0 = int(compact_doc[s])
        d1 = int(compact_doc[e - 1])
        if d1 < cand_min or d0 > cand_max:
            continue
        kept.append(wi)
    if not kept:
        return None
    if len(kept) == n_items:
        return work
    keep_wi = np.asarray(kept, dtype=np.int32)
    keep_qt = work_qt_item[keep_wi]
    keep_seg = work_seg[keep_wi]
    keep_vq = work_vq[keep_wi]
    n_qt = int(work_qt_offsets.shape[0]) - 1
    k_qt_offsets, k_seg, k_vq, _k_qt = _pack_work_by_qt_np(
        keep_qt, keep_seg, keep_vq, n_qt
    )
    return (
        seg_start,
        seg_end,
        _seg_qt_offsets,
        _seg_qt,
        _seg_vq,
        k_qt_offsets,
        k_seg,
        k_vq,
        keep_qt,
        _item_offsets,
        _seg_compact_row,
    )


def filter_cold_work(
    work: tuple,
    is_hot_row: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Keep only work items whose segment compact row is not on GPU."""
    (
        seg_start,
        seg_end,
        _seg_qt_offsets,
        _seg_qt,
        _seg_vq,
        work_qt_offsets,
        work_seg,
        work_vq,
        work_qt_item,
        item_offsets,
        seg_compact_row,
    ) = work
    n_items = int(work_seg.shape[0])
    cold_items = [
        wi
        for wi in range(n_items)
        if not bool(is_hot_row[int(seg_compact_row[int(work_seg[wi])])])
    ]
    if not cold_items:
        n_qt = int(work_qt_offsets.shape[0]) - 1
        return (
            seg_start,
            seg_end,
            np.zeros(n_qt + 1, dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            work_qt_offsets,
            work_seg,
            work_vq,
        )
    cold_wi = np.asarray(cold_items, dtype=np.int32)
    cold_qt = work_qt_item[cold_wi]
    cold_seg = work_seg[cold_wi]
    cold_vq = work_vq[cold_wi]
    n_qt = int(work_qt_offsets.shape[0]) - 1
    cold_qt_offsets, cold_seg_i, cold_vq_i, _cold_qt_sorted = _pack_work_by_qt_np(
        cold_qt, cold_seg, cold_vq, n_qt
    )
    return (
        seg_start,
        seg_end,
        cold_qt_offsets,
        cold_seg_i,
        cold_vq_i,
        work_qt_offsets,
        work_seg,
        work_vq,
    )


def accumulate_maxsim_from_plan(
    plan,
    index,
    per_qt: np.ndarray | None,
    *,
    doc_total: np.ndarray | None = None,
    doc_to_slot: np.ndarray | None = None,
    pool_candidate_doc_ids: np.ndarray | None = None,
    n_query_tokens: int = 0,
    parallel_workers: int = 0,
    index_accum_device: str = "cpu",
    cuda_device: str = "cuda",
    gpu_hot_budget_gb: float = 8.0,
    scan_postings: CompactPostings | None = None,
) -> str:
    if per_qt is not None:
        n_qt = int(per_qt.shape[0])
    else:
        n_qt = int(n_query_tokens)
    scan = scan_postings if scan_postings is not None else index.compact
    work = build_maxsim_work_from_plan(plan, index, n_qt, scan_postings=scan)
    if work is None:
        return "empty"
    (
        seg_start,
        seg_end,
        seg_qt_offsets,
        seg_qt,
        seg_vq,
        work_qt_offsets,
        work_seg,
        work_vq,
        _work_qt_item,
        _item_offsets,
        _seg_compact_row,
    ) = work
    if (
        pool_candidate_doc_ids is not None
        and int(pool_candidate_doc_ids.shape[0]) > 0
        and doc_to_slot is not None
    ):
        cand = np.asarray(pool_candidate_doc_ids, dtype=np.int32)
        filtered = filter_work_by_candidate_range(
            work,
            compact_doc=scan.doc_idx,
            cand_min=int(cand.min()),
            cand_max=int(cand.max()),
        )
        if filtered is None:
            return "empty"
        work = filtered
        (
            seg_start,
            seg_end,
            seg_qt_offsets,
            seg_qt,
            seg_vq,
            work_qt_offsets,
            work_seg,
            work_vq,
            _work_qt_item,
            _item_offsets,
            _seg_compact_row,
        ) = work
    if not (_NUMBA_OK and doc_total is not None):
        from .inverted_index import _query_latent_segments

        segments = _query_latent_segments(
            plan, scan, latent_row_index=index.latent_row_index
        )
        return accumulate_maxsim_fast(
            per_qt,
            scan.doc_idx,
            scan.values,
            segments,
            doc_total=doc_total,
            n_query_tokens=n_qt,
        )
    compact_doc = scan.doc_idx
    compact_val = scan.values
    if compact_val.dtype != np.float32:
        compact_val = compact_val.astype(np.float32, copy=False)

    if doc_to_slot is not None:
        n_pool = int(doc_total.shape[0])
        if (
            pool_candidate_doc_ids is not None
            and int(pool_candidate_doc_ids.shape[0]) > 0
            and index_accum_device in ("cuda", "auto", "hybrid")
        ):
            from .index_accumulate_gpu import try_accumulate_pool_doc_total_from_plan

            gpu_tag = try_accumulate_pool_doc_total_from_plan(
                plan,
                index,
                n_qt=n_qt,
                doc_total=doc_total,
                doc_to_slot=doc_to_slot,
                pool_candidate_doc_ids=pool_candidate_doc_ids,
                index_accum_device=index_accum_device,  # type: ignore[arg-type]
                cuda_device=cuda_device,
                gpu_hot_budget_gb=gpu_hot_budget_gb,
                scan_postings=scan,
            )
            if gpu_tag is not None:
                return gpu_tag
        qt_scores, qt_gen, generation = _get_pool_stamp_state(n_qt, n_pool)
        generation = _bump_generation(qt_gen, generation)
        _POOL_STAMP_STATE[(n_qt, n_pool)] = (qt_scores, qt_gen, generation)
        use_cand_merge = (
            _NUMBA_OK
            and pool_candidate_doc_ids is not None
            and int(pool_candidate_doc_ids.shape[0]) > 0
        )
        if use_cand_merge:
            cand_sorted = np.ascontiguousarray(
                np.sort(pool_candidate_doc_ids.astype(np.int32, copy=False))
            )
            _accumulate_stamp_qt_cand_merge(
                qt_scores,
                qt_gen,
                compact_doc,
                compact_val,
                doc_to_slot,
                cand_sorted,
                seg_start,
                seg_end,
                work_qt_offsets,
                work_seg,
                work_vq,
                generation,
            )
            tag = f"numba-stamp-pool-merge-{n_pool}"
        else:
            _accumulate_stamp_qt_slot(
                qt_scores,
                qt_gen,
                compact_doc,
                compact_val,
                doc_to_slot,
                seg_start,
                seg_end,
                work_qt_offsets,
                work_seg,
                work_vq,
                generation,
            )
            tag = f"numba-stamp-pool-{n_pool}"
        _sum_qt_stamp_to_pool_total(qt_scores, qt_gen, doc_total, generation)
        return tag

    n_docs = int(doc_total.shape[0])
    n_workers = resolve_index_parallel_workers(parallel_workers)

    if n_workers > 1:
        qt_scores_w, qt_gen_w, generation = _get_worker_state(n_workers, n_qt, n_docs)
        generation = _bump_generation(qt_gen_w, generation)
        _WORKER_STATE[(n_workers, n_qt, n_docs)] = (
            qt_scores_w,
            qt_gen_w,
            generation,
        )
        doc_total.fill(0.0)
        _accumulate_stamp_segment_workers(
            qt_scores_w,
            qt_gen_w,
            compact_doc,
            compact_val,
            seg_start,
            seg_end,
            seg_qt_offsets,
            seg_qt,
            seg_vq,
            generation,
            n_workers,
        )
        merge_worker_qt_to_doc_total(qt_scores_w, doc_total)
        return f"numba-workers-{n_workers}"

    qt_scores, qt_gen, _buf_total, generation = _get_stamp_state(n_qt, n_docs)
    generation = _bump_generation(qt_gen, generation)
    _STAMP_STATE[(n_qt, n_docs)] = (qt_scores, qt_gen, _buf_total, generation)
    _accumulate_stamp_qt_only(
        qt_scores,
        qt_gen,
        compact_doc,
        compact_val,
        seg_start,
        seg_end,
        work_qt_offsets,
        work_seg,
        work_vq,
        generation,
    )
    _sum_qt_stamp_to_doc_total(qt_scores, qt_gen, doc_total, generation)
    return "numba-stamp"


def _pack_work_by_qt(
    segments: list[tuple[list[tuple[int, float]], int, int]],
    n_qt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_qt_items: list[list[tuple[int, float]]] = [[] for _ in range(n_qt)]
    for si, (q_entries, _s, _e) in enumerate(segments):
        for qt, v_q in q_entries:
            if 0 <= qt < n_qt:
                per_qt_items[qt].append((si, float(v_q)))

    work_qt_offsets = np.zeros(n_qt + 1, dtype=np.int64)
    work_seg_list: list[int] = []
    work_vq_list: list[float] = []
    for qt in range(n_qt):
        work_qt_offsets[qt] = len(work_seg_list)
        for si, vq in per_qt_items[qt]:
            work_seg_list.append(si)
            work_vq_list.append(vq)
    work_qt_offsets[n_qt] = len(work_seg_list)
    return (
        work_qt_offsets,
        np.asarray(work_seg_list, dtype=np.int32),
        np.asarray(work_vq_list, dtype=np.float32),
    )


def accumulate_maxsim_fast(
    per_qt: np.ndarray | None,
    doc_idx: np.ndarray,
    values: np.ndarray,
    segments: list[tuple[list[tuple[int, float]], int, int]],
    *,
    prefer_numba: bool = True,
    use_torch_scatter: bool = False,
    doc_total: np.ndarray | None = None,
    n_query_tokens: int = 0,
    parallel_workers: int = 0,
) -> str:
    """Accumulate MaxSim. Use ``doc_total`` to avoid clearing ``n_qt x n_docs`` each query."""
    del use_torch_scatter, parallel_workers
    if not segments:
        return "empty"

    if per_qt is not None:
        n_qt = int(per_qt.shape[0])
    else:
        n_qt = int(n_query_tokens)

    seg_start = np.array([s for _, s, _ in segments], dtype=np.int64)
    seg_end = np.array([e for _, _, e in segments], dtype=np.int64)
    work_qt_offsets, work_seg, work_vq = _pack_work_by_qt(segments, n_qt)

    if prefer_numba and _NUMBA_OK and doc_total is not None:
        n_docs = int(doc_total.shape[0])
        qt_scores, qt_gen, _buf_total, generation = _get_stamp_state(n_qt, n_docs)
        generation = _bump_generation(qt_gen, generation)
        _STAMP_STATE[(n_qt, n_docs)] = (qt_scores, qt_gen, _buf_total, generation)
        _accumulate_stamp_qt_only(
            qt_scores,
            qt_gen,
            doc_idx,
            values,
            seg_start,
            seg_end,
            work_qt_offsets,
            work_seg,
            work_vq,
            generation,
        )
        _sum_qt_stamp_to_doc_total(qt_scores, qt_gen, doc_total, generation)
        return "numba-stamp"

    from .compact_postings import segment_max_by_doc

    assert per_qt is not None
    for q_entries, s, e in segments:
        docs = doc_idx[s:e]
        vals = values[s:e]
        if docs.size == 0:
            continue
        out_docs, max_vals = segment_max_by_doc(docs, vals)
        for qt, v_q in q_entries:
            if 0 <= qt < n_qt:
                scaled = max_vals * np.float32(v_q)
                per_qt[qt, out_docs] = np.maximum(per_qt[qt, out_docs], scaled)
    return "numpy"


def sum_per_qt_fast(per_qt: np.ndarray) -> np.ndarray:
    return per_qt.sum(axis=0, dtype=np.float32)
