"""Columnar inverted-index postings (~10 bytes/row vs Python Posting objects)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import numpy as np


@dataclass
class Posting:
    doc_idx: int
    token_idx: int
    value: float


@dataclass(frozen=True)
class CompactPostings:
    """Sorted-by-latent columnar postings (matches on-disk ``postings.npz`` layout)."""

    latent_ids: np.ndarray  # int32, sorted unique
    offsets: np.ndarray  # int64, len = n_latents + 1
    doc_idx: np.ndarray  # int32
    token_idx: np.ndarray  # int16
    values: np.ndarray  # float32

    def __post_init__(self) -> None:
        object.__setattr__(self, "latent_ids", np.asarray(self.latent_ids, dtype=np.int32))
        object.__setattr__(self, "offsets", np.asarray(self.offsets, dtype=np.int64))
        object.__setattr__(self, "doc_idx", np.asarray(self.doc_idx, dtype=np.int32))
        object.__setattr__(self, "token_idx", np.asarray(self.token_idx, dtype=np.int16))
        vals = np.asarray(self.values)
        if vals.dtype != np.float32:
            vals = vals.astype(np.float32, copy=False)
        object.__setattr__(self, "values", vals)

    @classmethod
    def empty(cls) -> CompactPostings:
        return cls(
            latent_ids=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int64),
            doc_idx=np.array([], dtype=np.int32),
            token_idx=np.array([], dtype=np.int16),
            values=np.array([], dtype=np.float32),
        )

    @property
    def n_postings(self) -> int:
        return int(self.doc_idx.shape[0])

    def nbytes(self) -> int:
        return int(
            self.latent_ids.nbytes
            + self.offsets.nbytes
            + self.doc_idx.nbytes
            + self.token_idx.nbytes
            + self.values.nbytes
        )

    def to_arrays_dict(self) -> dict[str, np.ndarray]:
        return {
            "latent_ids": self.latent_ids,
            "offsets": self.offsets,
            "doc_idx": self.doc_idx,
            "token_idx": self.token_idx,
            "values": self.values,
        }

    @classmethod
    def from_arrays_dict(cls, arrays: dict[str, np.ndarray]) -> CompactPostings:
        return cls(
            latent_ids=arrays["latent_ids"],
            offsets=arrays["offsets"],
            doc_idx=arrays["doc_idx"],
            token_idx=arrays["token_idx"],
            values=arrays["values"],
        )

    def latent_index(self, latent: int) -> int:
        i = int(np.searchsorted(self.latent_ids, latent))
        if i < self.latent_ids.shape[0] and int(self.latent_ids[i]) == latent:
            return i
        return -1

    def iter_latent_rows(self, latent: int) -> Iterator[Tuple[int, int, float]]:
        i = self.latent_index(latent)
        if i < 0:
            return
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        doc = self.doc_idx[s:e]
        tok = self.token_idx[s:e]
        val = self.values[s:e]
        for j in range(e - s):
            yield int(doc[j]), int(tok[j]), float(val[j])

    def doc_ids_for_latent(self, latent: int) -> np.ndarray:
        i = self.latent_index(latent)
        if i < 0:
            return np.array([], dtype=np.int32)
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.doc_idx[s:e]


def segment_max_by_doc(
    docs: np.ndarray,
    vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-``doc_idx`` max of ``vals`` (``docs`` sorted by doc)."""
    return _collapse_segment_doc_max(docs, vals)


def _collapse_segment_doc_max(
    docs: np.ndarray,
    vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge consecutive rows with the same ``doc_idx``, keeping max ``value``."""
    n = int(docs.shape[0])
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    run_starts = np.concatenate([[0], np.where(np.diff(docs) != 0)[0] + 1])
    out_docs = docs[run_starts].astype(np.int32, copy=False)
    out_vals = np.maximum.reduceat(vals, run_starts).astype(np.float32, copy=False)
    return out_docs, out_vals


def _merge_two_doc_max_runs(
    d1: np.ndarray,
    v1: np.ndarray,
    d2: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge sorted (doc, max-val) runs; duplicate docs take max(value)."""
    n1, n2 = int(d1.shape[0]), int(d2.shape[0])
    if n1 == 0:
        return d2.astype(np.int32, copy=False), v2.astype(np.float32, copy=False)
    if n2 == 0:
        return d1.astype(np.int32, copy=False), v1.astype(np.float32, copy=False)

    out_d: list[np.ndarray] = []
    out_v: list[np.ndarray] = []
    i, j = 0, 0
    while i < n1 or j < n2:
        if j >= n2 or (i < n1 and d1[i] < d2[j]):
            out_d.append(d1[i : i + 1])
            out_v.append(v1[i : i + 1])
            i += 1
        elif i >= n1 or d2[j] < d1[i]:
            out_d.append(d2[j : j + 1])
            out_v.append(v2[j : j + 1])
            j += 1
        else:
            out_d.append(d1[i : i + 1])
            out_v.append(
                np.array([max(float(v1[i]), float(v2[j]))], dtype=np.float32)
            )
            i += 1
            j += 1
    return np.concatenate(out_d), np.concatenate(out_v)


def downsample_compact_per_token_topk(
    compact: CompactPostings,
    *,
    topk: int,
    show_progress: bool = False,
) -> CompactPostings:
    """Keep at most ``topk`` highest-value postings per (doc, token) row."""
    from tqdm import tqdm

    topk = int(topk)
    if topk <= 0:
        return CompactPostings.empty()

    n_lat = int(compact.latent_ids.shape[0])
    if n_lat == 0:
        return CompactPostings.empty()

    doc_parts: list[np.ndarray] = []
    tok_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    out_latents: list[int] = []

    latent_iter: range | tqdm = range(n_lat)
    if show_progress:
        latent_iter = tqdm(
            latent_iter, desc="Global index [compact_coarse]", unit="latent"
        )

    for i in latent_iter:
        s, e = int(compact.offsets[i]), int(compact.offsets[i + 1])
        if e <= s:
            continue
        docs = compact.doc_idx[s:e]
        toks = compact.token_idx[s:e]
        vals = compact.values[s:e]
        pair = docs.astype(np.int64) * np.int64(65536) + toks.astype(np.int64)
        run_starts = np.concatenate([[0], np.where(np.diff(pair) != 0)[0] + 1])
        kept_doc: list[np.ndarray] = []
        kept_tok: list[np.ndarray] = []
        kept_val: list[np.ndarray] = []
        for rs, re in zip(run_starts, np.append(run_starts[1:], len(pair))):
            d_part = docs[rs:re]
            t_part = toks[rs:re]
            v_part = np.asarray(vals[rs:re], dtype=np.float32)
            if v_part.size > topk:
                pick = np.argpartition(v_part, -topk)[-topk:]
                d_part = d_part[pick]
                t_part = t_part[pick]
                v_part = v_part[pick]
            kept_doc.append(d_part)
            kept_tok.append(t_part)
            kept_val.append(v_part)
        if not kept_doc:
            continue
        out_latents.append(int(compact.latent_ids[i]))
        doc_parts.append(np.concatenate(kept_doc))
        tok_parts.append(np.concatenate(kept_tok))
        val_parts.append(np.concatenate(kept_val))

    if not out_latents:
        return CompactPostings.empty()

    latent_ids = np.asarray(out_latents, dtype=np.int32)
    doc_idx = np.concatenate(doc_parts)
    token_idx = np.concatenate(tok_parts)
    values = np.concatenate(val_parts)
    offsets = np.zeros(len(out_latents) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([int(a.shape[0]) for a in doc_parts])
    return CompactPostings(
        latent_ids=latent_ids,
        offsets=offsets,
        doc_idx=doc_idx,
        token_idx=token_idx,
        values=values,
    )


def merge_doc_latent_max(
    left: CompactPostings,
    right: CompactPostings,
) -> CompactPostings:
    """Merge per-(latent, doc) max tables (max on duplicate doc within a latent)."""
    if left.n_postings == 0:
        return right
    if right.n_postings == 0:
        return left

    li, ri = 0, 0
    n_l, n_r = int(left.latent_ids.shape[0]), int(right.latent_ids.shape[0])
    out_latents: list[int] = []
    doc_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    while li < n_l or ri < n_r:
        if ri >= n_r or (li < n_l and int(left.latent_ids[li]) < int(right.latent_ids[ri])):
            lid = int(left.latent_ids[li])
            s, e = int(left.offsets[li]), int(left.offsets[li + 1])
            li += 1
            doc_parts.append(left.doc_idx[s:e])
            val_parts.append(left.values[s:e])
        elif li >= n_l or int(right.latent_ids[ri]) < int(left.latent_ids[li]):
            lid = int(right.latent_ids[ri])
            s, e = int(right.offsets[ri]), int(right.offsets[ri + 1])
            ri += 1
            doc_parts.append(right.doc_idx[s:e])
            val_parts.append(right.values[s:e])
        else:
            lid = int(left.latent_ids[li])
            ls, le = int(left.offsets[li]), int(left.offsets[li + 1])
            rs, re = int(right.offsets[ri]), int(right.offsets[ri + 1])
            li += 1
            ri += 1
            md, mv = _merge_two_doc_max_runs(
                left.doc_idx[ls:le],
                left.values[ls:le],
                right.doc_idx[rs:re],
                right.values[rs:re],
            )
            doc_parts.append(md)
            val_parts.append(mv)
        out_latents.append(lid)

    if not out_latents:
        return CompactPostings.empty()

    latent_ids = np.array(out_latents, dtype=np.int32)
    doc_idx = np.concatenate(doc_parts)
    values = np.concatenate(val_parts)
    offsets = np.zeros(len(out_latents) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([int(a.shape[0]) for a in doc_parts])
    return CompactPostings(
        latent_ids=latent_ids,
        offsets=offsets,
        doc_idx=doc_idx,
        token_idx=np.zeros(0, dtype=np.int16),
        values=values,
    )


def build_doc_latent_max(
    compact: CompactPostings,
    *,
    show_progress: bool = False,
) -> CompactPostings:
    """Per-(latent, doc) max activation (safe upper-bound aid for MaxSim pruning).

    Typically far fewer rows than token-level ``compact`` when docs repeat per latent.
    """
    from tqdm import tqdm

    n_lat = int(compact.latent_ids.shape[0])
    if n_lat == 0:
        return CompactPostings.empty()

    run_counts = np.zeros(n_lat, dtype=np.int64)
    for i in range(n_lat):
        s, e = int(compact.offsets[i]), int(compact.offsets[i + 1])
        if e <= s:
            continue
        docs = compact.doc_idx[s:e]
        run_counts[i] = int(np.count_nonzero(np.diff(docs) != 0)) + 1

    offsets = np.zeros(n_lat + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(run_counts)
    total = int(offsets[-1])
    out_doc = np.empty(total, dtype=np.int32)
    out_val = np.empty(total, dtype=np.float32)

    latent_iter: range | tqdm = range(n_lat)
    if show_progress:
        latent_iter = tqdm(latent_iter, desc="Global index [doc_latent_max]", unit="latent")

    for i in latent_iter:
        s, e = int(compact.offsets[i]), int(compact.offsets[i + 1])
        o = int(offsets[i])
        n_runs = int(run_counts[i])
        if n_runs == 0:
            continue
        d_part, v_part = _collapse_segment_doc_max(compact.doc_idx[s:e], compact.values[s:e])
        out_doc[o : o + n_runs] = d_part
        out_val[o : o + n_runs] = v_part

    return CompactPostings(
        latent_ids=compact.latent_ids,
        offsets=offsets,
        doc_idx=out_doc,
        token_idx=np.zeros(0, dtype=np.int16),
        values=out_val,
    )


def query_latent_weight_sums(query: "SparseTokenEmbeddings") -> dict[int, float]:
    """Sum query weights per latent (linear in MaxSim upper bound)."""
    out: dict[int, float] = {}
    for qt in range(query.n_tokens):
        for j in range(query.k):
            latent = int(query.indices[qt, j])
            if latent < 0:
                break
            out[latent] = out.get(latent, 0.0) + float(query.values[qt, j])
    return out


def compute_doc_score_upper_bound(
    query: "SparseTokenEmbeddings",
    plan: "QueryBlockPlan",
    doc_id: int,
    doc_latent_max: CompactPostings,
    *,
    latent_weights: dict[int, float] | None = None,
) -> float:
    """Safe upper bound on exact MaxSim: sum_t sum_{l in q_t} v_q(l) * M_d(l)."""
    del plan  # block layout unused; bound uses all query (latent, weight) pairs
    weights = (
        latent_weights
        if latent_weights is not None
        else query_latent_weight_sums(query)
    )
    total = 0.0
    for latent, v_sum in weights.items():
        total += v_sum * doc_max_at_latent(doc_latent_max, latent, doc_id)
    return total


def doc_max_at_latent(
    doc_latent_max: CompactPostings,
    latent: int,
    doc_id: int,
) -> float:
    """Max activation of ``doc_id`` at ``latent`` (0.0 if no overlap)."""
    i = doc_latent_max.latent_index(latent)
    if i < 0:
        return 0.0
    s, e = int(doc_latent_max.offsets[i]), int(doc_latent_max.offsets[i + 1])
    if e <= s:
        return 0.0
    docs = doc_latent_max.doc_idx[s:e]
    pos = int(np.searchsorted(docs, doc_id))
    if pos < int(docs.shape[0]) and int(docs[pos]) == doc_id:
        return float(doc_latent_max.values[s + pos])
    return 0.0


def compute_latent_max(compact: CompactPostings, n_latents: int) -> np.ndarray:
    """Per-latent max posting value in the corpus (for score upper-bound heuristics)."""
    out = np.zeros(n_latents, dtype=np.float32)
    for i, lid in enumerate(compact.latent_ids):
        s, e = int(compact.offsets[i]), int(compact.offsets[i + 1])
        if e > s:
            out[int(lid)] = float(np.max(compact.values[s:e]))
    return out


def postings_dict_to_compact(
    postings: dict[int, list[Posting]],
    *,
    show_progress: bool = False,
) -> CompactPostings:
    """Legacy conversion (build path should prefer vectorized ingest)."""
    from tqdm import tqdm

    latent_ids = np.array(sorted(postings.keys()), dtype=np.int32)
    doc_parts: list[np.ndarray] = []
    tok_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    offsets = np.zeros(len(latent_ids) + 1, dtype=np.int64)
    latent_iter: Sequence[int] = latent_ids
    if show_progress and len(latent_ids) > 0:
        latent_iter = tqdm(latent_ids, desc="Global index cache [compact]", unit="latent")
    for i, lid in enumerate(latent_iter):
        posts = postings[int(lid)]
        n = len(posts)
        offsets[i + 1] = offsets[i] + n
        if n == 0:
            continue
        doc_parts.append(np.array([p.doc_idx for p in posts], dtype=np.int32))
        tok_parts.append(np.array([p.token_idx for p in posts], dtype=np.int16))
        val_parts.append(np.array([p.value for p in posts], dtype=np.float32))
    return CompactPostings(
        latent_ids=latent_ids,
        offsets=offsets,
        doc_idx=np.concatenate(doc_parts) if doc_parts else np.array([], dtype=np.int32),
        token_idx=np.concatenate(tok_parts) if tok_parts else np.array([], dtype=np.int16),
        values=np.concatenate(val_parts) if val_parts else np.array([], dtype=np.float32),
    )


def compact_from_sorted_latent_runs(
    latent_ids: np.ndarray,
    doc_idx: np.ndarray,
    token_idx: np.ndarray,
    values: np.ndarray,
) -> CompactPostings:
    """Build compact index from COO rows sorted by latent (then doc, token)."""
    n = int(doc_idx.shape[0])
    if n == 0:
        return CompactPostings.empty()
    latent_change = np.concatenate(
        [[0], np.where(np.diff(latent_ids) != 0)[0] + 1, [n]]
    )
    unique_latents = latent_ids[latent_change[:-1]].astype(np.int32, copy=False)
    offsets = np.zeros(unique_latents.shape[0] + 1, dtype=np.int64)
    offsets[1:] = latent_change[1:]
    return CompactPostings(
        latent_ids=unique_latents,
        offsets=offsets,
        doc_idx=doc_idx.astype(np.int32, copy=False),
        token_idx=token_idx.astype(np.int16, copy=False),
        values=values.astype(np.float32, copy=False),
    )


def compact_from_latent_major_coo(
    cols: np.ndarray,
    doc_idx: np.ndarray,
    token_idx: np.ndarray,
    values: np.ndarray,
) -> CompactPostings:
    """One corpus shard (or chunk): ``cols`` sorted by latent."""
    n = int(cols.shape[0])
    if n == 0:
        return CompactPostings.empty()
    return compact_from_sorted_latent_runs(
        cols.astype(np.int32, copy=False),
        doc_idx,
        token_idx,
        values,
    )


def merge_compact(left: CompactPostings, right: CompactPostings) -> CompactPostings:
    """Merge two compact indexes (latent-major, per-latent runs sorted by doc/token)."""
    if left.n_postings == 0:
        return right
    if right.n_postings == 0:
        return left

    li, ri = 0, 0
    n_l, n_r = int(left.latent_ids.shape[0]), int(right.latent_ids.shape[0])
    out_latents: list[int] = []
    doc_parts: list[np.ndarray] = []
    tok_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    while li < n_l or ri < n_r:
        if ri >= n_r or (li < n_l and int(left.latent_ids[li]) < int(right.latent_ids[ri])):
            lid = int(left.latent_ids[li])
            s, e = int(left.offsets[li]), int(left.offsets[li + 1])
            li += 1
            doc_parts.append(left.doc_idx[s:e])
            tok_parts.append(left.token_idx[s:e])
            val_parts.append(left.values[s:e])
        elif li >= n_l or int(right.latent_ids[ri]) < int(left.latent_ids[li]):
            lid = int(right.latent_ids[ri])
            s, e = int(right.offsets[ri]), int(right.offsets[ri + 1])
            ri += 1
            doc_parts.append(right.doc_idx[s:e])
            tok_parts.append(right.token_idx[s:e])
            val_parts.append(right.values[s:e])
        else:
            lid = int(left.latent_ids[li])
            ls, le = int(left.offsets[li]), int(left.offsets[li + 1])
            rs, re = int(right.offsets[ri]), int(right.offsets[ri + 1])
            li += 1
            ri += 1
            doc_parts.append(np.concatenate([left.doc_idx[ls:le], right.doc_idx[rs:re]]))
            tok_parts.append(np.concatenate([left.token_idx[ls:le], right.token_idx[rs:re]]))
            val_parts.append(np.concatenate([left.values[ls:le], right.values[rs:re]]))
        out_latents.append(lid)

    if not out_latents:
        return CompactPostings.empty()

    latent_ids = np.array(out_latents, dtype=np.int32)
    doc_idx = np.concatenate(doc_parts)
    token_idx = np.concatenate(tok_parts)
    values = np.concatenate(val_parts)
    offsets = np.zeros(len(out_latents) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([int(a.shape[0]) for a in doc_parts])
    return CompactPostings(
        latent_ids=latent_ids,
        offsets=offsets,
        doc_idx=doc_idx,
        token_idx=token_idx,
        values=values,
    )


def remap_compact(
    compact: CompactPostings,
    latent_remap: np.ndarray,
) -> CompactPostings:
    """Apply latent permutation and regroup rows (vectorized)."""
    n = compact.n_postings
    if n == 0:
        return compact
    latent_per_row = np.empty(n, dtype=np.int32)
    for i, lid in enumerate(compact.latent_ids):
        new_lid = int(latent_remap[int(lid)])
        s, e = int(compact.offsets[i]), int(compact.offsets[i + 1])
        latent_per_row[s:e] = new_lid
    order = np.lexsort(
        (compact.token_idx, compact.doc_idx, latent_per_row)
    )
    return compact_from_sorted_latent_runs(
        latent_per_row[order],
        compact.doc_idx[order],
        compact.token_idx[order],
        compact.values[order],
    )
