"""BEIR-style retrieval metrics (compatible with sentence-transformers IR evaluator)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass
class RetrievalMetrics:
    accuracy_at_k: Sequence[int] = (1, 3, 5, 10)
    precision_recall_at_k: Sequence[int] = (1, 3, 5, 10)
    mrr_at_k: Sequence[int] = (10,)
    ndcg_at_k: Sequence[int] = (10,)
    map_at_k: Sequence[int] = (100,)

    def all_k(self) -> int:
        return max(
            max(self.accuracy_at_k, default=0),
            max(self.precision_recall_at_k, default=0),
            max(self.mrr_at_k, default=0),
            max(self.ndcg_at_k, default=0),
            max(self.map_at_k, default=0),
        )


def _dcg_at_k(relevances: Sequence[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def compute_retrieval_metrics(
    *,
    query_ids: Sequence[str],
    ranked_results: Sequence[Sequence[tuple[str, float]]],
    relevant_docs: Mapping[str, set[str]],
    metrics: RetrievalMetrics | None = None,
) -> Dict[str, float]:
    """Compute IR metrics from ranked (corpus_id, score) lists per query."""
    metrics = metrics or RetrievalMetrics()
    n_queries = len(query_ids)

    num_hits_at_k = {k: 0 for k in metrics.accuracy_at_k}
    precisions_at_k = {k: [] for k in metrics.precision_recall_at_k}
    recall_at_k = {k: [] for k in metrics.precision_recall_at_k}
    mrr = {k: 0.0 for k in metrics.mrr_at_k}
    ndcg = {k: [] for k in metrics.ndcg_at_k}
    avep = {k: [] for k in metrics.map_at_k}

    for qid, hits in zip(query_ids, ranked_results):
        rel_set = relevant_docs.get(qid, set())
        top_hits = sorted(hits, key=lambda x: x[1], reverse=True)

        for k_val in metrics.accuracy_at_k:
            for doc_id, _ in top_hits[:k_val]:
                if doc_id in rel_set:
                    num_hits_at_k[k_val] += 1
                    break

        for k_val in metrics.precision_recall_at_k:
            num_correct = sum(
                1 for doc_id, _ in top_hits[:k_val] if doc_id in rel_set
            )
            precisions_at_k[k_val].append(num_correct / k_val)
            if rel_set:
                recall_at_k[k_val].append(num_correct / len(rel_set))
            else:
                recall_at_k[k_val].append(0.0)

        for k_val in metrics.mrr_at_k:
            for rank, (doc_id, _) in enumerate(top_hits[:k_val]):
                if doc_id in rel_set:
                    mrr[k_val] += 1.0 / (rank + 1)
                    break

        for k_val in metrics.ndcg_at_k:
            predicted = [
                1 if doc_id in rel_set else 0 for doc_id, _ in top_hits[:k_val]
            ]
            ideal = [1] * min(len(rel_set), k_val)
            idcg = _dcg_at_k(ideal, k_val)
            ndcg[k_val].append(
                _dcg_at_k(predicted, k_val) / idcg if idcg > 0 else 0.0
            )

        for k_val in metrics.map_at_k:
            num_correct = 0
            sum_prec = 0.0
            for rank, (doc_id, _) in enumerate(top_hits[:k_val]):
                if doc_id in rel_set:
                    num_correct += 1
                    sum_prec += num_correct / (rank + 1)
            denom = min(len(rel_set), k_val) if rel_set else 1
            avep[k_val].append(sum_prec / denom)

    out: Dict[str, float] = {}
    for k_val in metrics.accuracy_at_k:
        out[f"accuracy@{k_val}"] = num_hits_at_k[k_val] / max(n_queries, 1)
    for k_val in metrics.precision_recall_at_k:
        out[f"precision@{k_val}"] = float(
            sum(precisions_at_k[k_val]) / max(len(precisions_at_k[k_val]), 1)
        )
        out[f"recall@{k_val}"] = float(
            sum(recall_at_k[k_val]) / max(len(recall_at_k[k_val]), 1)
        )
    for k_val in metrics.mrr_at_k:
        out[f"mrr@{k_val}"] = mrr[k_val] / max(n_queries, 1)
    for k_val in metrics.ndcg_at_k:
        out[f"ndcg@{k_val}"] = float(
            sum(ndcg[k_val]) / max(len(ndcg[k_val]), 1)
        )
    for k_val in metrics.map_at_k:
        out[f"map@{k_val}"] = float(sum(avep[k_val]) / max(len(avep[k_val]), 1))

    return out
