"""MTEB BEIR-style sparse MaxSim retrieval for SSR."""

from .exact_retriever import ExactSparseMaxSimRetriever
from .metrics import RetrievalMetrics, compute_retrieval_metrics
from .retriever import RetrieverConfig, SparseMaxSimRetriever, build_retriever
from .sparse_repr import load_sparse_corpus, save_sparse_corpus

__all__ = [
    "RetrieverConfig",
    "SparseMaxSimRetriever",
    "ExactSparseMaxSimRetriever",
    "build_retriever",
    "RetrievalMetrics",
    "compute_retrieval_metrics",
    "load_sparse_corpus",
    "save_sparse_corpus",
    "run_mteb_eval",
]


def run_mteb_eval(args):
    from .eval_mteb import run_mteb_eval as _run

    return _run(args)
