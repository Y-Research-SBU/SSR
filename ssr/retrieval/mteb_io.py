"""Load MTEB retrieval data prepared by prepare_mteb_eval.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class MTEBRetrievalSplit:
    slug: str
    split: str
    corpus_ids: List[str]
    corpus_texts: List[str]
    query_ids: List[str]
    query_texts: List[str]
    relevant_docs: Dict[str, set[str]]


def count_corpus_jsonl(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def iter_corpus_jsonl_batches(
    path: Path,
    *,
    batch_size: int,
    max_docs: int | None = None,
) -> Iterable[Tuple[List[str], List[str]]]:
    """Stream ``corpus.jsonl`` in fixed-size chunks without loading the full corpus."""
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    n_seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            batch_ids.append(str(row["_id"]))
            batch_texts.append(str(row["text"]))
            n_seen += 1
            if len(batch_ids) >= batch_size:
                yield batch_ids, batch_texts
                batch_ids, batch_texts = [], []
            if max_docs is not None and n_seen >= max_docs:
                break
    if batch_ids:
        yield batch_ids, batch_texts


def load_corpus_jsonl(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["_id"]))
            texts.append(str(row["text"]))
    return ids, texts


def load_queries_tsv(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, text = line.split("\t", 1)
            ids.append(qid)
            texts.append(text)
    return ids, texts


def load_qrels_tsv(path: Path) -> Dict[str, set[str]]:
    qrels: Dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, doc_id, rel = line.split("\t")
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def load_mteb_split(
    data_dir: Path,
    slug: str,
    split: str = "test",
) -> MTEBRetrievalSplit:
    root = data_dir / slug
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries" / f"{split}.tsv"
    qrels_path = root / "qrels" / f"{split}.tsv"

    for p in (corpus_path, queries_path, qrels_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    corpus_ids, corpus_texts = load_corpus_jsonl(corpus_path)
    query_ids, query_texts = load_queries_tsv(queries_path)
    relevant_docs = load_qrels_tsv(qrels_path)

    return MTEBRetrievalSplit(
        slug=slug,
        split=split,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        relevant_docs=relevant_docs,
    )


def iter_dataset_slugs(data_dir: Path, slugs: Sequence[str] | None) -> Iterable[str]:
    if slugs:
        yield from slugs
        return
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and (child / "corpus.jsonl").is_file():
            yield child.name
