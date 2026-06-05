#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and prepare MS MARCO passage/document data in three stages:

1. Index (index/): corpus.jsonl, queries/*.tsv, qrels/*.tsv
2. Training pairs (pairs/): query + all positive passages (ready for training)
3. Hard negatives (hard_negatives/): query + positives + negatives (materialized text)

Passage hard negatives:
  1. Prefer HF sentence-transformers/msmarco-hard-negatives (BM25 mined)
  2. Local BM25 for queries not covered by HF

Document hard negatives: no public HF set; build all with local BM25.

Skip writing a file if it exists and line count matches expectation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Sequence

from tqdm import tqdm

if TYPE_CHECKING:
    import bm25s

logger = logging.getLogger(__name__)

BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"

PASSAGE_ARCHIVE = f"{BASE_URL}/collectionandqueries.tar.gz"
# collectionandqueries.tar.gz lacks full dev qrels; download separately
PASSAGE_EXTRA_FILES = {
    "qrels.dev.tsv": f"{BASE_URL}/qrels.dev.tsv",
}

DOCUMENT_FILES = {
    "corpus": f"{BASE_URL}/msmarco-docs.tsv.gz",
    "train_queries": f"{BASE_URL}/msmarco-doctrain-queries.tsv.gz",
    "train_qrels": f"{BASE_URL}/msmarco-doctrain-qrels.tsv.gz",
    "validation_queries": f"{BASE_URL}/msmarco-docdev-queries.tsv.gz",
    "validation_qrels": f"{BASE_URL}/msmarco-docdev-qrels.tsv.gz",
    "test_queries": f"{BASE_URL}/docleaderboard-queries.tsv.gz",
}

SPLITS = ("train", "validation", "test")

# MS MARCO passage BM25 defaults (Anserini)
BM25_K1 = 0.82
BM25_B = 0.68

# Hugging Face: sentence-transformers/msmarco-hard-negatives
HF_HARD_NEGATIVES_BASE = (
    "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main"
)
HF_PASSAGE_BM25_FILE = "msmarco-hard-negatives-bm25_1k.jsonl.gz"


def _count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _should_skip_output(path: Path, expected: int, *, label: str = "") -> bool:
    """Skip write if file exists with expected non-empty line count."""
    if not path.is_file():
        return False
    actual = _count_nonempty_lines(path)
    tag = f"{label} " if label else ""
    if actual == expected:
        logger.info(
            "%s exists with expected lines (%d), skip: %s",
            tag,
            expected,
            path,
        )
        return True
    logger.warning(
        "%s exists but line count mismatch (actual=%d, expected=%d), reprocessing: %s",
        tag,
        actual,
        expected,
        path,
    )
    return False


def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("Already exists, skip download: %s", dest)
        return
    logger.info("Downloading %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "MVR-msmarco-prep/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            block = resp.read(chunk_size)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if total:
                pct = downloaded * 100 // total
                if downloaded % (50 * chunk_size) == 0:
                    logger.info("  %d / %d MiB (%d%%)", downloaded // (1 << 20), total // (1 << 20), pct)
    tmp.replace(dest)
    logger.info("Download done: %s (%d MiB)", dest, dest.stat().st_size // (1 << 20))


def _extract_tar_gz(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)


def _read_tsv_lines(path: Path) -> Iterator[List[str]]:
    open_fn = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with open_fn(path, mode, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            yield line.split("\t")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_queries_tsv(path: Path, queries: Dict[str, str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(queries.keys(), key=lambda x: int(x) if x.isdigit() else x):
            text = queries[qid].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")
    return len(queries)


def _write_qrels_tsv(path: Path, qrels: Dict[str, Dict[str, int]]) -> int:
    """TREC qrels: qid 0 docid relevance"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys(), key=lambda x: int(x) if x.isdigit() else x):
            for doc_id, rel in sorted(qrels[qid].items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
                f.write(f"{qid}\t0\t{doc_id}\t{rel}\n")
                n += 1
    return n


def _load_queries_tsv(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    for parts in _read_tsv_lines(path):
        if len(parts) >= 2:
            queries[str(parts[0])] = parts[1]
    return queries


def _load_qrels_tsv(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for parts in _read_tsv_lines(path):
        if len(parts) >= 4:
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
        elif len(parts) == 3:
            qid, doc_id, rel = parts[0], parts[1], parts[2]
        else:
            continue
        if int(rel) > 0:
            qrels[str(qid)][str(doc_id)] = int(rel)
    return dict(qrels)


def _load_corpus_text_map(corpus_path: Path) -> Dict[str, str]:
    """Load corpus text once for pairs/hard_negatives materialization."""
    logger.info("Loading corpus text index: %s", corpus_path)
    text_map: Dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="corpus text"):
            if not line.strip():
                continue
            row = json.loads(line)
            text_map[str(row["_id"])] = row["text"]
    logger.info("Corpus text index done: %d docs", len(text_map))
    return text_map


def _load_corpus_doc_ids_and_texts(corpus_path: Path) -> tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    texts: List[str] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_ids.append(str(row["_id"]))
            texts.append(row["text"])
    return doc_ids, texts


def _get_split_target_qids(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> List[str]:
    if qrels:
        return [qid for qid in queries if qid in qrels and qrels[qid]]
    return list(queries.keys())


def _expected_pair_count(subset_dir: Path, split: str) -> int:
    queries_path = subset_dir / "queries" / f"{split}.tsv"
    if not queries_path.is_file():
        return 0
    queries = _load_queries_tsv(queries_path)
    qrels_path = subset_dir / "qrels" / f"{split}.tsv"
    qrels = _load_qrels_tsv(qrels_path) if qrels_path.is_file() else {}
    return len(_get_split_target_qids(queries, qrels))


def _materialize_positives(
    positive_ids: Sequence[str],
    text_map: Dict[str, str],
) -> List[dict]:
    positives: List[dict] = []
    for doc_id in positive_ids:
        text = text_map.get(str(doc_id))
        if text is None:
            logger.debug("doc_id=%s missing from corpus, skip positive", doc_id)
            continue
        positives.append({"doc_id": str(doc_id), "text": text})
    return positives


def _ensure_passage_extra_files(extract_dir: Path) -> None:
    """Fetch qrels.dev.tsv missing from collectionandqueries.tar.gz."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in PASSAGE_EXTRA_FILES.items():
        _download(url, extract_dir / filename)


def download_passage_raw(raw_dir: Path) -> Path:
    passage_dir = raw_dir / "passage"
    passage_dir.mkdir(parents=True, exist_ok=True)
    archive = passage_dir / "collectionandqueries.tar.gz"
    _download(PASSAGE_ARCHIVE, archive)
    extract_dir = passage_dir / "extracted"
    if not (extract_dir / "collection.tsv").is_file():
        logger.info("Extracting passage archive ...")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        _extract_tar_gz(archive, extract_dir)
    _ensure_passage_extra_files(extract_dir)
    return extract_dir


def download_document_raw(raw_dir: Path) -> Path:
    doc_dir = raw_dir / "document"
    doc_dir.mkdir(parents=True, exist_ok=True)
    for _name, url in DOCUMENT_FILES.items():
        filename = url.rsplit("/", 1)[-1]
        _download(url, doc_dir / filename)
    return doc_dir


def organize_passage(raw_extracted: Path, out_dir: Path) -> None:
    """Stage 1 (passage): corpus + queries + qrels index."""
    split_files = {
        "train": ("queries.train.tsv", "qrels.train.tsv"),
        "validation": ("queries.dev.tsv", "qrels.dev.tsv"),
        "test": ("queries.dev.small.tsv", "qrels.dev.small.tsv"),
    }
    collection_path = raw_extracted / "collection.tsv"
    if not collection_path.is_file():
        raise FileNotFoundError(f"Missing {collection_path}")

    _ensure_passage_extra_files(raw_extracted)

    out_passage = out_dir / "passage"
    corpus_out = out_passage / "corpus.jsonl"
    expected_corpus = sum(
        1 for parts in _read_tsv_lines(collection_path) if len(parts) >= 2
    )

    if not _should_skip_output(corpus_out, expected_corpus, label="[index/passage corpus]"):
        logger.info("Writing passage corpus.jsonl ...")

        def corpus_rows() -> Iterator[dict]:
            for parts in _read_tsv_lines(collection_path):
                if len(parts) >= 2:
                    yield {"_id": str(parts[0]), "text": parts[1].replace("\n", " ")}

        n = _write_jsonl(corpus_out, corpus_rows())
        logger.info("passage corpus: %d docs", n)

    for split, (q_file, r_file) in split_files.items():
        q_path = raw_extracted / q_file
        r_path = raw_extracted / r_file
        if not q_path.is_file():
            raise FileNotFoundError(q_path)
        if not r_path.is_file():
            raise FileNotFoundError(r_path)

        queries = _load_queries_tsv(q_path)
        qrels = _load_qrels_tsv(r_path)
        expected_qrels = sum(len(v) for v in qrels.values())

        q_out = out_passage / "queries" / f"{split}.tsv"
        if not _should_skip_output(q_out, len(queries), label=f"[index/passage/{split} queries]"):
            _write_queries_tsv(q_out, queries)

        r_out = out_passage / "qrels" / f"{split}.tsv"
        if not _should_skip_output(r_out, expected_qrels, label=f"[index/passage/{split} qrels]"):
            _write_qrels_tsv(r_out, qrels)

        logger.info(
            "passage %s index: %d queries, %d qrel pairs, %d train qids",
            split,
            len(queries),
            expected_qrels,
            _expected_pair_count(out_passage, split),
        )


def organize_document(raw_doc_dir: Path, out_dir: Path) -> None:
    """Stage 1 (document): corpus + queries + qrels index."""
    split_files = {
        "train": ("msmarco-doctrain-queries.tsv.gz", "msmarco-doctrain-qrels.tsv.gz"),
        "validation": ("msmarco-docdev-queries.tsv.gz", "msmarco-docdev-qrels.tsv.gz"),
        "test": ("docleaderboard-queries.tsv.gz", None),
    }
    corpus_path = raw_doc_dir / "msmarco-docs.tsv.gz"
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)

    out_doc = out_dir / "document"
    corpus_out = out_doc / "corpus.jsonl"
    expected_corpus = sum(
        1 for parts in _read_tsv_lines(corpus_path) if len(parts) >= 4
    )

    if not _should_skip_output(corpus_out, expected_corpus, label="[index/document corpus]"):
        logger.info("Writing document corpus.jsonl ...")

        def corpus_rows() -> Iterator[dict]:
            for parts in _read_tsv_lines(corpus_path):
                if len(parts) >= 4:
                    title, body = parts[2], parts[3]
                    text = f"{title} {body}".strip().replace("\n", " ")
                    yield {"_id": str(parts[0]), "text": text, "title": title}

        n = _write_jsonl(corpus_out, corpus_rows())
        logger.info("document corpus: %d docs", n)

    for split, (q_file, r_file) in split_files.items():
        q_path = raw_doc_dir / q_file
        if not q_path.is_file():
            raise FileNotFoundError(q_path)
        queries = _load_queries_tsv(q_path)

        q_out = out_doc / "queries" / f"{split}.tsv"
        if not _should_skip_output(q_out, len(queries), label=f"[index/document/{split} queries]"):
            _write_queries_tsv(q_out, queries)

        if r_file is not None:
            r_path = raw_doc_dir / r_file
            qrels = _load_qrels_tsv(r_path)
            expected_qrels = sum(len(v) for v in qrels.values())
            r_out = out_doc / "qrels" / f"{split}.tsv"
            if not _should_skip_output(r_out, expected_qrels, label=f"[index/document/{split} qrels]"):
                _write_qrels_tsv(r_out, qrels)
            logger.info(
                "document %s index: %d queries, %d qrel pairs, %d train qids",
                split,
                len(queries),
                expected_qrels,
                _expected_pair_count(out_doc, split),
            )
        else:
            logger.info(
                "document %s index: %d queries (no public qrels), %d train qids",
                split,
                len(queries),
                _expected_pair_count(out_doc, split),
            )


def build_training_pairs_for_split(
    subset_dir: Path,
    split: str,
    text_map: Dict[str, str],
) -> int:
    """Stage 2: materialize query + all positives for one split."""
    expected = _expected_pair_count(subset_dir, split)
    if expected == 0:
        logger.info("[pairs/%s] no train qids, skip", split)
        return 0

    out_path = subset_dir / "pairs" / f"{split}.jsonl"
    if _should_skip_output(out_path, expected, label=f"[pairs/{split}]"):
        return expected

    queries = _load_queries_tsv(subset_dir / "queries" / f"{split}.tsv")
    qrels_path = subset_dir / "qrels" / f"{split}.tsv"
    qrels = _load_qrels_tsv(qrels_path) if qrels_path.is_file() else {}
    target_qids = _get_split_target_qids(queries, qrels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_doc_total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for qid in sorted(target_qids, key=lambda x: int(x) if x.isdigit() else x):
            positive_ids = sorted(
                qrels.get(qid, {}).keys(),
                key=lambda x: int(x) if x.isdigit() else x,
            )
            positives = _materialize_positives(positive_ids, text_map)
            missing_doc_total += max(0, len(positive_ids) - len(positives))
            record = {
                "qid": qid,
                "query": queries[qid],
                "positives": positives,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    actual = _count_nonempty_lines(out_path)
    logger.info(
        "[pairs/%s] wrote %s: %d rows (expected %d, missing positive docs %d)",
        split,
        out_path,
        actual,
        expected,
        missing_doc_total,
    )
    if actual != expected:
        raise RuntimeError(
            f"pairs/{split}.jsonl line count mismatch: actual={actual}, expected={expected}"
        )
    return actual


def run_training_pairs(subset_name: str, processed_dir: Path) -> None:
    """Stage 2: write pairs/{split}.jsonl."""
    subset_dir = processed_dir / subset_name
    corpus_path = subset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus; run index stage first: {corpus_path}")

    text_map = _load_corpus_text_map(corpus_path)
    for split in SPLITS:
        queries_path = subset_dir / "queries" / f"{split}.tsv"
        if not queries_path.is_file():
            logger.info("[%s/pairs/%s] no queries file, skip", subset_name, split)
            continue
        build_training_pairs_for_split(subset_dir, split, text_map)


def _require_bm25s():
    try:
        import bm25s  # noqa: WPS433
        import Stemmer  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "BM25 requires: pip install -r requirements.txt"
        ) from exc
    return bm25s, Stemmer


def _build_bm25_index(texts: Sequence[str], index_dir: Path) -> "bm25s.BM25":
    bm25s, Stemmer = _require_bm25s()
    index_dir.mkdir(parents=True, exist_ok=True)
    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B, method="lucene")
    logger.info("BM25 tokenize + index (%d docs)...", len(texts))
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    retriever.index(corpus_tokens)
    retriever.save(str(index_dir), corpus=texts)
    logger.info("BM25 index saved: %s", index_dir)
    return retriever


def _load_or_build_bm25(texts: Sequence[str], index_dir: Path) -> "bm25s.BM25":
    bm25s, _ = _require_bm25s()
    params_file = index_dir / "params.json"
    params_index_file = index_dir / "params.index.json"
    if params_file.is_file() or params_index_file.is_file():
        logger.info("Loading BM25 index: %s", index_dir)
        return bm25s.BM25.load(str(index_dir), load_corpus=False, mmap=True)
    return _build_bm25_index(texts, index_dir)


def download_hf_passage_hard_negatives(raw_dir: Path, hf_file: Optional[Path] = None) -> Path:
    """Download HF passage BM25 hard negatives to raw_dir/passage/."""
    passage_dir = raw_dir / "passage"
    passage_dir.mkdir(parents=True, exist_ok=True)
    dest = hf_file or (passage_dir / HF_PASSAGE_BM25_FILE)
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("HF hard negatives exist, skip download: %s", dest)
        return dest
    url = f"{HF_HARD_NEGATIVES_BASE}/{HF_PASSAGE_BM25_FILE}"
    _download(url, dest)
    return dest


def _extract_ids_from_hf_field(field) -> List[str]:
    ids: List[str] = []
    if not isinstance(field, list):
        return ids
    for item in field:
        if isinstance(item, dict):
            pid = item.get("pid")
            if pid is not None:
                ids.append(str(pid))
        else:
            ids.append(str(item))
    return ids


def _parse_hf_bm25_negatives(row: dict, num_negatives: int) -> List[str]:
    neg = row.get("neg")
    pids: List[str] = []
    if isinstance(neg, dict):
        raw = neg.get("bm25", [])
        pids = _extract_ids_from_hf_field(raw)
    elif isinstance(neg, list):
        pids = _extract_ids_from_hf_field(neg)
    elif "bm25" in row:
        pids = _extract_ids_from_hf_field(row["bm25"])

    positive_ids = set(_extract_ids_from_hf_field(row.get("pos", [])))
    filtered: List[str] = []
    for pid in pids:
        if pid in positive_ids:
            continue
        filtered.append(pid)
        if len(filtered) >= num_negatives:
            break
    return filtered


def _stream_hf_negatives_lookup(
    hf_path: Path,
    needed_qids: set[str],
    num_negatives: int,
) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    if not needed_qids:
        return lookup
    logger.info(
        "Streaming HF hard negatives (match %d qids): %s",
        len(needed_qids),
        hf_path,
    )
    open_fn = gzip.open if hf_path.suffix == ".gz" or str(hf_path).endswith(".gz") else open
    mode = "rt" if str(hf_path).endswith(".gz") else "r"
    remaining = set(needed_qids)
    with open_fn(hf_path, mode, encoding="utf-8") as f:
        for line in tqdm(f, desc="HF hard negatives"):
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in remaining:
                continue
            negs = _parse_hf_bm25_negatives(row, num_negatives)
            if negs:
                lookup[qid] = negs
            remaining.discard(qid)
    logger.info(
        "HF hit %d / %d qids, miss %d (local BM25 fill)",
        len(lookup),
        len(needed_qids),
        len(needed_qids) - len(lookup),
    )
    return lookup


def _make_hard_negative_record(
    *,
    qid: str,
    query_text: str,
    positives: List[dict],
    negative_doc_ids: List[str],
    text_map: Dict[str, str],
    source: str,
    scores: Optional[List[float]] = None,
) -> dict:
    negatives: List[dict] = []
    for i, doc_id in enumerate(negative_doc_ids):
        text = text_map.get(str(doc_id))
        if text is None:
            continue
        entry: dict = {"doc_id": str(doc_id), "text": text, "source": source}
        if scores is not None and i < len(scores):
            entry["score"] = scores[i]
        negatives.append(entry)
    return {
        "qid": qid,
        "query": query_text,
        "positives": positives,
        "negatives": negatives,
    }


def _load_pairs_records(pairs_path: Path) -> List[dict]:
    records: List[dict] = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def extract_hard_negatives_bm25(
    *,
    subset_name: str,
    split: str,
    doc_ids: List[str],
    texts: List[str],
    queries: Dict[str, str],
    positive_ids_by_qid: Dict[str, set[str]],
    target_qids: List[str],
    index_dir: Path,
    num_negatives: int,
    retrieve_k: int,
) -> Dict[str, dict]:
    if not target_qids:
        return {}

    bm25s, Stemmer = _require_bm25s()
    retriever = _load_or_build_bm25(texts, index_dir)
    stemmer = Stemmer.Stemmer("english")
    records: Dict[str, dict] = {}
    batch_size = 64

    for start in tqdm(
        range(0, len(target_qids), batch_size),
        desc=f"BM25 {subset_name}/{split}",
    ):
        batch_qids = target_qids[start : start + batch_size]
        batch_queries = [queries[qid] for qid in batch_qids]
        query_tokens = bm25s.tokenize(batch_queries, stopwords="en", stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=retrieve_k, show_progress=False)

        for qid, query_text, doc_indices, doc_scores in zip(
            batch_qids, batch_queries, results, scores
        ):
            positive_ids = positive_ids_by_qid.get(qid, set())
            neg_ids: List[str] = []
            neg_scores: List[float] = []
            for idx, score in zip(doc_indices, doc_scores):
                if idx < 0 or idx >= len(doc_ids):
                    continue
                doc_id = doc_ids[int(idx)]
                if doc_id in positive_ids:
                    continue
                neg_ids.append(doc_id)
                neg_scores.append(float(score))
                if len(neg_ids) >= num_negatives:
                    break
            if neg_ids:
                records[qid] = {
                    "negative_doc_ids": neg_ids,
                    "scores": neg_scores,
                    "source": "local_bm25",
                }
    return records


def build_hard_negatives_for_split(
    *,
    subset_name: str,
    subset_dir: Path,
    split: str,
    text_map: Dict[str, str],
    doc_ids: List[str],
    texts: List[str],
    index_dir: Path,
    num_negatives: int,
    retrieve_k: int,
    hf_lookup: Dict[str, List[str]],
) -> int:
    """Stage 3: triplets with materialized text for one split."""
    pairs_path = subset_dir / "pairs" / f"{split}.jsonl"
    if not pairs_path.is_file():
        logger.warning("[hard_negatives/%s] missing pairs, skip: %s", split, pairs_path)
        return 0

    expected = _count_nonempty_lines(pairs_path)
    out_path = subset_dir / "hard_negatives" / f"{split}.jsonl"
    if _should_skip_output(out_path, expected, label=f"[hard_negatives/{split}]"):
        return expected

    pair_records = _load_pairs_records(pairs_path)
    if len(pair_records) != expected:
        raise RuntimeError(
            f"pairs/{split}.jsonl line mismatch: loaded={len(pair_records)}, expected={expected}"
        )

    queries = {str(row["qid"]): row["query"] for row in pair_records}
    positive_ids_by_qid = {
        str(row["qid"]): {p["doc_id"] for p in row.get("positives", [])}
        for row in pair_records
    }
    positives_by_qid = {str(row["qid"]): row.get("positives", []) for row in pair_records}
    target_qids = [str(row["qid"]) for row in pair_records]

    missing_qids = [qid for qid in target_qids if qid not in hf_lookup]
    bm25_partial: Dict[str, dict] = {}
    if missing_qids:
        logger.info(
            "[%s/hard_negatives/%s] %d qids filled with local BM25",
            subset_name,
            split,
            len(missing_qids),
        )
        bm25_partial = extract_hard_negatives_bm25(
            subset_name=subset_name,
            split=split,
            doc_ids=doc_ids,
            texts=texts,
            queries=queries,
            positive_ids_by_qid=positive_ids_by_qid,
            target_qids=missing_qids,
            index_dir=index_dir,
            num_negatives=num_negatives,
            retrieve_k=retrieve_k,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_hf, n_bm25, n_empty = 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for qid in target_qids:
            positives = positives_by_qid[qid]
            if qid in hf_lookup:
                record = _make_hard_negative_record(
                    qid=qid,
                    query_text=queries[qid],
                    positives=positives,
                    negative_doc_ids=hf_lookup[qid],
                    text_map=text_map,
                    source="hf_bm25",
                )
                n_hf += 1
            elif qid in bm25_partial:
                partial = bm25_partial[qid]
                record = _make_hard_negative_record(
                    qid=qid,
                    query_text=queries[qid],
                    positives=positives,
                    negative_doc_ids=partial["negative_doc_ids"],
                    text_map=text_map,
                    source=partial["source"],
                    scores=partial.get("scores"),
                )
                n_bm25 += 1
            else:
                record = {
                    "qid": qid,
                    "query": queries[qid],
                    "positives": positives,
                    "negatives": [],
                }
                n_empty += 1
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    actual = _count_nonempty_lines(out_path)
    logger.info(
        "[%s/hard_negatives/%s] wrote %s: %d rows (HF=%d, BM25=%d, no negatives=%d)",
        subset_name,
        split,
        out_path,
        actual,
        n_hf,
        n_bm25,
        n_empty,
    )
    if actual != expected:
        raise RuntimeError(
            f"hard_negatives/{split}.jsonl line mismatch: actual={actual}, expected={expected}"
        )
    return actual


def run_passage_hard_negatives(
    processed_dir: Path,
    raw_dir: Path,
    num_negatives: int,
    retrieve_k: int,
    hf_path: Optional[Path],
    skip_hf: bool,
) -> None:
    """Stage 3 (passage): HF BM25 negatives + local BM25 fill, materialized text."""
    subset_dir = processed_dir / "passage"
    corpus_path = subset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    text_map = _load_corpus_text_map(corpus_path)
    doc_ids, texts = _load_corpus_doc_ids_and_texts(corpus_path)
    index_dir = subset_dir / "bm25_index"

    all_needed: set[str] = set()
    splits_to_process: List[str] = []
    for split in SPLITS:
        pairs_path = subset_dir / "pairs" / f"{split}.jsonl"
        if not pairs_path.is_file():
            logger.info("[passage/hard_negatives/%s] missing pairs, skip", split)
            continue
        expected = _count_nonempty_lines(pairs_path)
        out_path = subset_dir / "hard_negatives" / f"{split}.jsonl"
        if _should_skip_output(out_path, expected, label=f"[hard_negatives/{split}]"):
            continue
        splits_to_process.append(split)
        for row in _load_pairs_records(pairs_path):
            all_needed.add(str(row["qid"]))

    if not splits_to_process:
        logger.info("passage hard negatives: nothing to do (exists or missing pairs)")
        return

    hf_lookup: Dict[str, List[str]] = {}
    if not skip_hf:
        if hf_path is None:
            hf_path = download_hf_passage_hard_negatives(raw_dir)
        if not hf_path.is_file():
            raise FileNotFoundError(f"HF hard negatives file missing: {hf_path}")
        hf_lookup = _stream_hf_negatives_lookup(hf_path, all_needed, num_negatives)

    for split in splits_to_process:
        build_hard_negatives_for_split(
            subset_name="passage",
            subset_dir=subset_dir,
            split=split,
            text_map=text_map,
            doc_ids=doc_ids,
            texts=texts,
            index_dir=index_dir,
            num_negatives=num_negatives,
            retrieve_k=retrieve_k,
            hf_lookup=hf_lookup,
        )


def run_document_hard_negatives(
    processed_dir: Path,
    num_negatives: int,
    retrieve_k: int,
) -> None:
    """Stage 3 (document): local BM25 hard negatives only."""
    subset_dir = processed_dir / "document"
    corpus_path = subset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    text_map = _load_corpus_text_map(corpus_path)
    doc_ids, texts = _load_corpus_doc_ids_and_texts(corpus_path)
    index_dir = subset_dir / "bm25_index"

    for split in SPLITS:
        pairs_path = subset_dir / "pairs" / f"{split}.jsonl"
        if not pairs_path.is_file():
            logger.info("[document/hard_negatives/%s] missing pairs, skip", split)
            continue
        build_hard_negatives_for_split(
            subset_name="document",
            subset_dir=subset_dir,
            split=split,
            text_map=text_map,
            doc_ids=doc_ids,
            texts=texts,
            index_dir=index_dir,
            num_negatives=num_negatives,
            retrieve_k=retrieve_k,
            hf_lookup={},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MS MARCO data prep pipeline:"
            "1) index  2) pairs  3) hard_negatives triplets."
            "Passage: HF first, local BM25 fill; Document: local BM25 only."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./data/raw/msmarco"),
        help="Raw data dir (default: ./data/raw/msmarco)"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("./data/processed/msmarco"),
        help="Processed data dir (default: ./data/processed/msmarco)"),
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=32,
        help="Hard negatives per query (default: 32)"),
    )
    parser.add_argument(
        "--hf-hard-negatives-path",
        type=Path,
        default=None,
        help=(
            "Local path to HF passage BM25 hard negatives;"
            "default raw-dir/passage/msmarco-hard-negatives-bm25_1k.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--skip-hf-hard-negatives",
        action="store_true",
        help="Passage: skip HF, build all with local BM25"),
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=1000,
        help="BM25 retrieve-k, must exceed num-hard-negatives (default: 1000)"),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; use existing raw-dir files"),
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip stage 1: index (corpus/queries/qrels)"),
    )
    parser.add_argument(
        "--skip-pairs",
        action="store_true",
        help="Skip stage 2: pairs/"),
    )
    parser.add_argument(
        "--skip-hard-negatives",
        action="store_true",
        help="Skip stage 3: hard negatives"),
    )
    # Legacy flag aliases
    parser.add_argument(
        "--skip-organize",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--subset",
        choices=("passage", "document", "both"),
        default="both",
        help="Process passage, document, or both (default: both)"),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.num_hard_negatives <= 0:
        raise ValueError("--num-hard-negatives must be positive")
    if args.retrieve_k <= args.num_hard_negatives:
        raise ValueError("--retrieve-k must exceed --num-hard-negatives")

    skip_index = args.skip_index or args.skip_organize
    skip_hard_negatives = args.skip_hard_negatives or args.skip_bm25

    args.raw_dir = args.raw_dir.resolve()
    args.processed_dir = args.processed_dir.resolve()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    passage_extracted: Optional[Path] = None
    document_dir: Optional[Path] = None

    if not args.skip_download:
        if args.subset in ("passage", "both"):
            passage_extracted = download_passage_raw(args.raw_dir)
        if args.subset in ("document", "both"):
            document_dir = download_document_raw(args.raw_dir)
    else:
        pe = args.raw_dir / "passage" / "extracted"
        dd = args.raw_dir / "document"
        if args.subset in ("passage", "both") and pe.is_dir():
            passage_extracted = pe
        if args.subset in ("document", "both") and dd.is_dir():
            document_dir = dd

    if not skip_index:
        logger.info("=== Stage 1/3: index (corpus / queries / qrels) ===")
        if args.subset in ("passage", "both"):
            if passage_extracted is None:
                passage_extracted = args.raw_dir / "passage" / "extracted"
            organize_passage(passage_extracted, args.processed_dir)
        if args.subset in ("document", "both"):
            if document_dir is None:
                document_dir = args.raw_dir / "document"
            organize_document(document_dir, args.processed_dir)

    if not args.skip_pairs:
        logger.info("=== Stage 2/3: pairs ===")
        if args.subset in ("passage", "both"):
            run_training_pairs("passage", args.processed_dir)
        if args.subset in ("document", "both"):
            run_training_pairs("document", args.processed_dir)

    if not skip_hard_negatives:
        logger.info("=== Stage 3/3: hard negatives ===")
        if args.subset in ("passage", "both"):
            run_passage_hard_negatives(
                args.processed_dir,
                args.raw_dir,
                args.num_hard_negatives,
                args.retrieve_k,
                args.hf_hard_negatives_path,
                args.skip_hf_hard_negatives,
            )
        if args.subset in ("document", "both"):
            run_document_hard_negatives(
                args.processed_dir,
                args.num_hard_negatives,
                args.retrieve_k,
            )

    logger.info("Done. Processed data: %s", args.processed_dir)


if __name__ == "__main__":
    main()
