#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and prepare MTEB BEIR retrieval eval data (eval only, not training).

Same layout as prepare_msmarco.py stage 1 (index), one subdir per dataset:
  {processed_dir}/{slug}/
    corpus.jsonl
    queries/{split}.tsv
    qrels/{split}.tsv

No pairs/ or hard_negatives/.

Source: mteb/*; queries filtered to ids present in qrels
(same as mteb RetrievalDatasetLoader).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm

logger = logging.getLogger(__name__)

# slug -> MTEB task name (official)
MTEB_EVAL_DATASETS: Dict[str, str] = {
    "arguana": "ArguAna",
    "climate-fever": "ClimateFEVER",
    "trec-covid": "TRECCOVID",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quora": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}

DEFAULT_SPLITS = ("test",)


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _should_skip_output(path: Path, expected: int, *, label: str = "") -> bool:
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
        for qid in sorted(queries.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            text = queries[qid].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")
    return len(queries)


def _write_qrels_tsv(path: Path, qrels: Dict[str, Dict[str, int]]) -> int:
    """TREC qrels: qid 0 docid relevance"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            for doc_id, rel in sorted(
                qrels[qid].items(),
                key=lambda x: (0, int(x[0])) if str(x[0]).isdigit() else (1, str(x[0])),
            ):
                if rel > 0:
                    f.write(f"{qid}\t0\t{doc_id}\t{rel}\n")
                    n += 1
    return n


def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    if title and body:
        return f"{title} {body}".replace("\n", " ")
    return (title or body).replace("\n", " ")


def _load_mteb_retrieval_split(
    hf_repo: str,
    revision: str,
    split: str,
) -> tuple[List[dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load corpus/queries/qrels for one split (aligned with MTEB RetrievalDatasetLoader)."""
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    loader = RetrievalDatasetLoader(
        hf_repo=hf_repo,
        revision=revision,
        split=split,
    )
    data = loader.load()

    corpus_rows: List[dict] = []
    for row in tqdm(data["corpus"], desc=f"{hf_repo}/corpus"):
        doc_id = str(row["id"])
        corpus_rows.append({"_id": doc_id, "text": _corpus_text(row)})

    queries: Dict[str, str] = {}
    for row in data["queries"]:
        qid = str(row["id"])
        queries[qid] = str(row["text"]).replace("\n", " ")

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for qid, doc_scores in data["relevant_docs"].items():
        for doc_id, score in doc_scores.items():
            rel = int(score)
            if rel > 0:
                qrels[str(qid)][str(doc_id)] = rel

    return corpus_rows, dict(queries), dict(qrels)


def organize_mteb_dataset(
    *,
    slug: str,
    task_name: str,
    out_dir: Path,
    splits: Sequence[str],
) -> None:
    import mteb

    task = mteb.get_task(task_name)
    hf_meta = task.metadata.dataset
    if isinstance(hf_meta, dict):
        hf_repo = hf_meta["path"]
        revision = hf_meta.get("revision", "main")
    else:
        hf_repo = str(hf_meta)
        revision = "main"

    eval_splits = list(task.metadata.eval_splits or splits)
    target_splits = [s for s in splits if s in eval_splits] or list(eval_splits)

    dataset_dir = out_dir / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for split in target_splits:
        logger.info("=== %s (%s) split=%s ===", slug, task_name, split)
        corpus_rows, queries, qrels = _load_mteb_retrieval_split(
            hf_repo, revision, split
        )

        corpus_out = dataset_dir / "corpus.jsonl"
        if split == target_splits[0]:
            if not _should_skip_output(
                corpus_out, len(corpus_rows), label=f"[{slug} corpus]"
            ):
                n = _write_jsonl(corpus_out, corpus_rows)
                logger.info("%s corpus: %d docs", slug, n)
        elif not corpus_out.is_file():
            n = _write_jsonl(corpus_out, corpus_rows)
            logger.info("%s corpus: %d docs (shared corpus)", slug, n)

        expected_qrels = sum(len(v) for v in qrels.values())
        q_out = dataset_dir / "queries" / f"{split}.tsv"
        if not _should_skip_output(q_out, len(queries), label=f"[{slug}/{split} queries]"):
            _write_queries_tsv(q_out, queries)

        r_out = dataset_dir / "qrels" / f"{split}.tsv"
        if not _should_skip_output(r_out, expected_qrels, label=f"[{slug}/{split} qrels]"):
            _write_qrels_tsv(r_out, qrels)

        logger.info(
            "%s %s: %d queries, %d qrel pairs, corpus=%d",
            slug,
            split,
            len(queries),
            expected_qrels,
            len(corpus_rows),
        )


def parse_args() -> argparse.Namespace:
    all_slugs = sorted(MTEB_EVAL_DATASETS.keys())
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare MTEB retrieval eval data (index only: corpus/queries/qrels)."
            "Default: 13 English BEIR retrieval tasks."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("./data/processed/mteb"),
        help="Processed data root (default: ./data/processed/mteb)"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=all_slugs,
        default=None,
        metavar="DATASET",
        help=f"Dataset slugs to process (default: all). Choices: {', '.join(all_slugs)}",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split name to write (default: test; must be in task eval_splits)"),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    slugs = args.datasets or sorted(MTEB_EVAL_DATASETS.keys())
    args.processed_dir = args.processed_dir.resolve()
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        task_name = MTEB_EVAL_DATASETS[slug]
        try:
            organize_mteb_dataset(
                slug=slug,
                task_name=task_name,
                out_dir=args.processed_dir,
                splits=args.split,
            )
        except Exception:
            logger.exception("Failed %s (%s)", slug, task_name)
            raise

    logger.info("Done. Processed data: %s", args.processed_dir)


if __name__ == "__main__":
    main()
