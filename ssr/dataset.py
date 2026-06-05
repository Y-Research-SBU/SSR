"""Dataset helpers for ColBERT + SAE training on MS MARCO passage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

SampleFormat = Literal["pair", "triplet"]


def _corpus_doc_id(row: dict) -> str:
    if "_id" in row:
        return str(row["_id"])
    if "id" in row:
        return str(row["id"])
    raise KeyError(f"corpus row missing _id/id: {row.keys()}")


__all__ = [
    "MSMarcoPassagePaths",
    "CorpusLookup",
    "load_hf_msmarco_triplets",
    "load_msmarco_passage_datasets",
    "load_msmarco_hard_negatives",
    "validate_loss_weights",
    "validate_loss_sample_format",
]


@dataclass(frozen=True)
class MSMarcoPassagePaths:
    """Standard layout produced by prepare_msmarco.py for the passage subset."""

    root: Path

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "MSMarcoPassagePaths":
        return cls(root=Path(data_dir))

    @property
    def corpus(self) -> Path:
        return self.root / "corpus.jsonl"

    def queries(self, split: str) -> Path:
        return self.root / "queries" / f"{split}.tsv"

    def qrels(self, split: str) -> Path:
        return self.root / "qrels" / f"{split}.tsv"

    def pairs(self, split: str) -> Path:
        return self.root / "pairs" / f"{split}.jsonl"

    def hard_negatives(self, split: str) -> Path:
        return self.root / "hard_negatives" / f"{split}.jsonl"

    def validate(self, *, require_corpus: bool = True) -> None:
        if require_corpus and not self.corpus.is_file():
            raise FileNotFoundError(f"Missing corpus: {self.corpus}")


def validate_loss_weights(
    sample_format: SampleFormat,
    *,
    recon_coef: float,
    auxk_coef: float,
    ucl_coef: float,
    maxsim_coef: float,
) -> None:
    """Validate loss weights vs sample format; raises ValueError on mismatch."""
    if recon_coef <= 0 and auxk_coef <= 0 and ucl_coef <= 0 and maxsim_coef <= 0:
        raise ValueError(
            "At least one loss coefficient must be > 0 (--recon-coef / --auxk-coef / --ucl-coef / --maxsim-coef)."
        )
    if maxsim_coef > 0 and sample_format != "triplet":
        raise ValueError(
            f"maxsim_coef={maxsim_coef} > 0 requires sample_format=triplet (query+positive+negative),"
            f"got {sample_format!r}. Change --sample-format or set --maxsim-coef to 0."
        )
    if ucl_coef > 0 and sample_format not in ("pair", "triplet"):
        raise ValueError(f"ucl_coef > 0 requires sample_format pair or triplet, got {sample_format!r}.")


def validate_loss_sample_format(loss_type: str, sample_format: SampleFormat) -> None:
    """[deprecated] Use validate_loss_weights."""
    mapping = {
        "unsupervised-cl": (1.0, 0.0, 1.0, 0.0),
        "maxsim": (0.0, 0.0, 0.0, 1.0),
    }
    if loss_type not in mapping:
        raise ValueError(f"Unknown loss_type={loss_type!r}")
    recon, auxk, ucl, maxsim = mapping[loss_type]
    del auxk
    validate_loss_weights(
        sample_format,
        recon_coef=recon,
        auxk_coef=0.1 if loss_type == "unsupervised-cl" else 0.0,
        ucl_coef=ucl,
        maxsim_coef=maxsim,
    )


def load_hf_msmarco_triplets(split: str = "train") -> Dataset:
    """Load HuggingFace MS MARCO BM25 triplet dataset."""
    return load_dataset(
        "sentence-transformers/msmarco-bm25",
        "triplet",
        split=split,
    )


class CorpusLookup:
    """Lazy doc_id -> text lookup for MS MARCO corpus.jsonl (legacy fallback)."""

    def __init__(self, corpus_path: str | Path) -> None:
        self.corpus_path = Path(corpus_path)
        self._index_path = self.corpus_path.with_suffix(".id_index.json")
        self._index: dict[str, int] | None = None

    def _build_index(self) -> dict[str, int]:
        if self._index_path.is_file():
            with open(self._index_path, encoding="utf-8") as f:
                return json.load(f)

        logger.info("Building corpus index: %s", self.corpus_path)
        index: dict[str, int] = {}
        with open(self.corpus_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                index[_corpus_doc_id(row)] = line_no

        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)
        logger.info("Corpus index built: %d documents", len(index))
        return index

    def get(self, doc_id: str) -> str:
        if self._index is None:
            self._index = self._build_index()
        line_no = self._index[str(doc_id)]
        with open(self.corpus_path, encoding="utf-8") as f:
            for current, line in enumerate(f):
                if current == line_no:
                    row = json.loads(line)
                    return row["text"]
        raise KeyError(doc_id)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _doc_text(entry: dict | str, corpus: CorpusLookup | None) -> str | None:
    doc_id: str | None
    if isinstance(entry, dict):
        if entry.get("text"):
            return entry["text"]
        raw_id = entry.get("doc_id")
        doc_id = str(raw_id) if raw_id is not None else None
    else:
        doc_id = str(entry)

    if doc_id is None:
        return None
    if corpus is None:
        logger.debug("No corpus; cannot resolve doc_id=%s text", doc_id)
        return None
    try:
        return corpus.get(doc_id)
    except KeyError:
        logger.debug("doc_id=%s missing from corpus", doc_id)
        return None


def _iter_pair_rows(
    pairs_path: Path,
    *,
    corpus: CorpusLookup | None = None,
    max_samples: int | None = None,
) -> Iterator[dict[str, str]]:
    """Expand pairs/{split}.jsonl to query+positive rows (multiple positives per query)."""
    count = 0
    for row in _iter_jsonl(pairs_path):
        query = row["query"]
        qid = str(row.get("qid", ""))
        positives = row.get("positives") or []
        emitted = False
        for pos in positives:
            text = _doc_text(pos, corpus)
            if not text:
                continue
            yield {"qid": qid, "query": query, "positive": text}
            emitted = True
            count += 1
            if max_samples is not None and count >= max_samples:
                return
        if not emitted:
            logger.debug("pairs row qid=%s has no positive; skip", qid)


def _iter_triplet_rows(
    hard_negatives_path: Path,
    *,
    corpus: CorpusLookup | None = None,
    negative_rank: int = 0,
    max_samples: int | None = None,
) -> Iterator[dict[str, str]]:
    """Read query+positive+negative from hard_negatives/{split}.jsonl."""
    count = 0
    for row in _iter_jsonl(hard_negatives_path):
        qid = str(row.get("qid", ""))
        query = row["query"]
        positives = row.get("positives") or []
        negatives = row.get("negatives") or []
        if not positives or not negatives:
            continue

        positive_text = _doc_text(positives[0], corpus)
        rank = min(negative_rank, len(negatives) - 1)
        negative_text = _doc_text(negatives[rank], corpus)
        if not positive_text or not negative_text:
            continue

        yield {
            "qid": qid,
            "query": query,
            "positive": positive_text,
            "negative": negative_text,
        }
        count += 1
        if max_samples is not None and count >= max_samples:
            return


def _load_split_dataset(
    paths: MSMarcoPassagePaths,
    split: str,
    *,
    sample_format: SampleFormat,
    negative_rank: int = 0,
    max_samples: int | None = None,
) -> Dataset:
    corpus: CorpusLookup | None = None
    if sample_format == "pair":
        data_path = paths.pairs(split)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"sample_format=pair requires {data_path}."
                "Run first: python prepare_msmarco.py --skip-download --skip-index --skip-hard-negatives"
            )
        corpus = CorpusLookup(paths.corpus) if paths.corpus.is_file() else None
        logger.info("Loading pairs %s: %s", split, data_path)
        iterator = _iter_pair_rows(data_path, corpus=corpus, max_samples=max_samples)
        label = "pair"
    else:
        data_path = paths.hard_negatives(split)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"sample_format=triplet requires {data_path}."
                "Run first: python prepare_msmarco.py --skip-download --skip-index --skip-pairs"
            )
        # Fallback to corpus doc_id lookup when hard_negatives lack materialized text.
        corpus = CorpusLookup(paths.corpus) if paths.corpus.is_file() else None
        if corpus is None:
            raise FileNotFoundError(
                f"hard_negatives without text fields need corpus: {paths.corpus}"
            )
        logger.info("Loading hard_negatives %s: %s", split, data_path)
        iterator = _iter_triplet_rows(
            data_path,
            corpus=corpus,
            negative_rank=negative_rank,
            max_samples=max_samples,
        )
        label = "triplet"

    rows = list(iterator)
    if not rows:
        raise ValueError(
            f"split={split!r} ({label}) produced no samples; check {data_path} or run prepare_msmarco.py"
        )

    logger.info("split=%s %s samples: %d", split, label, len(rows))
    return Dataset.from_list(rows)


def load_msmarco_passage_datasets(
    data_dir: str | Path,
    *,
    sample_format: SampleFormat,
    train_split: str = "train",
    eval_split: str = "validation",
    negative_rank: int = 0,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Load train/eval datasets from prepare_msmarco.py outputs."""
    paths = MSMarcoPassagePaths.from_dir(data_dir)
    paths.validate(require_corpus=False)

    train_dataset = _load_split_dataset(
        paths,
        train_split,
        sample_format=sample_format,
        negative_rank=negative_rank,
        max_samples=max_train_samples,
    )
    eval_dataset = _load_split_dataset(
        paths,
        eval_split,
        sample_format=sample_format,
        negative_rank=negative_rank,
        max_samples=max_eval_samples,
    )
    return train_dataset, eval_dataset


def load_msmarco_hard_negatives(
    hard_negatives_path: str | Path,
    corpus_path: str | Path,
    *,
    negative_rank: int = 0,
    validation_split: float = 0.01,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Backward-compatible loader for a single hard_negatives JSONL file."""
    corpus = CorpusLookup(corpus_path)
    rows = list(
        _iter_triplet_rows(
            Path(hard_negatives_path),
            corpus=corpus,
            negative_rank=negative_rank,
        )
    )
    dataset = Dataset.from_list(rows)
    if validation_split <= 0 or len(dataset) < 2:
        return dataset, None

    split = dataset.train_test_split(test_size=validation_split, seed=seed)
    return split["train"], split["test"]
