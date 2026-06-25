# Data Preprocessing

This repository uses two preprocessing pipelines:

- `prepare_msmarco.py` for MS MARCO training data and MS MARCO evaluation data.
- `prepare_mteb_eval.py` for retrieval evaluation data.

## Training Data

Training data comes from MS MARCO passage. Use `prepare_msmarco.py --subset passage`.

Run the full passage pipeline:

```bash
python prepare_msmarco.py \
  --subset passage \
  --raw-dir data/raw/msmarco \
  --processed-dir data/processed/msmarco
```

The output layout is:

```text
data/processed/msmarco/
  passage/
    corpus.jsonl
    queries/{train,validation,test}.tsv
    qrels/{train,validation}.tsv
    pairs/{train,validation}.jsonl
    hard_negatives/{train,validation}.jsonl
```

The preprocessing stages are:

1. `index`: writes corpus, queries, and qrels.
2. `pairs`: materializes query-positive text pairs from qrels.
3. `hard_negatives`: materializes query-positive-negative triplets.

For passage hard negatives, the script prefers the Hugging Face `sentence-transformers/msmarco-hard-negatives` BM25 file, then fills missing queries with local BM25.

Useful options:

| Option | Meaning |
| --- | --- |
| `--subset passage` | Prepare MS MARCO passage. Use this for training. |
| `--raw-dir PATH` | Raw download/cache directory. |
| `--processed-dir PATH` | Processed output root. |
| `--num-hard-negatives N` | Number of negatives per query. |
| `--retrieve-k N` | BM25 retrieval depth for mining negatives. |
| `--hf-hard-negatives-path PATH` | Use an existing HF passage hard-negative file. |
| `--skip-hf-hard-negatives` | Build all passage negatives locally with BM25. |
| `--skip-download` | Reuse existing raw files. |
| `--skip-index` | Skip corpus/query/qrel generation. |
| `--skip-pairs` | Skip pair generation. |
| `--skip-hard-negatives` | Skip triplet generation. |

## Evaluation Data

### MS MARCO Passage and Document

MS MARCO passage and document can be prepared for evaluation with the index stage of `prepare_msmarco.py`. For evaluation you only need `corpus.jsonl`, `queries/`, and `qrels/`, so skip pair and hard-negative generation:

```bash
python prepare_msmarco.py \
  --subset passage \
  --raw-dir data/raw/msmarco \
  --processed-dir data/processed/msmarco \
  --skip-pairs \
  --skip-hard-negatives
```

```bash
python prepare_msmarco.py \
  --subset document \
  --raw-dir data/raw/msmarco \
  --processed-dir data/processed/msmarco \
  --skip-pairs \
  --skip-hard-negatives
```

The output layout is:

```text
data/processed/msmarco/
  passage/
    corpus.jsonl
    queries/{train,validation,test}.tsv
    qrels/{train,validation}.tsv
  document/
    corpus.jsonl
    queries/{train,validation,test}.tsv
    qrels/{train,validation}.tsv
```

Use the corresponding subset directory as `--data-dir` when evaluating MS MARCO. MS MARCO-style slugs default to `MRR@10`.

### MTEB / BEIR

Evaluation data is prepared with:

```bash
python prepare_mteb_eval.py \
  --processed-dir data/processed/mteb
```

The script supports these MTEB/BEIR-style retrieval slugs:

```text
arguana
climate-fever
trec-covid
dbpedia
fever
fiqa
hotpotqa
nfcorpus
nq
quora
scidocs
scifact
touche2020
```

Prepare selected datasets:

```bash
python prepare_mteb_eval.py \
  --datasets nfcorpus scifact hotpotqa \
  --split test \
  --processed-dir data/processed/mteb
```

The output layout is:

```text
data/processed/mteb/
  nfcorpus/
    corpus.jsonl
    queries/test.tsv
    qrels/test.tsv
  scifact/
    ...
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--datasets ...` | Dataset slugs to prepare. Omit to prepare all supported slugs. |
| `--split ...` | Split names to write. Default is `test`. |
| `--processed-dir PATH` | Evaluation data output root. |
| `--verbose` | Enable debug logging. |

## Dataset Metrics

During evaluation, MS MARCO-style slugs use `MRR@10` by default. Other supported retrieval datasets use `nDCG@10` by default. You can override this with `--mrr-k` and `--ndcg-k` in the evaluation command.
