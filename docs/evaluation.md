# Evaluation

Evaluation is handled by:

```bash
python -m ssr.retrieval.eval_mteb [options]
```

It supports two tasks:

- `--task index`: stream corpus text through the encoder and build an on-disk global inverted index.
- `--task retrieval`: run query retrieval and compute IR metrics.

## Method Names

| Variant | Meaning |
| --- | --- |
| `--variant ssr` | Token-level exact MaxSim. No `[CLS]` channel. |
| `--variant ssr-cls` | Token-level exact MaxSim plus a separate `[CLS]` SAE channel. Requires `--cls-sae-path`. |
| `--variant ssr++` | Pruned retrieval. Uses the sparse-cache pruned retriever or e2e-index two-phase pruning. |

If you do not pass `--variant`, the lower-level flags such as `--mode`, `--index-two-phase`, and `--cls-sae-path` are used directly.

With `--cls-sae-path` and pruning enabled, logs report the effective method as `SSR-CLS++`.

## Index Build

For MS MARCO passage/document evaluation, first prepare the relevant subset with `prepare_msmarco.py --subset passage --skip-pairs --skip-hard-negatives` or `prepare_msmarco.py --subset document --skip-pairs --skip-hard-negatives`, then pass that subset directory as `--data-dir`.

Build an e2e index for token-only `SSR`:

```bash
python -m ssr.retrieval.eval_mteb \
  --task index \
  --variant ssr \
  --model-path output/ssr-token/final \
  --dataset nfcorpus \
  --data-dir data/processed/mteb \
  --index-cache-dir data/cache/mteb_index_e2e/nfcorpus \
  --encode-device cuda:0
```

Build an e2e index for `SSR-CLS`:

```bash
python -m ssr.retrieval.eval_mteb \
  --task index \
  --variant ssr-cls \
  --model-path output/ssr-token/final \
  --cls-sae-path output/ssr-cls/final \
  --dataset nfcorpus \
  --data-dir data/processed/mteb \
  --index-cache-dir data/cache/mteb_index_e2e/nfcorpus_cls \
  --encode-device cuda:0
```

The CLS path is implemented in one transformer pass per batch: token SAE rows and `[CLS]` SAE rows are encoded together, then CLS latent ids are offset into a separate latent range. Token-only `SSR` does not instantiate or call the CLS encoder, so its fast path is unchanged.

Index options:

| Option | Meaning |
| --- | --- |
| `--task index` | Build an index only. |
| `--model-path PATH` | Token SAE SSR checkpoint. |
| `--cls-sae-path PATH` | Optional separate CLS SAE checkpoint. |
| `--cls-topk N` | CLS SAE top-k. Defaults to the CLS checkpoint config. |
| `--dataset SLUG` | Single dataset slug. |
| `--datasets ...` | Multiple dataset slugs. |
| `--data-dir PATH` | Prepared evaluation data root. |
| `--index-cache-dir PATH` | Index output root. |
| `--encode-device DEVICE` | Encoding device, such as `cuda:0` or `cpu`. |
| `--encode-batch-size N` | Corpus encode batch size. |
| `--block-size N` | Latent block size for the inverted index. |
| `--latent-reorder {none,frequency,cooc}` | Reorder latent ids for better index locality. |
| `--cooc-sample-rate FLOAT` | Sampling rate for co-occurrence reorder. |
| `--force-rebuild-index` | Ignore existing matching cache. |
| `--max-corpus N` | Limit corpus size for debugging. |

## Retrieval Backends

### Sparse Cache

The sparse-cache backend stores corpus sparse embeddings in `.npz`, then builds chunk indexes during retrieval.

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend sparse-cache \
  --variant ssr \
  --model-path output/ssr-token/final \
  --dataset nfcorpus \
  --data-dir data/processed/mteb \
  --cache-dir data/cache/mteb_sparse \
  --score-device index
```

Pre-encode corpus only:

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend sparse-cache \
  --encode-corpus-only \
  --model-path output/ssr-token/final \
  --dataset nfcorpus \
  --data-dir data/processed/mteb
```

### E2E Index

The e2e-index backend loads a pre-built global index and only encodes queries at retrieval time.

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend e2e-index \
  --variant ssr \
  --model-path output/ssr-token/final \
  --dataset nfcorpus \
  --data-dir data/processed/mteb \
  --index-cache-dir data/cache/mteb_index_e2e/nfcorpus \
  --score-device index
```

For `SSR-CLS`, pass the same `--cls-sae-path` used during index build, and use a CLS-specific index cache directory.

## CPU and GPU Modes

CPU exact `SSR`:

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --variant ssr \
  --score-device index \
  --index-accum-device cpu \
  ...
```

GPU/hybrid exact `SSR`:

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --variant ssr \
  --score-device index \
  --index-accum-device hybrid \
  --gpu-hot-budget-gb 16 \
  ...
```

CPU pruned `SSR++` on sparse-cache:

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend sparse-cache \
  --variant ssr++ \
  --score-device cpu \
  ...
```

GPU/hybrid pruned `SSR++` on e2e-index:

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend e2e-index \
  --variant ssr++ \
  --score-device index \
  --index-accum-device hybrid \
  ...
```

Pruning options:

| Option | Meaning |
| --- | --- |
| `--variant ssr++` | Enable the pruning preset. |
| `--mode pruned` | Lower-level sparse-cache pruned retriever. |
| `--index-two-phase` | Lower-level e2e-index two-phase pruning. |
| `--prune-topk N` | Query latent top-k used in pruned sparse-cache mode. |
| `--n-candidates N` | Candidate count before reranking in sparse-cache mode. |
| `--two-phase-pool-size N` | Candidate pool for e2e-index two-phase retrieval. |
| `--coarse-topk N` | Coarse postings per token for two-phase scoring. |
| `--query-latent-top-k N` | Optional cap on query latents per token. |

## CLS Options

| Option | Meaning |
| --- | --- |
| `--variant ssr-cls` | Use the public exact CLS preset. |
| `--cls-sae-path PATH` | Load a separate SAE for `[CLS]` embeddings. |
| `--cls-topk N` | Override CLS SAE top-k. |

The CLS SAE path can point directly to a `SparseAutoencoder` module directory, or to a checkpoint directory containing one. The loader scans one level of child directories when needed.

## Metrics

MS MARCO-style slugs default to:

```text
mrr@10
```

Other retrieval slugs default to:

```text
ndcg@10
```

Override metric cutoffs with:

| Option | Meaning |
| --- | --- |
| `--mrr-k 10` | Compute MRR at the requested cutoffs. |
| `--ndcg-k 10` | Compute nDCG at the requested cutoffs. |
| `--recall-k 1 10 100` | Compute recall/accuracy at cutoffs. |
| `--map-k 100` | Compute MAP at cutoffs. |
| `--top-k N` | Final retrieval depth and metric heap size floor. |

## Useful Script Wrappers

The `scripts/` directory contains thin wrappers around `eval_mteb.py`:

| Script | Purpose |
| --- | --- |
| `scripts/e2e_build_index.sh` | Build an e2e index. |
| `scripts/e2e_retrieval_cpu.sh` | E2E retrieval with CPU index accumulation. |
| `scripts/e2e_retrieval_gpu.sh` | E2E retrieval with hybrid GPU accumulation. |
| `scripts/sparse_retrieval_cpu.sh` | Sparse-cache retrieval with CPU index accumulation. |
| `scripts/sparse_retrieval_gpu.sh` | Sparse-cache retrieval with hybrid GPU accumulation. |
| `scripts/cache_mteb_embeddings.sh` | Encode corpus embeddings only. |

All wrappers forward extra arguments to `python -m ssr.retrieval.eval_mteb`.
