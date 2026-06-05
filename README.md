# SSR — Sparse Semantic Retrieval

Cloud release bundle: ColBERT backbone + sparse autoencoder (SAE) for data prep, training, and retrieval. No 1M synthetic benchmark.

---

## Layout

```
final_update/
├── README.md
├── pyproject.toml
├── install_env_cu128.sh
├── prepare_mteb_eval.py      # MTEB eval data
├── prepare_msmarco.py        # MS MARCO training data
├── ssr/
│   ├── model.py              # SSR model
│   ├── train.py              # Training
│   └── retrieval/            # E2E and sparse-cache retrieval
├── scripts/                  # Shell helpers
└── tests/
```

`pylate` (ColBERT training backbone) is installed from PyPI via `uv sync` / `pip install pylate>=1.5.0`, not vendored in this repo.

Create `data/` and `output/` at runtime (listed in `.gitignore`).

---

## Environment

```bash
cd final_update
bash install_env_cu128.sh
# Or: uv sync && uv pip install -e ".[eval,data]"
```

For GPU retrieval, install eval extras: `pip install -e ".[eval]"` (includes numba).

---

## 1. Raw data preparation

### MTEB (BEIR layout, retrieval eval)

```bash
.venv/bin/python prepare_mteb_eval.py \
  --processed-dir data/processed/mteb \
  --datasets nfcorpus hotpotqa
```

Per dataset: `corpus.jsonl`, `queries/{split}.tsv`, `qrels/{split}.tsv`.

### MS MARCO (SSR training)

```bash
.venv/bin/python prepare_msmarco.py --help
```

See script help for `data/processed/msmarco/` (passage, pairs, hard negatives, etc.).

---

## 4. SSR training

```bash
.venv/bin/python -m ssr.train \
  --gpu 0 \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --data-dir data/processed/msmarco \
  --output-dir output/my-ssr-run \
  --report-to wandb
```

Checkpoints under `output/.../final/` (`config.json`, `1_SparseAutoencoder/`). Use `--model-path` on that dir or its parent (auto-resolves `final/`).

---

## 2. Streaming E2E: index + retrieval (no full embedding bank on disk)

Code: `ssr/retrieval/streaming_index_build.py`, `eval_mteb.py --task index|retrieval --corpus-backend e2e-index`.

### Build index (GPU encode → CPU global inverted index)

```bash
bash scripts/e2e_build_index.sh \
  --model-path output/my-ssr-run/final \
  --dataset nfcorpus \
  --index-cache-dir data/cache/mteb_index_e2e/nfcorpus \
  --encode-device cuda:0 \
  --encode-batch-size 8
```

Output: `index-cache-dir/global_index_v2_bs512_frequency/` + `doc_id_map.json`.

### Retrieval eval — CPU

```bash
bash scripts/e2e_retrieval_cpu.sh \
  --model-path output/my-ssr-run/final \
  --dataset nfcorpus \
  --index-cache-dir data/cache/mteb_index_e2e/nfcorpus
```

### Retrieval eval — GPU (hybrid accumulate)

```bash
bash scripts/e2e_retrieval_gpu.sh \
  --model-path output/my-ssr-run/final \
  --dataset nfcorpus \
  --index-cache-dir data/cache/mteb_index_e2e/nfcorpus
```

Optional two-phase pruning (faster, approximate):

```bash
  --index-two-phase --two-phase-pool-size 5000 --coarse-topk 8
```

`--block-size` and `--latent-reorder` must match between index build and retrieval.

---

## 3. Cache embeddings first, then index + retrieval

Embeddings saved to `data/cache/mteb_sparse/{slug}/corpus_{topk}k_{split}.npz`; retrieval builds per-chunk inverted indexes.

### 3.1 Encode and cache corpus only (no queries)

```bash
bash scripts/cache_mteb_embeddings.sh \
  --model-path output/my-ssr-run/final \
  --dataset nfcorpus \
  --cache-dir data/cache/mteb_sparse \
  --encode-device cuda:0
```

Force re-encode: add `--force-reencode`.

Cache example: `data/cache/mteb_sparse/nfcorpus/corpus_32k_test.npz`.

### 3.2 Retrieval eval — CPU

```bash
bash scripts/sparse_retrieval_cpu.sh \
  --model-path output/my-ssr-run/final \
  --dataset nfcorpus \
  --cache-dir data/cache/mteb_sparse
```

(If cache is missing, corpus is encoded automatically before retrieval.)

### 3.3 Retrieval eval — GPU

```bash
bash scripts/sparse_retrieval_gpu.sh \
  --model-path output/my-ssr-run/final \
  --dataset nfcorpus \
  --cache-dir data/cache/mteb_sparse
```

Optional: `--index-two-phase`, `--index-accum-device hybrid` (GPU script defaults to hybrid).

---

## Unified CLI

You can also call:

```bash
.venv/bin/python -m ssr.retrieval.eval_mteb --help
```

| Use case | Key flags |
|------|----------|
| E2E index build | `--task index` |
| E2E retrieval | `--task retrieval --corpus-backend e2e-index` |
| Embedding cache only | `--encode-corpus-only --corpus-backend sparse-cache --cache-dir ...` |
| Sparse-cache retrieval | `--task retrieval --corpus-backend sparse-cache --cache-dir ...` |

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## Migration from MVR-update

| Old | New |
|----|-----|
| `ColBERTWithSAE` | `SSR` |
| `python -m training.train` | `python -m ssr.train` |
| `python -m training.eval.eval_mteb` | `python -m ssr.retrieval.eval_mteb` |

Existing `output/.../final` checkpoints load as-is; no file renames needed.
