# Training

Training builds an SSR checkpoint by adding a Sparse Autoencoder (SAE) projection on top of a ColBERT-style encoder. The documented training path uses MS MARCO passage data.

The main entry point is:

```bash
python -m ssr.train [options]
```

## Token SAE: SSR

`SSR` uses only regular, non-`[CLS]` token embeddings for SAE training:

```bash
python -m ssr.train \
  --dataset msmarco-passage \
  --data-dir data/processed/msmarco/passage \
  --sae-token-scope non-cls \
  --sample-format triplet \
  --output-dir output/ssr-token
```

This is the standard token-level model used as `--model-path` in evaluation.

## CLS SAE: SSR-CLS

`SSR-CLS` adds a separate SAE trained only on `[CLS]` embeddings:

```bash
python -m ssr.train \
  --dataset msmarco-passage \
  --data-dir data/processed/msmarco/passage \
  --sae-token-scope cls \
  --sample-format triplet \
  --output-dir output/ssr-cls
```

Evaluation loads this checkpoint with `--cls-sae-path`. The token SAE and CLS SAE are separate models, and retrieval keeps their latent spaces separate by offsetting CLS latent ids.

## Token Scope

| Option | Meaning |
| --- | --- |
| `--sae-token-scope non-cls` | Train on regular tokens only. This is the default and corresponds to `SSR`. |
| `--sae-token-scope cls` | Train on `[CLS]` only. Use this to produce the CLS SAE for `SSR-CLS`. |
| `--sae-token-scope all` | Train on all attended tokens, including `[CLS]`. Useful for ablations. |

## Data Options

| Option | Meaning |
| --- | --- |
| `--dataset msmarco-passage` | Training data source for the documented setup; uses local MS MARCO passage output from `prepare_msmarco.py`. |
| `--data-dir PATH` | MS MARCO passage processed root, usually `data/processed/msmarco/passage`. |
| `--sample-format {pair,triplet}` | `pair` loads query-positive pairs; `triplet` loads query-positive-negative triplets. |
| `--train-split NAME` | Train split name. Default is `train`. |
| `--eval-split NAME` | Eval split name. Default is `validation`. |
| `--negative-rank N` | Which hard negative to select from triplet data. |
| `--max-train-samples N` | Cap train examples for debugging. |
| `--max-eval-samples N` | Cap eval examples for debugging. |
| `--validation-split FLOAT` | Fraction used for validation for legacy/HF loaders. |

## SAE Options

| Option | Meaning |
| --- | --- |
| `--model-name NAME` | Hugging Face base encoder. Default is `bert-base-uncased`. |
| `--n-latents N` | SAE latent dimension. |
| `--topk N` | Final top-k sparse activations. |
| `--initial-topk N` | Optional initial top-k for annealing. |
| `--auxk N` | AuxK dead-neuron revival count. |
| `--dead-threshold N` | Steps before a latent is treated as dead. |
| `--normalize-input` | Apply layer normalization before SAE encoding. |

## Loss Options

| Option | Meaning |
| --- | --- |
| `--recon-coef FLOAT` | Reconstruction loss weight. |
| `--auxk-coef FLOAT` | AuxK loss weight. |
| `--ucl-coef FLOAT` | Unsupervised in-batch contrastive loss weight. |
| `--maxsim-coef FLOAT` | Supervised MaxSim contrastive loss weight. Requires triplet data when enabled. |
| `--ucl-temperature FLOAT` | Temperature for unsupervised contrastive loss. |
| `--maxsim-temperature FLOAT` | Temperature for MaxSim contrastive loss. |
| `--k-decay-ratio FLOAT` | Fraction of training used to anneal from `--initial-topk` to `--topk`. |

## Runtime Options

| Option | Meaning |
| --- | --- |
| `--output-dir PATH` | Training output root. Final checkpoint is written to `PATH/final`. |
| `--run-name NAME` | Trainer/W&B run name. |
| `--epochs N` | Number of epochs. |
| `--batch-size N` | Per-device train batch size. |
| `--eval-batch-size N` | Per-device eval batch size. Defaults to train batch size. |
| `--learning-rate FLOAT` | Learning rate. |
| `--fp16` | Enable fp16 training. |
| `--bf16` | Enable bf16 training. |
| `--compile` | Use `torch.compile`. |
| `--gpu N` | Set `CUDA_VISIBLE_DEVICES` to one GPU. |
| `--device DEVICE` | Explicit training device, such as `cuda:0` or `cpu`. |
| `--report-to NAME` | Logging backend, for example `wandb` or `none`. |

## Outputs

The important artifact is:

```text
output/.../final/
```

Use the token SAE checkpoint as:

```bash
--model-path output/ssr-token/final
```

Use the CLS SAE checkpoint as:

```bash
--cls-sae-path output/ssr-cls/final
```
