#!/usr/bin/env python3
"""Train SSR (ColBERT + Sparse Autoencoder) via PyLate."""

from __future__ import annotations

from ssr import _bootstrap  # noqa: F401  — cuda/venv bootstrap before torch

import argparse
import logging
from pathlib import Path

from transformers import TrainerCallback

from ssr.model import build_ssr
from ssr.dataset import (
    load_hf_msmarco_triplets,
    load_msmarco_hard_negatives,
    load_msmarco_passage_datasets,
    validate_loss_weights,
)
from ssr.device_utils import configure_cuda_env, get_device_info, resolve_training_device
from ssr.losses import CombinedSAELoss, LossWeights
from ssr.sparse_autoencoder import get_current_k
from ssr.wandb_utils import WandbConfigCallback, log_wandb_metric

logger = logging.getLogger(__name__)


class TopKAnnealingCallback(TrainerCallback):
    """K-annealing callback adapted from CSRv2/text/CSR_training.py."""

    def __init__(
        self,
        *,
        initial_topk: int,
        final_topk: int,
        k_decay_ratio: float = 0.7,
    ) -> None:
        self.initial_topk = initial_topk
        self.final_topk = final_topk
        self.k_decay_ratio = k_decay_ratio

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        st_model = model.module if hasattr(model, "module") else model
        if not hasattr(st_model, "sae_module"):
            return

        current_k = get_current_k(
            global_step=state.global_step,
            initial_k=self.initial_topk,
            final_k=self.final_topk,
            total_steps=state.max_steps,
            k_decay_ratio=self.k_decay_ratio,
        )
        st_model.sae_module.set_topk(current_k)

        if state.global_step % max(args.logging_steps, 1) == 0:
            logger.info("Step %d: SAE topk=%d", state.global_step, current_k)
            log_wandb_metric("sae/topk", float(current_k), step=state.global_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SSR (ColBERT + Sparse Autoencoder projector)."
    )

    # Model
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="HuggingFace base encoder.",
    )
    parser.add_argument(
        "--n-latents",
        type=int,
        default=16384,
        help="SAE latent dimension (default: 4 * hidden_size).",
    )
    parser.add_argument("--topk", type=int, default=32, help="Final Top-K sparsity.")
    parser.add_argument(
        "--initial-topk",
        type=int,
        default=None,
        help="Initial Top-K for annealing (default: 2 * topk).",
    )
    parser.add_argument("--auxk", type=int, default=1024, help="AuxK dead-neuron revival count.")
    parser.add_argument(
        "--dead-threshold",
        type=int,
        default=30,
        help="Steps before a latent is considered dead.",
    )
    parser.add_argument(
        "--normalize-input",
        action="store_true",
        help="Apply layer norm to SAE inputs (CSR normalize flag).",
    )
    parser.add_argument(
        "--sae-token-scope",
        choices=("all", "non-cls", "cls"),
        default="non-cls",
        help=(
            "Which encoder token embeddings train the SAE: non-cls (default), "
            "cls ([CLS] only), or all."
        ),
    )

    # Loss weights — exactly four terms; coef=0 skips that term at train time.
    parser.add_argument(
        "--recon-coef",
        type=float,
        default=1.0,
        help="Top-K SAE reconstruction loss weight; 0 disables this term.",
    )
    parser.add_argument(
        "--auxk-coef",
        type=float,
        default=0.1,
        help="AuxK dead-neuron revival loss weight; 0 disables this term.",
    )
    parser.add_argument(
        "--ucl-coef",
        type=float,
        default=0.1,
        help="Unsupervised in-batch contrastive loss weight; 0 disables this term.",
    )
    parser.add_argument(
        "--maxsim-coef",
        type=float,
        default=0.0,
        help="MaxSim supervised contrastive loss weight; 0 disables. >0 requires triplet data.",
    )
    parser.add_argument(
        "--ucl-temperature",
        type=float,
        default=0.2,
        help="Unsupervised CL temperature (--ucl-coef > 0 only).",
    )
    parser.add_argument(
        "--maxsim-temperature",
        type=float,
        default=1.0,
        help="MaxSim contrastive temperature (--maxsim-coef > 0 only).",
    )
    parser.add_argument(
        "--k-decay-ratio",
        type=float,
        default=0.7,
        help="[initial-topk only] fraction of training for K anneal completion.",
    )

    # Data
    parser.add_argument(
        "--dataset",
        choices=("msmarco-passage", "hf", "msmarco"),
        default="msmarco-passage",
        help="Training data source. Default: local msmarco-passage from prepare_msmarco.py.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/msmarco/passage"),
        help="MS MARCO passage data root (prepare_msmarco.py output).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Train split name (file under pairs/ or hard_negatives/).",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Eval split name.",
    )
    parser.add_argument(
        "--negative-rank",
        type=int,
        default=0,
        help="Which hard negative index to use (0=hardest).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training samples (debug).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap eval samples (debug).",
    )
    parser.add_argument(
        "--hard-negatives-path",
        type=Path,
        default=None,
        help="[legacy --dataset msmarco] hard negatives JSONL path.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="[legacy --dataset msmarco] corpus.jsonl path.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.01,
        help="[legacy --dataset msmarco/hf] validation fraction from train.",
    )
    parser.add_argument(
        "--sample-format",
        choices=("pair", "triplet"),
        default=None,
        help=(
            "Training sample format: pair loads query+positive from pairs/;"
            "triplet loads query+positive+negative from hard_negatives/."
        ),
    )

    # Training
    parser.add_argument("--output-dir", type=Path, default=Path("output/ssr"))
    parser.add_argument("--run-name", default="ssr-bert-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size per device (default: same as --batch-size).",
    )
    parser.add_argument(
        "--skip-triplet-eval",
        action="store_true",
        help=(
            "Skip ColBERTTripletEvaluator (avoids OOM from encoding full eval set on GPU). "
            "Trainer eval loss is still computed when eval_strategy is enabled."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Use a single GPU by index (0, 1, ...). Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Training device, e.g. "cuda:0" or "cpu". Ignored when --gpu is set.',
    )

    # Weights & Biases
    parser.add_argument(
        "--report-to",
        default="wandb",
        help='Logging integrations, e.g. "wandb", "tensorboard", "none", or "all".',
    )
    parser.add_argument(
        "--wandb-project",
        default="ssr",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity (team or username). Uses default if omitted.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional W&B run tags.",
    )

    return parser.parse_args()


def resolve_sample_format(args: argparse.Namespace) -> str:
    if args.sample_format is not None:
        return args.sample_format
    return "triplet"


def load_datasets(args: argparse.Namespace, sample_format: str):
    if args.dataset == "hf":
        train_dataset = load_hf_msmarco_triplets("train")
        split = train_dataset.train_test_split(
            test_size=args.validation_split,
            seed=args.seed,
        )
        return split["train"], split["test"]

    if args.dataset == "msmarco-passage":
        return load_msmarco_passage_datasets(
            args.data_dir,
            sample_format=sample_format,
            train_split=args.train_split,
            eval_split=args.eval_split,
            negative_rank=args.negative_rank,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )

    if sample_format != "triplet":
        raise ValueError(
            "legacy --dataset msmarco only supports sample_format=triplet."
            "Use --dataset msmarco-passage instead."
        )
    hard_negatives_path = args.hard_negatives_path or (
        args.data_dir / "hard_negatives" / f"{args.train_split}.jsonl"
    )
    corpus_path = args.corpus_path or (args.data_dir / "corpus.jsonl")
    train_dataset, eval_dataset = load_msmarco_hard_negatives(
        hard_negatives_path,
        corpus_path,
        negative_rank=args.negative_rank,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if eval_dataset is None:
        raise ValueError("Local dataset too small for train/eval split.")
    return train_dataset, eval_dataset


def main(args: argparse.Namespace) -> None:
    import torch
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    from pylate import evaluation, utils

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu is not None and args.device is not None:
        raise ValueError("Use only one of --gpu or --device, not both.")

    device = resolve_training_device(args)
    device_info = get_device_info()
    logger.info("Training device: %s (%s)", device, device_info)

    initial_topk = args.initial_topk

    logger.info("Building SSR model from %s", args.model_name)
    model = build_ssr(
        args.model_name,
        n_latents=args.n_latents,
        topk=args.topk,
        initial_topk=initial_topk,
        auxk=args.auxk,
        normalize=args.normalize_input,
        dead_threshold=args.dead_threshold,
        token_scope=args.sae_token_scope,
        device=device,
    )
    if initial_topk is not None and initial_topk != args.topk:
        model.sae_module.set_topk(initial_topk)
    if args.compile:
        model = torch.compile(model)

    sample_format = resolve_sample_format(args)
    args.sample_format = sample_format

    loss_weights = LossWeights(
        recon=args.recon_coef,
        auxk=args.auxk_coef,
        ucl=args.ucl_coef,
        maxsim=args.maxsim_coef,
    )
    validate_loss_weights(
        sample_format,
        recon_coef=loss_weights.recon,
        auxk_coef=loss_weights.auxk,
        ucl_coef=loss_weights.ucl,
        maxsim_coef=loss_weights.maxsim,
    )
    logger.info(
        "sample_format=%s | loss weights: recon=%s auxk=%s ucl=%s maxsim=%s",
        sample_format,
        loss_weights.recon,
        loss_weights.auxk,
        loss_weights.ucl,
        loss_weights.maxsim,
    )
    logger.info("SAE token scope: %s", args.sae_token_scope)

    train_dataset, eval_dataset = load_datasets(args, sample_format)
    text_columns = ["query", "positive"]
    if sample_format == "triplet":
        text_columns.append("negative")
    train_dataset = train_dataset.select_columns(text_columns)
    eval_dataset = eval_dataset.select_columns(text_columns)
    logger.info("Train samples: %d, eval samples: %d", len(train_dataset), len(eval_dataset))

    required_columns = {"query", "positive"}
    if sample_format == "triplet":
        required_columns.add("negative")
    missing = required_columns - set(train_dataset.column_names)
    if missing:
        raise ValueError(
            f"Train dataset missing columns {sorted(missing)}; columns: {train_dataset.column_names}."
            f"Check sample_format={sample_format} matches data files."
        )

    train_loss = CombinedSAELoss(
        model=model,
        weights=loss_weights,
        ucl_temperature=args.ucl_temperature,
        maxsim_temperature=args.maxsim_temperature,
    )

    eval_batch_size = (
        args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    )

    dev_evaluator = None
    if (
        loss_weights.maxsim > 0
        and sample_format == "triplet"
        and not args.skip_triplet_eval
    ):
        dev_evaluator = evaluation.ColBERTTripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            batch_size=eval_batch_size,
        )
    elif loss_weights.maxsim > 0 and args.skip_triplet_eval:
        logger.info(
            "skip_triplet_eval: skipping ColBERTTripletEvaluator (trainer eval loss kept)"
        )
    elif loss_weights.maxsim > 0:
        logger.info("maxsim loss enabled but eval needs triplet data; skipping ColBERTTripletEvaluator")
    elif sample_format == "pair":
        logger.info("sample_format=pair: skipping ColBERTTripletEvaluator (needs negative column)")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        fp16=args.fp16,
        bf16=args.bf16,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        report_to=[] if args.report_to == "none" else args.report_to,
    )

    if args.report_to != "none":
        import os

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    callbacks = [
        WandbConfigCallback(
            args,
            model=model,
            initial_topk=initial_topk if initial_topk is not None else args.topk,
            wandb_tags=args.wandb_tags,
            device_info=device_info,
        )
    ]
    if initial_topk is not None and initial_topk != args.topk:
        callbacks.append(
            TopKAnnealingCallback(
                initial_topk=initial_topk,
                final_topk=args.topk,
                k_decay_ratio=args.k_decay_ratio,
            )
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=callbacks,
    )

    logger.info("Starting training ...")
    trainer.train()

    final_dir = args.output_dir / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved model to %s", final_dir)


if __name__ == "__main__":
    cli_args = parse_args()
    configure_cuda_env(cli_args)
    main(cli_args)
