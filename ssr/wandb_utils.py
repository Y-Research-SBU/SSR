"""Weights & Biases helpers for ColBERT + SAE training."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from transformers import TrainerCallback

__all__ = ["WandbConfigCallback", "log_wandb_metric"]


def log_wandb_metric(name: str, value: float, step: int | None = None) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log({name: value}, step=step)


def _collect_hyperparameters(
    args: Namespace,
    *,
    model,
    initial_topk: int,
    device_info: dict | None = None,
) -> dict[str, Any]:
    sae = model.sae_module
    hparams = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "data_dir": str(getattr(args, "data_dir", "")),
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "negative_rank": getattr(args, "negative_rank", None),
        "max_train_samples": getattr(args, "max_train_samples", None),
        "max_eval_samples": getattr(args, "max_eval_samples", None),
        "sample_format": getattr(args, "sample_format", None),
        "use_hard_negatives": getattr(args, "sample_format", None) == "triplet",
        "hard_negatives_path": str(getattr(args, "hard_negatives_path", "") or ""),
        "corpus_path": str(getattr(args, "corpus_path", "") or ""),
        "validation_split": args.validation_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "compile": args.compile,
        "gpu": args.gpu,
        "device": args.device,
        "n_latents": sae.n_latents,
        "sae_in_features": sae.in_features,
        "topk": args.topk,
        "initial_topk": initial_topk,
        "auxk": args.auxk,
        "dead_threshold": args.dead_threshold,
        "normalize_input": args.normalize_input,
        "recon_coef": args.recon_coef,
        "auxk_coef": args.auxk_coef,
        "ucl_coef": args.ucl_coef,
        "maxsim_coef": args.maxsim_coef,
        "ucl_temperature": args.ucl_temperature,
        "maxsim_temperature": args.maxsim_temperature,
        "k_decay_ratio": args.k_decay_ratio,
        "query_length": getattr(model, "query_length", None),
        "document_length": getattr(model, "document_length", None),
    }
    if device_info:
        hparams.update(device_info)
    return hparams


class WandbConfigCallback(TrainerCallback):
    """Log CLI / SAE hyperparameters to wandb at train start."""

    def __init__(
        self,
        args: Namespace,
        *,
        model,
        initial_topk: int,
        wandb_tags: list[str] | None = None,
        device_info: dict | None = None,
    ) -> None:
        self.hparams = _collect_hyperparameters(
            args,
            model=model,
            initial_topk=initial_topk,
            device_info=device_info,
        )
        self.wandb_tags = wandb_tags or []

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.config.update(self.hparams, allow_val_change=True)
        if self.wandb_tags:
            wandb.run.tags = tuple(set(wandb.run.tags or ()) | set(self.wandb_tags))
