"""SAE reconstruction + unsupervised / MaxSim contrastive losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from pylate.losses.contrastive import extract_skiplist_mask
from pylate.models import ColBERT
from pylate.scores import ColBERTScores
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from torch import Tensor

from .sparse_autoencoder import normalized_mse

__all__ = [
    "CombinedSAELoss",
    "ContrastiveWithSAE",
    "LossWeights",
    "SAEWithUnsupervisedCL",
    "compute_sae_loss",
    "compute_sae_loss_components",
    "pool_sparse_latents",
    "unsupervised_inbatch_cl",
]


@dataclass(frozen=True)
class LossWeights:
    """Four independent loss coefficients configured from the CLI."""

    recon: float = 1.0
    auxk: float = 0.1
    ucl: float = 0.0
    maxsim: float = 0.0

    def any_sae_loss(self) -> bool:
        return self.recon > 0.0 or self.auxk > 0.0

    def any_active(self) -> bool:
        return self.recon > 0.0 or self.auxk > 0.0 or self.ucl > 0.0 or self.maxsim > 0.0


def pool_sparse_latents(
    token_embeddings: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Mean-pool sparse token latents into one vector per sentence."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def unsupervised_inbatch_cl(
    latents_view_a: Tensor,
    latents_view_b: Tensor,
    *,
    temperature: float = 0.2,
    gather_across_devices: bool = False,
) -> Tensor:
    """CSR-style unsupervised in-batch contrastive loss (CSRv2/text/CSR_training.py)."""
    batch_size = latents_view_a.size(0)
    device = latents_view_a.device

    indexes = torch.arange(batch_size, device=device).repeat(2)
    z = torch.cat(
        [
            F.normalize(latents_view_a, p=2, dim=-1),
            F.normalize(latents_view_b, p=2, dim=-1),
        ],
        dim=0,
    )

    if gather_across_devices:
        gathered_z = torch.cat(all_gather(z), dim=0)
        gathered_indexes = torch.cat(all_gather(indexes), dim=0)
    else:
        gathered_z = z
        gathered_indexes = indexes

    sim = torch.exp(torch.einsum("if,jf->ij", z, gathered_z) / temperature)

    indexes_row = indexes.unsqueeze(0)
    gathered_indexes_col = gathered_indexes.unsqueeze(0)

    pos_mask = indexes_row.t() == gathered_indexes_col
    pos_mask[:, batch_size * get_rank() : batch_size * (get_rank() + 1)].fill_diagonal_(0)
    neg_mask = indexes_row.t() != gathered_indexes_col

    pos = torch.sum(sim * pos_mask, dim=1)
    neg = torch.sum(sim * neg_mask, dim=1)
    pos = pos.clamp(min=1e-8)
    neg = neg.clamp(min=1e-8)
    return -torch.mean(torch.log(pos / (pos + neg)))


def compute_sae_loss_components(
    sae_aux: dict[str, Tensor | None] | None,
    *,
    need_recon: bool = True,
    need_auxk: bool = True,
) -> dict[str, Tensor]:
    """Return unweighted SAE loss components; skip terms that are not needed."""
    device = (
        sae_aux["input"].device
        if sae_aux is not None
        else torch.device("cpu")
    )
    zero = torch.tensor(0.0, device=device)

    if sae_aux is None or (not need_recon and not need_auxk):
        return {
            "reconstruction": zero,
            "auxk": zero,
        }

    criterion = nn.MSELoss()
    x = sae_aux["input"]

    reconstruction = zero
    if need_recon:
        reconstruction = criterion(x, sae_aux["recons_k"])

    auxk = zero
    if need_auxk and sae_aux["recons_aux"] is not None:
        residual = x - sae_aux["recons_k"].detach() + sae_aux["pre_bias"].detach()
        auxk = normalized_mse(sae_aux["recons_aux"], residual, criterion).nan_to_num(0)

    return {
        "reconstruction": reconstruction,
        "auxk": auxk,
    }


def compute_sae_loss(
    sae_aux: dict[str, Tensor | None] | None,
    *,
    recon_coef: float = 1.0,
    auxk_coef: float = 0.1,
) -> Tensor:
    """Compute weighted SAE loss from module outputs."""
    parts = compute_sae_loss_components(
        sae_aux,
        need_recon=recon_coef > 0.0,
        need_auxk=auxk_coef > 0.0,
    )
    return recon_coef * parts["reconstruction"] + auxk_coef * parts["auxk"]


def _aggregate_sae_losses(
    outputs: list[dict[str, Tensor]],
    weights: LossWeights,
) -> dict[str, Tensor]:
    if not weights.any_sae_loss():
        return {}

    need_recon = weights.recon > 0.0
    need_auxk = weights.auxk > 0.0
    sae_parts = [
        compute_sae_loss_components(
            out.get("sae_aux"),
            need_recon=need_recon,
            need_auxk=need_auxk,
        )
        for out in outputs
    ]
    mean_recon = torch.stack([part["reconstruction"] for part in sae_parts]).mean()
    mean_auxk = torch.stack([part["auxk"] for part in sae_parts]).mean()

    losses: dict[str, Tensor] = {}
    if need_recon:
        losses["reconstruction"] = weights.recon * mean_recon
    if need_auxk:
        losses["auxk"] = weights.auxk * mean_auxk
    return losses


def _compute_maxsim_loss(
    model: ColBERT,
    *,
    sentence_features: list[dict[str, Tensor]],
    outputs: list[dict[str, Tensor]],
    score_metric: ColBERTScores,
    score_mini_batch_size: int | None,
    size_average: bool,
    gather_across_devices: bool,
    temperature: float,
) -> Tensor:
    embeddings = [
        F.normalize(out["token_embeddings"], p=2, dim=-1) for out in outputs
    ]

    skiplist = (
        model.skiplist if hasattr(model, "skiplist") else model.module.skiplist
    )
    do_query_expansion = (
        model.do_query_expansion
        if hasattr(model, "do_query_expansion")
        else model.module.do_query_expansion
    )
    masks = extract_skiplist_mask(
        sentence_features=sentence_features,
        skiplist=skiplist,
    )

    batch_size = embeddings[0].size(0)
    if gather_across_devices:
        embeddings = [
            embeddings[0],
            *[
                torch.cat(all_gather_with_gradients(embedding))
                for embedding in embeddings[1:]
            ],
        ]
        masks = [
            masks[0],
            *[torch.cat(all_gather(mask)) for mask in masks[1:]],
        ]

    n_docs = len(embeddings) - 1
    docs_stacked = torch.stack(embeddings[1:], dim=1)
    docs_mask_stacked = torch.stack(masks[1:], dim=1)
    q_mask = masks[0] if not do_query_expansion else None

    step = score_mini_batch_size or batch_size
    score_chunks = []
    for begin in range(0, batch_size, step):
        end = begin + step
        score_chunks.append(
            score_metric(
                embeddings[0][begin:end],
                docs_stacked,
                queries_mask=q_mask[begin:end] if q_mask is not None else None,
                documents_mask=docs_mask_stacked,
            )
        )
    scores = torch.cat(score_chunks, dim=0)

    contrastive_labels = torch.arange(batch_size, device=embeddings[0].device) * n_docs
    if gather_across_devices:
        contrastive_labels = contrastive_labels + get_rank() * batch_size * n_docs

    contrastive_loss = F.cross_entropy(
        input=scores / temperature,
        target=contrastive_labels,
        reduction="mean" if size_average else "sum",
    )
    if gather_across_devices:
        contrastive_loss *= get_world_size()
    return contrastive_loss


class CombinedSAELoss(nn.Module):
    """ColBERT + SAE training loss with four independently weighted terms."""

    def __init__(
        self,
        model: ColBERT,
        weights: LossWeights,
        *,
        ucl_temperature: float = 0.2,
        maxsim_temperature: float = 1.0,
        score_metric=None,
        score_mini_batch_size: int | None = None,
        size_average: bool = True,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        if not weights.any_active():
            raise ValueError("At least one loss coefficient must be > 0")

        self.model = model
        self.weights = weights
        self.ucl_temperature = ucl_temperature
        self.maxsim_temperature = maxsim_temperature
        self.score_metric = score_metric if score_metric is not None else ColBERTScores()
        self.score_mini_batch_size = score_mini_batch_size
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        sentence_features = list(sentence_features)
        losses: dict[str, Tensor] = {}

        if self.weights.ucl > 0.0:
            cl_losses: list[Tensor] = []
            all_outputs: list[dict[str, Tensor]] = []
            for sentence_feature in sentence_features:
                output_a = self.model(sentence_feature)
                output_b = self.model(sentence_feature)
                latents_a = pool_sparse_latents(
                    output_a["token_embeddings"],
                    sentence_feature["attention_mask"],
                )
                latents_b = pool_sparse_latents(
                    output_b["token_embeddings"],
                    sentence_feature["attention_mask"],
                )
                cl_losses.append(
                    unsupervised_inbatch_cl(
                        latents_a,
                        latents_b,
                        temperature=self.ucl_temperature,
                        gather_across_devices=self.gather_across_devices,
                    )
                )
                all_outputs.append(output_a)
            losses["unsupervised_cl"] = self.weights.ucl * torch.stack(cl_losses).mean()
        else:
            all_outputs = [self.model(sentence_feature) for sentence_feature in sentence_features]

        losses.update(_aggregate_sae_losses(all_outputs, self.weights))

        if self.weights.maxsim > 0.0:
            maxsim_loss = _compute_maxsim_loss(
                self.model,
                sentence_features=sentence_features,
                outputs=all_outputs,
                score_metric=self.score_metric,
                score_mini_batch_size=self.score_mini_batch_size,
                size_average=self.size_average,
                gather_across_devices=self.gather_across_devices,
                temperature=self.maxsim_temperature,
            )
            losses["maxsim"] = self.weights.maxsim * maxsim_loss

        return losses


class SAEWithUnsupervisedCL(CombinedSAELoss):
    """Backward-compatible wrapper around :class:`CombinedSAELoss`."""

    def __init__(
        self,
        model: ColBERT,
        *,
        cl_coef: float = 0.1,
        cl_temperature: float = 0.2,
        gather_across_devices: bool = False,
        sae_coef: float = 1.0,
        auxk_coef: float = 0.1,
        loss_4k_coef: float = 0.125,
    ) -> None:
        del loss_4k_coef  # 4x-K term merged into single reconstruction coefficient
        super().__init__(
            model,
            LossWeights(recon=sae_coef, auxk=auxk_coef, ucl=cl_coef, maxsim=0.0),
            ucl_temperature=cl_temperature,
            gather_across_devices=gather_across_devices,
        )


class ContrastiveWithSAE(CombinedSAELoss):
    """Backward-compatible wrapper around :class:`CombinedSAELoss`."""

    def __init__(
        self,
        model: ColBERT,
        *,
        score_metric=None,
        score_mini_batch_size: int | None = None,
        size_average: bool = True,
        gather_across_devices: bool = False,
        temperature: float = 1.0,
        sae_coef: float = 1.0,
        auxk_coef: float = 0.1,
        loss_4k_coef: float = 0.125,
    ) -> None:
        del loss_4k_coef
        super().__init__(
            model,
            LossWeights(recon=sae_coef, auxk=auxk_coef, ucl=0.0, maxsim=1.0),
            maxsim_temperature=temperature,
            score_metric=score_metric,
            score_mini_batch_size=score_mini_batch_size,
            size_average=size_average,
            gather_across_devices=gather_across_devices,
        )
