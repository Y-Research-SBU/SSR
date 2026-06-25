"""CLS-token sparse encoder used to augment token-level SSR retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from ssr.sparse_autoencoder import SparseAutoencoder

from .sparse_repr import (
    SparseTokenEmbeddings,
    append_cls_sparse_rows,
    batch_dense_to_sparse,
)


class CLSSparseEncoder:
    """Encode the backbone [CLS] embedding with a separate SAE checkpoint."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        *,
        device: str,
        topk: int | None = None,
    ) -> None:
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device
        self.topk = int(topk if topk is not None else sae.topk)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        device: str,
        topk: int | None = None,
    ) -> "CLSSparseEncoder":
        return cls(_load_sparse_autoencoder(path), device=device, topk=topk)

    @property
    def n_latents(self) -> int:
        return int(self.sae.n_latents)

    def encode(
        self,
        model,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> list[SparseTokenEmbeddings]:
        """Return one sparse CLS row per input text."""
        if not texts:
            return []
        features = _tokenize_for_colbert(model, list(texts), is_query=is_query)
        features = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }
        with torch.inference_mode():
            transformer_out = model._first_module()(features)
            token_embeddings = transformer_out["token_embeddings"]
            positions = _cls_positions(model, features, token_embeddings)
            batch_idx = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
            cls_dense = token_embeddings[batch_idx, positions]
            sae_out = self.sae.sae(cls_dense, topk=self.topk, topk_4x=None)
            cls_latents = sae_out["latents_k"].to(dtype=token_embeddings.dtype)
        return batch_dense_to_sparse(
            [row.unsqueeze(0) for row in cls_latents],
            n_latents=self.n_latents,
            topk=self.topk,
        )

    def encode_with_token_sae(
        self,
        model,
        texts: Sequence[str],
        *,
        is_query: bool,
        token_n_latents: int,
        token_topk: int,
    ) -> list[SparseTokenEmbeddings]:
        """Encode token SAE rows and CLS SAE rows in a single transformer pass."""
        if not texts:
            return []
        features = _tokenize_for_colbert(model, list(texts), is_query=is_query)
        features = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }
        with torch.inference_mode():
            transformer_out = model._first_module()(features)
            token_embeddings = transformer_out["token_embeddings"]
            positions = _cls_positions(model, features, token_embeddings)
            batch_idx = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
            cls_dense = token_embeddings[batch_idx, positions]

            token_out = model.sae_module(transformer_out)
            token_sparse = batch_dense_to_sparse(
                [row for row in token_out["token_embeddings"]],
                n_latents=token_n_latents,
                topk=token_topk,
            )

            cls_out = self.sae.sae(cls_dense, topk=self.topk, topk_4x=None)
            cls_latents = cls_out["latents_k"].to(dtype=token_embeddings.dtype)
            cls_sparse = batch_dense_to_sparse(
                [row.unsqueeze(0) for row in cls_latents],
                n_latents=self.n_latents,
                topk=self.topk,
            )
        return append_cls_sparse_rows(token_sparse, cls_sparse)


def _tokenize_for_colbert(model, texts: list[str], *, is_query: bool) -> dict:
    try:
        return model.tokenize(texts, is_query=is_query)
    except TypeError:
        return model.tokenize(texts)


def _load_sparse_autoencoder(path: str | Path) -> SparseAutoencoder:
    path = Path(path)
    try:
        return SparseAutoencoder.load(str(path))
    except Exception as exc:
        for child in sorted(path.iterdir()) if path.is_dir() else []:
            if _looks_like_sae_dir(child):
                return SparseAutoencoder.load(str(child))
        raise ValueError(
            f"Could not load CLS SAE from {path}. Pass the SparseAutoencoder module "
            "directory, or a checkpoint directory containing one."
        ) from exc


def _looks_like_sae_dir(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        return False
    return {"in_features", "n_latents", "topk"}.issubset(config)


def _cls_positions(model, features: dict, token_embeddings: torch.Tensor) -> torch.Tensor:
    input_ids = features.get("input_ids")
    cls_token_id = getattr(getattr(model, "tokenizer", None), "cls_token_id", None)
    if input_ids is not None and cls_token_id is not None:
        matches = input_ids.eq(int(cls_token_id))
        has_match = matches.any(dim=1)
        positions = matches.float().argmax(dim=1).long()
        return torch.where(has_match, positions, torch.zeros_like(positions))
    return torch.zeros(token_embeddings.size(0), dtype=torch.long, device=token_embeddings.device)
