"""Sparse Autoencoder module adapted from CSRv2 for PyLate ColBERT training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers.models.Module import Module
from torch import Tensor

__all__ = ["SparseAutoencoder", "TiedTranspose", "get_current_k"]


class TiedTranspose(nn.Module):
    """Decoder that shares weights with the encoder (transposed)."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)


class SparseAutoencoderCore(nn.Module):
    """CSR-style tied-weight sparse autoencoder (CSRv2/text/model_zoo.py)."""

    def __init__(
        self,
        n_inputs: int,
        n_latents: int,
        topk: int,
        auxk: int,
        normalize: bool = False,
        dead_threshold: int = 30,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.topk = topk
        self.auxk = auxk
        self.normalize = normalize
        self.dead_threshold = dead_threshold

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.decoder = TiedTranspose(self.encoder)

        self.register_buffer(
            "stats_last_nonzero",
            torch.zeros(n_latents, dtype=torch.long),
        )

    @property
    def in_features(self) -> int:
        return self.n_inputs

    @property
    def out_features(self) -> int:
        return self.n_latents

    def _auxk_mask(self, x: torch.Tensor) -> torch.Tensor:
        dead_mask = self.stats_last_nonzero > self.dead_threshold
        x = x.clone()
        x *= dead_mask
        return x

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return F.linear(x, self.encoder.weight, self.latent_bias)

    @staticmethod
    def _layer_norm(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        return x / (std + eps), mu, std

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.normalize:
            return x, {}
        x, mu, std = self._layer_norm(x)
        return x, {"mu": mu, "std": std}

    def top_k(
        self,
        x: torch.Tensor,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if topk is None:
            topk = self.topk

        values, indices = torch.topk(x, k=topk, dim=-1)
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, indices, values)
        latents_k = F.relu(z_topk)

        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            indices.reshape(-1),
            (values > 1e-5).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1

        latents_auxk = None
        if self.auxk:
            masked = self._auxk_mask(x)
            aux_values, aux_indices = torch.topk(masked, k=self.auxk, dim=-1)
            z_auxk = torch.zeros_like(x)
            z_auxk.scatter_(-1, aux_indices, aux_values)
            latents_auxk = F.relu(z_auxk)

        return latents_k, latents_auxk

    def decode(self, latents: torch.Tensor, info: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(
        self,
        x: torch.Tensor,
        topk: int | None = None,
        topk_4x: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        x_proc, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x_proc)

        effective_topk = topk if topk is not None else self.topk
        latents_k, latents_auxk = self.top_k(latents_pre_act, topk=effective_topk)
        recons_k = self.decode(latents_k, info)

        recons_4k = None
        if topk_4x is not None and topk_4x > effective_topk:
            latents_4k, _ = self.top_k(latents_pre_act, topk=topk_4x)
            recons_4k = self.decode(latents_4k, info)

        recons_aux = None
        if latents_auxk is not None:
            recons_aux = self.decode(latents_auxk, info)

        return {
            "input": x,
            "latents_pre_act": latents_pre_act,
            "latents_k": latents_k,
            "recons_k": recons_k,
            "recons_4k": recons_4k,
            "recons_aux": recons_aux,
        }


class SparseAutoencoder(Module):
    """SentenceTransformer module that replaces ColBERT Dense with a sparse autoencoder."""

    config_keys: list[str] = [
        "in_features",
        "n_latents",
        "topk",
        "auxk",
        "normalize",
        "dead_threshold",
        "token_scope",
        "cls_token_id",
    ]

    def __init__(
        self,
        in_features: int,
        n_latents: int | None = None,
        topk: int = 32,
        auxk: int = 512,
        normalize: bool = False,
        dead_threshold: int = 30,
        token_scope: str = "all",
        cls_token_id: int | None = None,
    ) -> None:
        super().__init__()
        if token_scope not in {"all", "non-cls", "cls"}:
            raise ValueError("token_scope must be one of: all, non-cls, cls")
        self.in_features = in_features
        self.n_latents = n_latents or in_features * 4
        self.topk = topk
        self.auxk = auxk
        self.normalize = normalize
        self.dead_threshold = dead_threshold
        self.token_scope = token_scope
        self.cls_token_id = cls_token_id

        self.sae = SparseAutoencoderCore(
            n_inputs=in_features,
            n_latents=self.n_latents,
            topk=topk,
            auxk=auxk,
            normalize=normalize,
            dead_threshold=dead_threshold,
        )

    @property
    def out_features(self) -> int:
        return self.n_latents

    def set_topk(self, topk: int) -> None:
        self.topk = topk
        self.sae.topk = topk

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        token_embeddings = features["token_embeddings"]
        attention_mask = features.get("attention_mask")
        input_ids = features.get("input_ids")

        batch_size, seq_len, hidden = token_embeddings.shape
        flat = token_embeddings.reshape(batch_size * seq_len, hidden)

        if attention_mask is not None:
            flat_mask = attention_mask.reshape(batch_size * seq_len).bool()
            active = flat[flat_mask]
        else:
            flat_mask = None
            active = flat

        if self.token_scope != "all":
            if input_ids is not None and self.cls_token_id is not None:
                cls_mask = input_ids.eq(int(self.cls_token_id))
            else:
                cls_mask = torch.zeros(
                    batch_size,
                    seq_len,
                    dtype=torch.bool,
                    device=token_embeddings.device,
                )
                cls_mask[:, 0] = True
            if attention_mask is not None:
                cls_mask = cls_mask & attention_mask.bool()
            scope_mask = cls_mask if self.token_scope == "cls" else ~cls_mask
            if attention_mask is not None:
                scope_mask = scope_mask & attention_mask.bool()
            flat_mask = scope_mask.reshape(batch_size * seq_len)
            active = flat[flat_mask]

        if active.numel() == 0:
            features["token_embeddings"] = token_embeddings.new_zeros(
                batch_size, seq_len, self.n_latents
            )
            features["sae_aux"] = None
            return features

        sae_out = self.sae(active, topk=self.topk, topk_4x=4 * self.topk)

        latents_k = sae_out["latents_k"].to(dtype=flat.dtype)
        sparse_flat = flat.new_zeros(batch_size * seq_len, self.n_latents)
        if flat_mask is not None:
            sparse_flat[flat_mask] = latents_k
        else:
            sparse_flat = latents_k

        features["token_embeddings"] = sparse_flat.reshape(batch_size, seq_len, self.n_latents)
        features["sae_aux"] = {
            "input": sae_out["input"],
            "recons_k": sae_out["recons_k"],
            "recons_4k": sae_out["recons_4k"],
            "recons_aux": sae_out["recons_aux"],
            "pre_bias": self.sae.pre_bias,
        }
        return features

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "n_latents": self.n_latents,
            "topk": self.topk,
            "auxk": self.auxk,
            "normalize": self.normalize,
            "dead_threshold": self.dead_threshold,
            "token_scope": self.token_scope,
            "cls_token_id": self.cls_token_id,
        }

    def save(self, output_path: str, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        return cls.load_torch_weights(
            model_name_or_path=model_name_or_path,
            model=model,
            **hub_kwargs,
        )

    def __repr__(self) -> str:
        return f"SparseAutoencoder({self.get_config_dict()})"


def get_current_k(
    global_step: int,
    initial_k: int,
    final_k: int,
    total_steps: int,
    k_decay_ratio: float = 0.7,
) -> int:
    """Linear K-annealing schedule from CSRv2 training."""
    k_decay_steps = int(total_steps * k_decay_ratio)
    if global_step >= k_decay_steps:
        return final_k
    progress = global_step / k_decay_steps
    return round(initial_k * (1 - progress) + final_k * progress)


def normalized_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """AuxK normalized MSE from CSRv2/text/CSR_training.py."""
    target_mu = target.mean(dim=0)
    return criterion(recon, target) / criterion(
        target_mu[None, :].expand_as(target),
        target,
    )
