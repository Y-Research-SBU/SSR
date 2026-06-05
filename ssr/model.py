"""SSR: ColBERT backbone with Sparse Autoencoder replacing the Dense projector."""

from __future__ import annotations

import logging
import string
from typing import Iterable

from pylate.models.colbert import ColBERT
from pylate.models.Dense import Dense
from sentence_transformers.models import Transformer
from torch import nn

from .sparse_autoencoder import SparseAutoencoder

logger = logging.getLogger(__name__)

__all__ = ["SSR", "build_ssr"]


class SSR(ColBERT):
    """Sparse Semantic Retrieval model (ColBERT backbone + SAE sparse projector)."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Iterable[nn.Module] | None = None,
        sae_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        sae_kwargs = sae_kwargs or {}
        embedding_size = kwargs.pop("embedding_size", None)

        if modules is None and model_name_or_path is not None:
            transformer = Transformer(model_name_or_path)
            hidden_size = transformer.get_word_embedding_dimension()
            sae = SparseAutoencoder(in_features=hidden_size, **sae_kwargs)
            modules = [transformer, sae]

        if modules is not None:
            last = list(modules)[-1]
            if not isinstance(last, SparseAutoencoder):
                raise TypeError(
                    "SSR requires SparseAutoencoder as the last module."
                )

        # Prevent ColBERT from appending an extra Dense layer.
        kwargs["embedding_size"] = None

        # Call SentenceTransformer.__init__ via ColBERT's parent chain, but skip
        # ColBERT's Dense auto-append / conversion logic.
        self.query_prefix = kwargs.get("query_prefix")
        self.document_prefix = kwargs.get("document_prefix")
        self.query_length = kwargs.get("query_length")
        self.document_length = kwargs.get("document_length")
        self.do_query_expansion = kwargs.get("do_query_expansion")
        self.attend_to_expansion_tokens = kwargs.get("attend_to_expansion_tokens")
        self.skiplist_words = kwargs.get("skiplist_words")

        from pylate.models.colbert import PylateModelCardData
        from sentence_transformers.similarity_functions import SimilarityFunction

        model_card_data = kwargs.get("model_card_data") or PylateModelCardData()
        if kwargs.get("similarity_fn_name") is None:
            kwargs["similarity_fn_name"] = SimilarityFunction.COSINE

        # SentenceTransformer init (skip ColBERT.__init__ Dense handling).
        # When custom modules are supplied, do not pass model_name_or_path or ST
        # will reload a checkpoint and discard the SAE module.
        st_model_name = None if modules is not None else model_name_or_path
        super(ColBERT, self).__init__(
            model_name_or_path=st_model_name,
            modules=modules,
            model_card_data=model_card_data,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in {
                    "query_prefix",
                    "document_prefix",
                    "query_length",
                    "document_length",
                    "do_query_expansion",
                    "attend_to_expansion_tokens",
                    "skiplist_words",
                    "model_card_data",
                    "similarity_fn_name",
                    "embedding_size",
                }
            },
        )

        # Convert any intermediate ST Dense layers, but keep SparseAutoencoder intact.
        for i in range(1, len(self)):
            if isinstance(self[i], SparseAutoencoder):
                continue
            if not isinstance(self[i], Dense):
                self[i] = Dense.from_sentence_transformers(dense=self[i])

        if embedding_size is not None:
            logger.warning(
                "embedding_size=%s is ignored for SSR; "
                "output dimension is n_latents=%s.",
                embedding_size,
                self.sae_module.n_latents,
            )

        try:
            dtype = next(self.parameters()).dtype
            self.to(dtype)
        except StopIteration:
            pass

        self._finalize_colbert_config(
            query_prefix=kwargs.get("query_prefix"),
            document_prefix=kwargs.get("document_prefix"),
            query_length=kwargs.get("query_length"),
            document_length=kwargs.get("document_length"),
            do_query_expansion=kwargs.get("do_query_expansion"),
            attend_to_expansion_tokens=kwargs.get("attend_to_expansion_tokens"),
            skiplist_words=kwargs.get("skiplist_words"),
            device=kwargs.get("device"),
        )

    @property
    def sae_module(self) -> SparseAutoencoder:
        module = self[-1]
        if not isinstance(module, SparseAutoencoder):
            raise TypeError("Expected SparseAutoencoder as the last module.")
        return module

    def _finalize_colbert_config(self, **kwargs) -> None:
        """Apply ColBERT post-init settings (prefixes, lengths, skiplist, device)."""
        device = kwargs.get("device")
        self.to(device)
        self.is_hpu_graph_enabled = False

        self.query_prefix = (
            kwargs.get("query_prefix")
            if kwargs.get("query_prefix") is not None
            else self.query_prefix
            if getattr(self, "query_prefix", None) is not None
            else "[Q] "
        )
        self.document_prefix = (
            kwargs.get("document_prefix")
            if kwargs.get("document_prefix") is not None
            else self.document_prefix
            if getattr(self, "document_prefix", None) is not None
            else "[D] "
        )

        prefix_tokens = [
            p for p in (self.query_prefix, self.document_prefix) if p != ""
        ]
        if prefix_tokens:
            try:
                self._first_module().auto_model.resize_token_embeddings(
                    len(self.tokenizer)
                )
                self.tokenizer.add_tokens(prefix_tokens)
                self._first_module().auto_model.resize_token_embeddings(
                    len(self.tokenizer)
                )
            except NotImplementedError:
                logger.warning(
                    "Tokenizer does not support resizing; query/doc prefixes were not added."
                )

        self.document_prefix_id = (
            self.tokenizer.convert_tokens_to_ids(self.document_prefix)
            if self.document_prefix
            else None
        )
        self.query_prefix_id = (
            self.tokenizer.convert_tokens_to_ids(self.query_prefix)
            if self.query_prefix
            else None
        )

        if self.tokenizer.mask_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.mask_token_id
        elif self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif self.tokenizer.pad_token_id is None:
            raise NotImplementedError(
                "No suitable padding token found for query expansion."
            )

        self.document_length = (
            kwargs.get("document_length")
            if kwargs.get("document_length") is not None
            else getattr(self, "document_length", None)
            if getattr(self, "document_length", None) is not None
            else 180
        )
        self.query_length = (
            kwargs.get("query_length")
            if kwargs.get("query_length") is not None
            else getattr(self, "query_length", None)
            if getattr(self, "query_length", None) is not None
            else 32
        )

        self.skiplist_words = (
            kwargs.get("skiplist_words")
            if kwargs.get("skiplist_words") is not None
            else getattr(self, "skiplist_words", None)
            if getattr(self, "skiplist_words", None) is not None
            else list(string.punctuation)
        )
        self.skiplist = [
            self.tokenizer.convert_tokens_to_ids(word)
            for word in self.skiplist_words
        ]

        self.do_query_expansion = (
            kwargs.get("do_query_expansion")
            if kwargs.get("do_query_expansion") is not None
            else getattr(self, "do_query_expansion", None)
            if getattr(self, "do_query_expansion", None) is not None
            else True
        )
        self.attend_to_expansion_tokens = (
            kwargs.get("attend_to_expansion_tokens")
            if kwargs.get("attend_to_expansion_tokens") is not None
            else getattr(self, "attend_to_expansion_tokens", None)
            if getattr(self, "attend_to_expansion_tokens", None) is not None
            else False
        )
        if not self.do_query_expansion:
            self.attend_to_expansion_tokens = False


def build_ssr(
    model_name_or_path: str,
    *,
    n_latents: int | None = None,
    topk: int = 32,
    initial_topk: int | None = None,
    auxk: int = 512,
    normalize: bool = False,
    dead_threshold: int = 30,
    **colbert_kwargs,
) -> SSR:
    """Build an SSR model (ColBERT backbone + Sparse Autoencoder projector)."""
    return SSR(
        model_name_or_path=model_name_or_path,
        sae_kwargs={
            "n_latents": n_latents,
            "topk": topk if initial_topk is None else initial_topk,
            "auxk": auxk,
            "normalize": normalize,
            "dead_threshold": dead_threshold,
        },
        **colbert_kwargs,
    )
