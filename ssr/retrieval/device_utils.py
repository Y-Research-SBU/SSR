"""Device resolution for MTEB SSR evaluation."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_encode_device(requested: str | None) -> str:
    """Device for model loading and encoding (ColBERT forward pass)."""
    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for encoding but torch.cuda.is_available() is False.")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_score_device(requested: str | None, *, encode_device: str) -> str:
    """Device for exact MaxSim scoring via torch.sparse (exact+cuda path only)."""
    if requested == "auto" or requested is None:
        if encode_device.startswith("cuda") and torch.cuda.is_available():
            return encode_device.split(":")[0] if ":" in encode_device else "cuda"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("score-device=cuda unavailable; falling back to cpu.")
        return "cpu"
    return requested


def describe_hardware(*, encode_device: str, score_device: str, retrieval_mode: str) -> str:
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "N/A"
    return (
        f"mode={retrieval_mode} | encode={encode_device} | score={score_device} | "
        f"cuda_available={cuda} | gpu_name={name}"
    )
