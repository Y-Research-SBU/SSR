"""CUDA device selection helpers (must run before torch initializes CUDA)."""

from __future__ import annotations

import os
from argparse import Namespace

__all__ = ["configure_cuda_env", "resolve_training_device", "get_device_info"]


def configure_cuda_env(args: Namespace) -> None:
    """Set CUDA_VISIBLE_DEVICES from --gpu / --device before importing torch."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        return

    if args.device and args.device.startswith("cuda:"):
        gpu_id = args.device.split(":", 1)[1]
        if gpu_id.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def resolve_training_device(args: Namespace) -> str:
    """Return the device string passed to the ColBERT model."""
    import torch

    if args.device == "cpu":
        return "cpu"
    if args.gpu is not None or (args.device and args.device.startswith("cuda")):
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "CUDA was requested (--gpu / --device cuda) but torch.cuda.is_available() is False. "
            "Check driver/CUDA 12.8 PyTorch install (./install_env_cu128.sh)."
        )
    if args.device:
        return args.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info() -> dict[str, str | int | None]:
    import torch

    info: dict[str, str | int | None] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count_visible"] = torch.cuda.device_count()
    return info
