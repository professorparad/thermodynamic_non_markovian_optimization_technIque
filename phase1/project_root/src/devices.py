"""Torch device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a user-facing device string to a torch device."""
    requested = (device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type
