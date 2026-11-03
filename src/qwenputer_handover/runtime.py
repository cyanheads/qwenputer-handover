"""src/qwenputer_handover/runtime.py — where models come from and which device they run on."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def device() -> str:
    """The CUDA device. Sampling and training run on a CUDA GPU, never on a CPU or MPS, so there
    is no fallback."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: scan and train need a CUDA GPU")
    return "cuda"


def resolve_model(name: str) -> Path:
    """A local checkpoint directory, or a Hugging Face repo id fetched into the HF cache."""
    path = Path(name)
    return path if path.is_dir() else Path(snapshot_download(name))
