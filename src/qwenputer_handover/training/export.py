"""src/qwenputer_handover/training/export.py — fold a LoRA adapter into a copy of the base
checkpoint, laid out exactly like the base.

Upstream's merge (``python -m chipformer.rv.merge SRC``) reads SRC as a stock Qwen3.5
checkpoint: ``config.json``, ``tokenizer.json`` and safetensors shards keyed
``model.language_model.*``. So the export copies every base file unchanged and rewrites only
the shards holding an adapted weight, each adapted weight replaced by
``W + scale * B @ A`` (computed in float32, stored in W's dtype). Keys, shard names, the index
and the metadata stay the base's.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ADAPTER_PREFIX = "base_model.model."
INDEX = "model.safetensors.index.json"


class ExportError(Exception):
    """The adapter cannot be folded into this base checkpoint as given."""


@dataclass(frozen=True)
class Delta:
    a: torch.Tensor  # (r, in)
    b: torch.Tensor  # (out, r)
    scale: float


def read_adapter(adapter_dir: Path) -> dict[str, Delta]:
    """Each adapted weight's checkpoint key and its low-rank update, from a PEFT LoRA adapter
    (``adapter_config.json`` + ``adapter_model.safetensors``)."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    unsupported = [k for k in ("use_dora", "fan_in_fan_out") if cfg.get(k)]
    unsupported += [k for k in ("rank_pattern", "alpha_pattern") if cfg.get(k)]
    if cfg.get("peft_type") != "LORA" or unsupported:
        raise ExportError(
            f"only plain LoRA folds here (peft_type={cfg.get('peft_type')}, "
            f"unsupported: {unsupported})"
        )
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r**0.5 if cfg.get("use_rslora") else alpha / r
    tensors = load_file(adapter_dir / "adapter_model.safetensors")
    modules = {k.removeprefix(ADAPTER_PREFIX).rsplit(".lora_", 1)[0] for k in tensors}
    deltas = {}
    for m in sorted(modules):
        a, b = f"{ADAPTER_PREFIX}{m}.lora_A.weight", f"{ADAPTER_PREFIX}{m}.lora_B.weight"
        if a not in tensors or b not in tensors:
            raise ExportError(f"{m}: the adapter holds tensors other than a lora_A/lora_B pair")
        deltas[f"{m}.weight"] = Delta(tensors[a], tensors[b], scale)
    return deltas


def shard_map(base_dir: Path) -> dict[str, str]:
    """Checkpoint key -> shard file name."""
    index = base_dir / INDEX
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        return weight_map
    single = base_dir / "model.safetensors"
    with safe_open(single, "pt") as f:
        return dict.fromkeys(f.keys(), single.name)


def export(base_dir: Path, adapter_dir: Path, out_dir: Path) -> list[str]:
    """Write the merged checkpoint to ``out_dir`` (which must not exist); returns the keys it
    changed."""
    if out_dir.exists():
        raise ExportError(f"{out_dir} exists: export writes a fresh directory")
    deltas = read_adapter(adapter_dir)
    shards = shard_map(base_dir)
    missing = sorted(k for k in deltas if k not in shards)
    if missing:
        raise ExportError(
            f"{len(missing)} adapted weights are not in the base checkpoint, e.g. {missing[0]}"
        )
    patched = {shards[k] for k in deltas}
    out_dir.mkdir(parents=True)
    for src in sorted(base_dir.iterdir()):
        if src.is_file() and src.name not in patched:
            shutil.copy2(src, out_dir / src.name)  # follows the HF cache's blob symlinks
    for shard in sorted(patched):
        with safe_open(base_dir / shard, "pt") as f:
            metadata = f.metadata()
        tensors = load_file(base_dir / shard)
        for key, d in deltas.items():
            if shards[key] == shard:
                w = tensors[key]
                if w.shape != (d.b.shape[0], d.a.shape[1]):
                    raise ExportError(
                        f"{key}: base {tuple(w.shape)} vs LoRA {(d.b.shape[0], d.a.shape[1])}"
                    )
                tensors[key] = (w.float() + d.scale * (d.b.float() @ d.a.float())).to(w.dtype)
        save_file(tensors, out_dir / shard, metadata=metadata)
    return sorted(deltas)
