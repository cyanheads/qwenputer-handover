"""src/qwenputer_handover/training/lora.py — LoRA SFT of base Qwen3.5 on selected rollouts.

Training runs on the stock checkpoint, never the merged one: the circuit's tokens are masked
out of the loss, and ``training/export.py`` folds the adapter back into a base-format checkpoint
that upstream's merge accepts. Adapters cover the language model's projections only (full
attention, Gated DeltaNet and MLP); the vision tower, embeddings and the tied head stay frozen.
"""

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForConditionalGeneration

from qwenputer_handover.rollout.prompts import prompt_ids
from qwenputer_handover.rollout.records import Rollout
from qwenputer_handover.rollout.variants import Variant
from qwenputer_handover.runtime import device
from qwenputer_handover.training.sft_data import IGNORE, Example, build_example

TARGETS = (
    r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)"
)


@dataclass(frozen=True)
class TrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 2
    grad_accum: int = 16  # examples per optimizer step
    max_len: int = 4096  # longer examples are dropped (and counted), never truncated
    warmup: float = 0.05  # share of steps with linear warmup, then cosine decay to 0
    seed: int = 0


def examples(
    tok: PreTrainedTokenizerBase, rollouts: list[Rollout], variant: Variant, max_len: int
) -> tuple[list[Example], int]:
    """Each rollout's generation under a ``variant`` prompt; also the count dropped as too long."""
    built = [
        build_example(prompt_ids(tok, r.question, variant), r.gen_ids, r.forced) for r in rollouts
    ]
    kept = [e for e in built if len(e.input_ids) <= max_len]
    return kept, len(built) - len(kept)


def lr_at(step: int, total: int, warmup: float) -> float:
    """Multiplier on the base learning rate at optimizer step ``step`` (0-based)."""
    w = max(1, round(total * warmup))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def train(
    model_dir: Path, rollouts: list[Rollout], variant: Variant, out_dir: Path, cfg: TrainConfig
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(model_dir)
    data, dropped = examples(tok, rollouts, variant, cfg.max_len)
    if not data:
        raise ValueError("no training examples fit max_len")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device()
    )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=TARGETS,
            bias="none",
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    total = math.ceil(len(data) / cfg.grad_accum) * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, cfg.warmup))
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "model": str(model_dir),
                "variant": variant,
                "examples": len(data),
                "dropped_too_long": dropped,
                "steps": total,
            },
            indent=2,
        )
    )
    rng = random.Random(cfg.seed)
    dev = next(model.parameters()).device
    step = 0
    model.train()
    with (out_dir / "train_log.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(cfg.epochs):
            order = list(range(len(data)))
            rng.shuffle(order)
            for w in range(0, len(order), cfg.grad_accum):
                window = [data[i] for i in order[w : w + cfg.grad_accum]]
                n_tok = sum(e.n_labeled for e in window)
                loss_sum = 0.0
                for e in window:
                    loss = _loss_sum(model, e, dev) / n_tok
                    torch.autograd.backward(loss)
                    loss_sum += loss.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss_sum, 5),
                    "tokens": n_tok,
                    "grad_norm": round(grad_norm, 4),
                    "lr": sched.get_last_lr()[0],
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), file=sys.stderr, flush=True)
                step += 1
    model.save_pretrained(str(out_dir))


def _loss_sum(model: torch.nn.Module, e: Example, dev: torch.device) -> torch.Tensor:
    """Summed next-token cross-entropy over the labeled tokens; logits are computed only at the
    positions that predict one (Qwen's 248K vocabulary makes full-sequence logits the largest
    activation by far)."""
    labels = torch.tensor(e.labels)
    targets_at = (labels[1:] != IGNORE).nonzero().squeeze(1)  # position t predicts token t + 1
    logits = model(
        input_ids=torch.tensor([e.input_ids], device=dev),
        use_cache=False,
        logits_to_keep=targets_at.to(dev),
    ).logits[0]
    return F.cross_entropy(logits.float(), labels[targets_at + 1].to(dev), reduction="sum")
