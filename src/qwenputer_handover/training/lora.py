"""src/qwenputer_handover/training/lora.py — LoRA SFT of base Qwen3.5 on selected rollouts.

Training runs on the stock checkpoint, never the merged one: the circuit's tokens are masked
out of the loss, and ``training/export.py`` folds the adapter back into a base-format checkpoint
that upstream's merge accepts. Adapters cover the language model's projections only (full
attention, Gated DeltaNet and MLP); the vision tower, embeddings and the tied head stay frozen.

Throughput: each optimizer step takes ``grad_accum`` examples of similar length (shuffled
pools sorted by length), run as right-padded micro-batches of at most ``batch_tokens`` padded
tokens. Right padding is exact for a causal model: no real position attends to, or carries
recurrent state from, a pad (Gated DeltaNet zeroes padded inputs). The head and cross-entropy
run only at labeled positions, in checkpointed chunks, so the 248K-vocabulary logits are never
held for a whole batch.
"""

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
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
POOL = 8  # optimizer steps' worth of shuffled examples sorted by length together
HEAD_CHUNK = 2048  # labeled positions per head + cross-entropy chunk (~2 GB of fp32 logits)


@dataclass(frozen=True)
class TrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 2
    grad_accum: int = 16  # examples per optimizer step
    max_len: int = 4096  # longer examples are dropped (and counted), never truncated
    batch_tokens: int = 16384  # padded tokens per forward pass
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


def windows(lengths: list[int], grad_accum: int, rng: random.Random) -> list[list[int]]:
    """One epoch's optimizer steps, as example indices: every example once, ``grad_accum`` per
    step, steps in random order. Examples are shuffled, then sorted by length within pools of
    ``POOL`` steps, so each step's examples have similar lengths and pad little."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    pool = POOL * grad_accum
    steps = []
    for p in range(0, len(order), pool):
        ranked = sorted(order[p : p + pool], key=lambda i: lengths[i])
        steps += [ranked[w : w + grad_accum] for w in range(0, len(ranked), grad_accum)]
    rng.shuffle(steps)
    return steps


def micro_batches(step: list[int], lengths: list[int], batch_tokens: int) -> list[list[int]]:
    """A step's examples as forward passes of at most ``batch_tokens`` padded tokens (rows times
    the longest row); an example longer than that runs alone."""
    batches: list[list[int]] = []
    for i in sorted(step, key=lambda i: -lengths[i]):
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= batch_tokens:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def train(
    model_dir: Path, rollouts: list[Rollout], variant: Variant, out_dir: Path, cfg: TrainConfig
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(model_dir)
    data, dropped = examples(tok, rollouts, variant, cfg.max_len)
    if not data:
        raise ValueError("no training examples fit max_len")
    lengths = [len(e.input_ids) for e in data]
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
                "labeled_tokens": sum(e.n_labeled for e in data),
                "steps": total,
            },
            indent=2,
        )
    )
    rng = random.Random(cfg.seed)
    lm = model.get_base_model()  # the adapted Qwen3_5ForConditionalGeneration
    step = 0
    model.train()
    with (out_dir / "train_log.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(cfg.epochs):
            for window in windows(lengths, cfg.grad_accum, rng):
                t0 = time.perf_counter()
                n_tok = sum(data[i].n_labeled for i in window)
                loss_sum = 0.0
                for mb in micro_batches(window, lengths, cfg.batch_tokens):
                    loss = loss_sum_of(lm, [data[i] for i in mb]) / n_tok
                    torch.autograd.backward(loss)
                    loss_sum += loss.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                seconds = time.perf_counter() - t0
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss_sum, 5),
                    "tokens": n_tok,
                    "grad_norm": round(grad_norm, 4),
                    "lr": sched.get_last_lr()[0],
                    "tok_per_s": round(sum(lengths[i] for i in window) / seconds),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), file=sys.stderr, flush=True)
                step += 1
    model.save_pretrained(str(out_dir))


def loss_sum_of(lm: Any, batch: list[Example], head_chunk: int = HEAD_CHUNK) -> torch.Tensor:
    """Summed next-token cross-entropy over the labeled tokens of ``batch``, run as one
    right-padded forward pass of ``lm`` (a Qwen3.5 ``ForConditionalGeneration``)."""
    dev = lm.lm_head.weight.device
    width = max(len(e.input_ids) for e in batch)
    ids = torch.zeros(len(batch), width, dtype=torch.long)  # pad id: never seen by a real token
    labels = torch.full((len(batch), width), IGNORE)
    mask = torch.zeros(len(batch), width, dtype=torch.long)
    for row, e in enumerate(batch):
        ids[row, : len(e.input_ids)] = torch.tensor(e.input_ids)
        labels[row, : len(e.labels)] = torch.tensor(e.labels)
        mask[row, : len(e.input_ids)] = 1
    hidden = lm.model(
        input_ids=ids.to(dev),
        attention_mask=mask.to(dev),
        position_ids=torch.arange(width, device=dev).expand(len(batch), width),
        use_cache=False,
    ).last_hidden_state
    target = labels[:, 1:].to(dev)  # position t predicts token t + 1
    keep = target != IGNORE
    return head_ce(lm.lm_head.weight, hidden[:, :-1][keep], target[keep], head_chunk)


def head_ce(
    weight: torch.Tensor, hidden: torch.Tensor, target: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Summed cross-entropy of ``hidden @ weight.T`` against ``target``, ``chunk`` rows at a
    time; each chunk's logits are recomputed in the backward pass instead of stored."""
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, len(target), chunk):
        part = checkpoint(
            _ce_sum, weight, hidden[i : i + chunk], target[i : i + chunk], use_reentrant=False
        )
        total = total + part
    return total


def _ce_sum(weight: torch.Tensor, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(hidden, weight).float(), target, reduction="sum")
