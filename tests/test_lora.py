import random
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from qwenputer_handover.training.lora import loss_sum_of, lr_at, micro_batches, windows
from qwenputer_handover.training.sft_data import IGNORE, Example, build_example


def test_warmup_then_cosine_to_zero() -> None:
    total = 100
    assert lr_at(0, total, 0.1) == pytest.approx(0.1)
    assert lr_at(9, total, 0.1) == pytest.approx(1.0)
    assert lr_at(10, total, 0.1) == pytest.approx(1.0)
    assert lr_at(55, total, 0.1) == pytest.approx(0.5)
    assert lr_at(100, total, 0.1) == pytest.approx(0.0)


def test_windows_take_every_example_once_in_similar_length_steps() -> None:
    rng = random.Random(0)
    lengths = [rng.randint(100, 4000) for _ in range(1000)]
    steps = windows(lengths, 16, random.Random(1))
    assert sorted(i for s in steps for i in s) == list(range(1000))
    assert len(steps) == 63 and all(len(s) <= 16 for s in steps)  # ceil(1000 / 16)
    spread = sum(max(lengths[i] for i in s) - min(lengths[i] for i in s) for s in steps)
    assert spread / len(steps) < 600  # a random step of 16 spans about 3,300
    assert windows(lengths, 16, random.Random(1)) == steps


def test_micro_batches_stay_under_the_padded_token_budget() -> None:
    lengths = [900, 1000, 300, 5000, 250, 400]
    batches = micro_batches([0, 1, 2, 3, 4, 5], lengths, 2048)
    assert batches == [[3], [1, 0], [5, 2, 4]]  # 5000 alone; 2 x 1000; 3 x 400
    assert sorted(i for b in batches for i in b) == list(range(6))


def tiny_model() -> Qwen3_5ForConditionalGeneration:
    """A random 4-layer Qwen3.5 with the 4B's layer pattern (three Gated DeltaNet layers, then
    full attention), tied embeddings and a tiny vocabulary, in float32 on the CPU."""
    torch.manual_seed(0)
    text = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
    model = construct(
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True),
    )
    assert model.dtype == torch.float32
    return model


def construct(
    cls: Callable[[Qwen3_5Config], Qwen3_5ForConditionalGeneration], cfg: Qwen3_5Config
) -> Qwen3_5ForConditionalGeneration:
    """``cls(cfg)``, with the constructor's type stated here (its ``__init__`` has none)."""
    return cls(cfg)


def reference_loss(model: Qwen3_5ForConditionalGeneration, e: Example) -> torch.Tensor:
    """The plain way: one unpadded sequence, full logits, shifted labels."""
    logits = model(input_ids=torch.tensor([e.input_ids])).logits[0, :-1]
    labels = torch.tensor(e.labels[1:])
    return F.cross_entropy(logits, labels, ignore_index=IGNORE, reduction="sum")


BATCH = [  # three lengths, so two rows are right-padded; forced spans masked
    build_example([1, 2, 3], [10, 11, 12, 13, 14, 15, 16], [(2, 4)]),
    build_example([4, 5], [20, 21, 22], []),
    build_example([6, 7, 8, 9], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], [(0, 3), (9, 11)]),
]


def test_padded_batch_loss_equals_the_per_sequence_loss() -> None:
    model = tiny_model()
    with torch.no_grad():
        expected = sum(float(reference_loss(model, e)) for e in BATCH)
        assert float(loss_sum_of(model, BATCH, head_chunk=3)) == pytest.approx(expected, rel=1e-5)


def test_padded_batch_gradients_equal_the_per_sequence_gradients() -> None:
    model = tiny_model()
    torch.autograd.backward(loss_sum_of(model, BATCH, head_chunk=3))
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    torch.autograd.backward(torch.stack([reference_loss(model, e) for e in BATCH]).sum())
    want = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    assert any("linear_attn" in n for n in got)
    for n, g in got.items():
        torch.testing.assert_close(g, want[n], rtol=1e-4, atol=1e-6, msg=n)
