"""tests/helpers.py — rollout fixtures."""

from dataclasses import replace
from typing import Any

from qwenputer_handover.rollout.records import Call, Rollout

BASE = Rollout(
    problem_id="synth-mul-0-0",
    dataset="synth",
    split="train",
    question="What is 12 * 34?",
    gold="408",
    variant="nudge",
    sample=0,
    temperature=0.7,
    prompt_ids=[1, 2, 3],
    gen_ids=[10, 11, 12, 13, 14],
    forced=[(2, 4)],
    calls=[Call(expression="12*34", response="408")],
    stop="eos",
    n_qwen=3,
    pred_exact="408",
    exact=True,
    text="...",
)


def rollout(**changes: Any) -> Rollout:
    return replace(BASE, **changes)
