"""src/qwenputer_handover/problems/problem.py — the problem record every stage passes around."""

from dataclasses import dataclass
from typing import Literal

Split = Literal["train", "eval"]


@dataclass(frozen=True)
class Problem:
    """One question with an exact gold answer.

    ``split`` keeps training and evaluation apart: rejection sampling only accepts ``train``
    problems, so an SFT set can never contain an item the evals score."""

    id: str
    dataset: str
    split: Split
    question: str
    gold: str
