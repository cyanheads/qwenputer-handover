"""src/qwenputer_handover/problems/sources.py — every problem set, by name, as ``Problem`` lists.

Evaluation sets are upstream's own loaders (Qwenputer's ``chipformer``), so a score here is
directly comparable with its published tables: ``arith`` (seeded synthetic), the
``q35.arith_bench`` sets (DeepMind Mathematics ``dmx_*`` / ``dmi_*`` / ``dmxc_*``, NUPA
``nupa_*``, ``bbh_mat2``, ``eleu_*``) and ``gsm8k`` / ``gsmhard``. Training sets never overlap
them: ``synth`` (this package's generator) and ``gsm8k_train`` (GSM8K's train split).
"""

import random

from chipformer.q35 import evals
from chipformer.rv import evalarith
from datasets import load_dataset

from qwenputer_handover.problems.problem import Problem
from qwenputer_handover.problems.synth import synth

TRAIN_SETS = ("synth", "gsm8k_train")


def load(name: str, n: int | None, seed: int) -> list[Problem]:
    """``n`` problems of set ``name`` (``None``: the loader's default count); the same
    arguments always return the same problems."""
    if name == "synth":
        if n is None:
            raise ValueError("synth has no default size: pass --n")
        return synth(n, seed)
    if name == "gsm8k_train":
        return _gsm8k_train(n, seed)
    return _upstream(name, n, seed)


def _gsm8k_train(n: int | None, seed: int) -> list[Problem]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = range(len(ds)) if n is None else sorted(random.Random(seed).sample(range(len(ds)), n))
    return [
        Problem(
            id=f"gsm8k_train-{i}",
            dataset="gsm8k_train",
            split="train",
            question=ds[i]["question"],
            gold=ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for i in idx
    ]


def _upstream(name: str, n: int | None, seed: int) -> list[Problem]:
    if name in ("gsm8k", "gsmhard"):
        raw = evals.load_data(name, n or 200, seed)
    else:
        raw = evalarith.problems(name, n or (40 if name == "arith" else None), seed)
    return [
        Problem(id=p["id"], dataset=name, split="eval", question=p["question"], gold=str(p["gold"]))
        for p in raw
    ]
