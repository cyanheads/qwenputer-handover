"""src/qwenputer_handover/rollout/records.py — one sampled conversation, as stored in JSONL.

A rollout keeps the exact token ids (prompt and generation) plus the spans of the generation
the machine wrote, so training can mask them and upstream's ``chipformer.rv.replay`` can check
them against the circuit. Files are appended one record per line and fsynced on every append: a
GPU box can lose power or be reclaimed mid-run, and a resumed scan trusts what is on disk.
"""

import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from qwenputer_handover.problems.problem import Split
from qwenputer_handover.rollout.variants import Variant

Stop = Literal["eos", "budget", "length"]


@dataclass(frozen=True)
class Call:
    expression: str
    response: str


@dataclass(frozen=True)
class Rollout:
    problem_id: str
    dataset: str
    split: Split
    question: str
    gold: str
    variant: Variant
    sample: int
    temperature: float
    prompt_ids: list[int]
    gen_ids: list[int]
    forced: list[tuple[int, int]]  # [start, end) spans of gen_ids the machine wrote
    calls: list[Call]
    stop: Stop
    n_qwen: int  # tokens Qwen wrote (gen_ids minus the forced spans)
    pred_exact: str | None
    exact: bool
    text: str

    @property
    def key(self) -> tuple[str, str, int]:
        return self.problem_id, self.variant, self.sample

    @staticmethod
    def from_json(line: str) -> "Rollout":
        d = json.loads(line)
        d["calls"] = [Call(**c) for c in d["calls"]]
        d["forced"] = [(s, e) for s, e in d["forced"]]
        return Rollout(**d)


TOOL_EXCHANGE = re.compile(
    r"<tool_call>(.*?)</tool_call>\s*<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL
)
EXPRESSION = re.compile(r"<parameter=expression>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_calls(text: str) -> list[Call]:
    """Each calc call in a decoded generation with the response that followed it; a call
    without an expression parameter keeps its raw body as the expression."""
    calls = []
    for body, response in TOOL_EXCHANGE.findall(text):
        m = EXPRESSION.search(body)
        calls.append(Call(expression=(m[1] if m else body).strip(), response=response.strip()))
    return calls


def forced_spans(flags: Iterable[bool]) -> list[tuple[int, int]]:
    """[start, end) runs of True in a per-token flag sequence."""
    spans: list[tuple[int, int]] = []
    start = None
    i = -1
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, i + 1))
    return spans


def read(path: Path) -> Iterator[Rollout]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Rollout.from_json(line)


def append(path: Path, rollouts: Iterable[Rollout]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
