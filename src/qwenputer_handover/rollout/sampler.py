"""src/qwenputer_handover/rollout/sampler.py — sample base Qwen3.5 with the reference machine
answering its calc calls, and store every rollout.

The machine is upstream's ``BridgeHost`` on the vendored ``calc-bridge.elf``: its responses are
byte-identical to what the merged rv1 checkpoint's circuit prints, so rollouts from the plain
base model are the merged model's conversations at base-model speed (upstream's ``pytool
--host bridge`` condition, sampled).

Decoding is upstream's ``BatchDecoder`` (continuous batching, one CUDA graph per step, GQA
attention without copying the cache), with ``Sampling`` in place of its argmax when the
temperature is above 0 and a ``Forcer`` per rollout in its token hook.
"""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from chipformer.q35 import arith_bench
from chipformer.q35.batch import BatchDecoder
from chipformer.rv import bridge
from chipformer.rv.evalarith import BridgeHost
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForConditionalGeneration

from qwenputer_handover.problems.problem import Problem
from qwenputer_handover.rollout import records
from qwenputer_handover.rollout.forcing import Forcer, RepeatCut
from qwenputer_handover.rollout.prompts import prompt_ids
from qwenputer_handover.rollout.records import Rollout, Stop
from qwenputer_handover.rollout.sampling import Sampling
from qwenputer_handover.rollout.variants import Variant
from qwenputer_handover.runtime import device

VIEW = 1024  # KV slots per attention view: a step reads the smallest view covering its rows


@dataclass(frozen=True)
class Decoding:
    temperature: float  # 0: greedy
    top_p: float
    top_k: int
    budget: int  # Qwen-written tokens per rollout
    stop_at_repeat: bool  # end a rollout at its first repeated calc call (training scans)


@dataclass(frozen=True)
class Generation:
    gen_ids: list[int]
    forced: list[bool]
    stop: Stop


class Sampler:
    def __init__(self, model_dir: Path, firmware: Path) -> None:
        self.tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_dir)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(  # returned in eval mode
            model_dir, dtype=torch.bfloat16, device_map=device()
        )
        self.text = bridge.Text(model_dir / "tokenizer.json")
        self.firmware = str(firmware)
        self.eos = {self._id("<|im_end|>"), self._id("<|endoftext|>")}

    def _id(self, token: str) -> int:
        i = self.tok.convert_tokens_to_ids(token)
        if not isinstance(i, int):
            raise KeyError(f"{token} is not a single token of this tokenizer")
        return i

    def decode(self, ids: list[int]) -> str:
        """Token ids as text, special tokens kept (the form ``records.parse_calls`` reads)."""
        text = self.tok.decode(ids, skip_special_tokens=False)
        if not isinstance(text, str):
            raise TypeError(f"decode returned {type(text).__name__} for one sequence")
        return text

    def _expression(self, body: list[int]) -> str:
        return records.call_expression(self.decode(body))

    def run(
        self,
        prompts: list[list[int]],
        decoding: Decoding,
        batch: int,
        seed: int,
        on_done: Callable[[int, Generation], None],
    ) -> None:
        """One rollout per prompt, ``batch`` decoding at a time; ``on_done(i, generation)`` as
        each finishes. Each rollout's machine answers its calls."""
        max_new = 2 * decoding.budget
        max_len = -(-(max(map(len, prompts)) + max_new) // VIEW) * VIEW
        decoder = BatchDecoder(
            self.model,
            batch=min(batch, len(prompts)),
            max_len=max_len,
            eos=sorted(self.eos),
            fold_gqa=True,
            view=VIEW,
        )
        if decoding.temperature > 0:
            Sampling(decoder, decoding.temperature, decoding.top_k, decoding.top_p, seed)
        forcers: dict[int, Forcer] = {}

        def hook(i: int, token: int) -> tuple[int, bool]:
            if i not in forcers:
                host = BridgeHost(self.text, self.firmware, prompts[i])
                cut = (
                    RepeatCut(self.text.TCO, self.text.TCC, self._expression)
                    if decoding.stop_at_repeat
                    else None
                )
                forcers[i] = Forcer(host, self.eos, decoding.budget, cut)
            return forcers[i].step(token)

        def done(i: int, gen: list[int]) -> None:
            f = forcers.pop(i)
            stop: Stop = (
                "repeat"
                if f.repeat_hit
                else "budget"
                if f.budget_hit
                else "eos"
                if gen[-1] in self.eos
                else "length"
            )
            on_done(i, Generation(list(gen), f.forced, stop))

        decoder.run(prompts, max_new, on_done=done, hook=hook)


def scan(
    sampler: Sampler,
    problems: list[Problem],
    variants: list[Variant],
    samples: int,
    decoding: Decoding,
    out_dir: Path,
    batch: int,
    seed: int,
) -> None:
    """Every (problem, variant, sample) not already in ``out_dir/<dataset>.<variant>.jsonl``,
    appended as each rollout finishes (resumable)."""
    if decoding.temperature == 0 and samples > 1:
        raise ValueError("greedy decoding gives one rollout per prompt: pass --samples 1")

    def path(dataset: str, variant: str) -> Path:
        return out_dir / f"{dataset}.{variant}.jsonl"

    done = {
        r.key
        for d in {p.dataset for p in problems}
        for v in variants
        if path(d, v).exists()
        for r in records.read(path(d, v))
    }
    jobs = [
        (p, v, s)
        for p in problems
        for v in variants
        for s in range(samples)
        if (p.id, v, s) not in done
    ]
    print(json.dumps({"todo": len(jobs), "done": len(done)}), file=sys.stderr, flush=True)
    if not jobs:
        return
    prompts = [prompt_ids(sampler.tok, p.question, v) for p, v, _ in jobs]
    t0 = time.time()
    tally = {"rollouts": 0, "exact": 0, "calls": 0, "budget": 0, "repeat": 0}

    def finished(i: int, g: Generation) -> None:
        p, v, s = jobs[i]
        r = _rollout(sampler, p, v, s, decoding, prompts[i], g)
        records.append(path(p.dataset, v), [r])
        tally["rollouts"] += 1
        tally["exact"] += r.exact
        tally["calls"] += len(r.calls)
        tally["budget"] += r.stop == "budget"
        tally["repeat"] += r.stop == "repeat"
        if tally["rollouts"] % batch == 0 or tally["rollouts"] == len(jobs):
            rate = tally["rollouts"] / (time.time() - t0) * 60
            progress = {**tally, "of": len(jobs), "per_min": round(rate, 1)}
            print(json.dumps(progress), file=sys.stderr, flush=True)

    sampler.run(prompts, decoding, batch, seed, finished)


def _rollout(
    sampler: Sampler,
    p: Problem,
    variant: Variant,
    sample: int,
    decoding: Decoding,
    prompt: list[int],
    g: Generation,
) -> Rollout:
    text = sampler.decode(g.gen_ids)
    scored = arith_bench.exact_fields({"gold": p.gold}, text)
    return Rollout(
        problem_id=p.id,
        dataset=p.dataset,
        split=p.split,
        question=p.question,
        gold=p.gold,
        variant=variant,
        sample=sample,
        temperature=decoding.temperature,
        prompt_ids=prompt,
        gen_ids=g.gen_ids,
        forced=records.forced_spans(g.forced),
        calls=records.parse_calls(text),
        stop=g.stop,
        n_qwen=g.forced.count(False),
        pred_exact=scored["pred_exact"],
        exact=scored["exact"],
        text=text,
    )
