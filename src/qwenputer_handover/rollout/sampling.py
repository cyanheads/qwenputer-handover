"""src/qwenputer_handover/rollout/sampling.py — temperature / top-k / top-p sampling for
upstream's batched decoder.

``chipformer.q35.batch.BatchDecoder`` picks each token with ``argmax`` inside a captured CUDA
graph. ``Sampling`` swaps that choice, on one decoder instance, for a Gumbel-max draw over the
top-k candidates. The noise is a counter-based hash (splitmix64) of the row's seed and its
position, both device buffers the graph reads, so every replay draws fresh noise with no host-side
RNG inside the graph. Each prompt the decoder admits gets the next seed, so two rollouts of one
prompt draw independently. A prompt's first token comes from the eager prefill and is drawn with a
seeded ``torch.Generator``.
"""

from typing import Any

import torch

# splitmix64's increment and multipliers, as int64 (tensor arithmetic wraps mod 2**64)
GOLDEN, MUL1, MUL2 = -7046029254386353131, -4658895280553007687, -7723592293110705685


def _shr(x: torch.Tensor, s: int) -> torch.Tensor:
    """Logical right shift of int64 (``>>`` on a tensor is arithmetic)."""
    return (x >> s) & ((1 << (64 - s)) - 1)


def uniform(seed: torch.Tensor, pos: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """(n, k) floats in (0, 1), a hash of (seed[i], pos[i], cand[j]): independent across rows,
    steps and candidates, and the same for the same inputs. ``cand`` is an int64 buffer of
    candidate indices (0..k-1), not an ``arange`` built in the graph: Inductor folds an arange
    times a 64-bit constant into an int32 index expression, which overflows."""
    z = seed[:, None] * GOLDEN + pos[:, None] * MUL1 + cand[None, :] * MUL2 + GOLDEN
    z = (z ^ _shr(z, 30)) * MUL1
    z = (z ^ _shr(z, 27)) * MUL2
    z = z ^ _shr(z, 31)
    return (_shr(z, 40).float() + 0.5) / 2**24


def choose(
    logits: torch.Tensor, u: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Token ids (n,) drawn from logits (n, V) as ``generate`` samples them (temperature, then
    the ``top_k`` most likely, then the shortest run of those holding ``top_p`` of the mass), by
    Gumbel-max with the uniforms ``u`` (n, top_k)."""
    v, idx = logits.float().topk(top_k, dim=-1)
    v = v / temperature
    p = v.softmax(-1)
    v = v.masked_fill(p.cumsum(-1) - p >= top_p, float("-inf"))
    gumbel = -torch.log(-torch.log(u.clamp(1e-12, 1 - 1e-7)))
    return idx.gather(-1, (v + gumbel).argmax(-1, keepdim=True)).squeeze(-1)


class Sampling:
    """Takes over the token choice of one ``BatchDecoder`` (built, not yet run)."""

    def __init__(self, decoder: Any, temperature: float, top_k: int, top_p: float, seed: int):
        self.dec = decoder
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.next_seed = seed << 32
        self.generator = torch.Generator(device=decoder.dev)
        self.generator.manual_seed(seed)
        self.seeds = {
            n: torch.zeros(n, dtype=torch.long, device=decoder.dev) for n in decoder.sizes
        }
        self.cand = torch.arange(top_k, device=decoder.dev)
        for s in [*self.seeds.values(), self.cand]:
            torch._dynamo.mark_static_address(s)
        self.greedy_mark = decoder.mark_prompt
        decoder._argmax = self.choose
        decoder.mark_prompt = self.mark_prompt

    def mark_prompt(self, n: int, b: int, ids: list[int]) -> None:
        """Row ``b`` of the batch-``n`` level took a new prompt: give it the next seed."""
        self.greedy_mark(n, b, ids)
        self.seeds[n][b] = self.next_seed
        self.next_seed += 1

    def choose(self, logits: torch.Tensor) -> torch.Tensor:
        if self.dec.banned is not None:
            logits = logits.index_fill(-1, self.dec.banned, float("-inf"))
        if logits.dim() == 1:  # a prompt's first token, from the eager prefill
            u = torch.rand(1, self.top_k, generator=self.generator, device=logits.device)
            return choose(logits[None], u, self.temperature, self.top_k, self.top_p)[0]
        n = logits.shape[0]
        pos = self.dec.levels[n][2]  # each row's input slot: one more every step
        u = uniform(self.seeds[n], pos, self.cand)
        return choose(logits, u, self.temperature, self.top_k, self.top_p)
