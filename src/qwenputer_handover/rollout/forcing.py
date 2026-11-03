"""src/qwenputer_handover/rollout/forcing.py — let a machine write its own tokens into a
generation, one decoder step at a time.

Each rollout has a host: upstream's ``BridgeHost``, the reference RV32 machine running the calc
firmware, fed Qwen's tokens one at a time. When Qwen closes a tool call the host returns the
response tokens the merged model's circuit would print, and ``Forcer.step`` puts them in place of
the decoder's next tokens, one per step. The decoder feeds the kept token back, so the cache holds
the machine's tokens, and the token the model chose at a forced step is dropped. Qwen's own tokens
go to the host; the machine's never do.

A rollout also stops once Qwen itself has written ``budget`` tokens (upstream's per-problem budget
counts Qwen's tokens, not the machine's), after any response still being forced.
"""

from collections import deque
from typing import Protocol


class Host(Protocol):
    def feed(self, token: int) -> list[int]:
        """Qwen's next token; the tokens to force after it (empty unless it ran a call)."""
        ...


class Forcer:
    """One rollout's forcing state, driven by the decoder's per-token hook."""

    def __init__(self, host: Host, eos: set[int], budget: int) -> None:
        self.host = host
        self.eos = eos
        self.budget = budget
        self.pending: deque[int] = deque()
        self.forced: list[bool] = []  # per generated token: machine-written
        self.n_qwen = 0
        self.budget_hit = False

    def step(self, token: int) -> tuple[int, bool]:
        """The decoder's next token; returns the token to keep and whether the rollout ends
        after it."""
        if self.pending:
            kept = self.pending.popleft()
            self.forced.append(True)
        else:
            kept = token
            self.forced.append(False)
            self.n_qwen += 1
            if token in self.eos:
                return kept, True
            self.pending.extend(self.host.feed(token))
        self.budget_hit = not self.pending and self.n_qwen >= self.budget
        return kept, self.budget_hit
