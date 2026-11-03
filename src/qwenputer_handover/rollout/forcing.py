"""src/qwenputer_handover/rollout/forcing.py — let a machine write its own tokens into a
generation, one decoder step at a time.

Each rollout has a host: upstream's ``BridgeHost``, the reference RV32 machine running the calc
firmware, fed Qwen's tokens one at a time. When Qwen closes a tool call the host returns the
response tokens the merged model's circuit would print, and ``Forcer.step`` puts them in place of
the decoder's next tokens, one per step. The decoder feeds the kept token back, so the cache holds
the machine's tokens, and the token the model chose at a forced step is dropped. Qwen's own tokens
go to the host; the machine's never do.

A rollout also stops once Qwen itself has written ``budget`` tokens (upstream's per-problem budget
counts Qwen's tokens, not the machine's), after any response still being forced. With a
``RepeatCut`` it stops as soon as Qwen closes a call repeating an expression it already sent:
selection rejects such a rollout whatever follows, and re-check loops are where most rollouts
that run to the budget spend it.
"""

from collections import deque
from collections.abc import Callable
from typing import Protocol


class Host(Protocol):
    def feed(self, token: int) -> list[int]:
        """Qwen's next token; the tokens to force after it (empty unless it ran a call)."""
        ...


class RepeatCut:
    """Spots a calc call whose expression the rollout already sent, from Qwen's own tokens.

    A call's body runs from the first ``<tool_call>`` after the previous call to the next
    ``</tool_call>``, as ``records.parse_calls`` reads a stored generation; ``expression`` turns
    the body's token ids into the expression the same way."""

    def __init__(self, open_id: int, close_id: int, expression: Callable[[list[int]], str]):
        self.open_id, self.close_id = open_id, close_id
        self.expression = expression
        self.body: list[int] | None = None  # inside a call: its tokens so far
        self.sent: set[str] = set()

    def repeats(self, token: int) -> bool:
        """Qwen's next token; whether it closes a call repeating an earlier expression."""
        if self.body is None:
            if token == self.open_id:
                self.body = []
            return False
        if token != self.close_id:
            self.body.append(token)
            return False
        expression = self.expression(self.body)
        self.body = None
        if expression in self.sent:
            return True
        self.sent.add(expression)
        return False


class Forcer:
    """One rollout's forcing state, driven by the decoder's per-token hook."""

    def __init__(
        self, host: Host, eos: set[int], budget: int, cut: RepeatCut | None = None
    ) -> None:
        self.host = host
        self.eos = eos
        self.budget = budget
        self.cut = cut
        self.pending: deque[int] = deque()
        self.forced: list[bool] = []  # per generated token: machine-written
        self.n_qwen = 0
        self.budget_hit = False
        self.repeat_hit = False

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
            if self.cut is not None and self.cut.repeats(token):
                self.repeat_hit = True
                return kept, True
            self.pending.extend(self.host.feed(token))
        self.budget_hit = not self.pending and self.n_qwen >= self.budget
        return kept, self.budget_hit
