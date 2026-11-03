from qwenputer_handover.rollout.forcing import Forcer

TCC, END, EOT = 50, 99, 97
RESPONSE = [100, 101, 102]


class FakeHost:
    """Answers every closing tool-call token with RESPONSE; records what it was fed."""

    def __init__(self) -> None:
        self.fed: list[int] = []

    def feed(self, token: int) -> list[int]:
        self.fed.append(token)
        return list(RESPONSE) if token == TCC else []


def decode(script: list[int], budget: int, steps: int = 20) -> tuple[Forcer, FakeHost, list[int]]:
    """A decoder whose model writes ``script`` in order: it moves on to its next token only
    when the hook kept the one it chose (a forced step drops the model's choice)."""
    host = FakeHost()
    forcer = Forcer(host, {END, EOT}, budget)
    gen: list[int] = []
    pos = 0
    for _ in range(steps):
        kept, stop = forcer.step(script[min(pos, len(script) - 1)])
        gen.append(kept)
        if not forcer.forced[-1]:
            pos += 1
        if stop:
            break
    return forcer, host, gen


def test_machine_response_is_forced_after_tool_call_close() -> None:
    forcer, host, gen = decode([5, TCC, 6, END], budget=100)
    assert gen == [5, TCC, 100, 101, 102, 6, END]
    assert forcer.forced == [False, False, True, True, True, False, False]
    assert host.fed == [5, TCC, 6]  # forced tokens and the end of turn are never fed back
    assert forcer.n_qwen == 4 and not forcer.budget_hit


def test_any_end_token_ends_the_rollout_unfed() -> None:
    forcer, host, gen = decode([5, 6, EOT, 7], budget=100)
    assert gen == [5, 6, EOT]
    assert host.fed == [5, 6]
    assert not forcer.budget_hit


def test_budget_stops_after_the_last_qwen_token() -> None:
    forcer, _, gen = decode([5, 6, 7, 8, 9, 10], budget=3)
    assert gen == [5, 6, 7]
    assert forcer.forced == [False, False, False]
    assert forcer.budget_hit


def test_pending_response_finishes_before_the_budget_stop() -> None:
    forcer, _, gen = decode([5, TCC, 6, 7], budget=2)
    assert gen == [5, TCC, 100, 101, 102]
    assert forcer.forced == [False, False, True, True, True]
    assert forcer.budget_hit
