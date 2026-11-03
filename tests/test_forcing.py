from qwenputer_handover.rollout.forcing import Forcer, RepeatCut

TCO, TCC, END, EOT = 49, 50, 99, 97
RESPONSE = [100, 101, 102]


class FakeHost:
    """Answers every closing tool-call token with RESPONSE; records what it was fed."""

    def __init__(self) -> None:
        self.fed: list[int] = []

    def feed(self, token: int) -> list[int]:
        self.fed.append(token)
        return list(RESPONSE) if token == TCC else []


def repeat_cut() -> RepeatCut:
    """A cut whose expression is the call body's token ids, comma-joined."""
    return RepeatCut(TCO, TCC, lambda body: ",".join(map(str, body)))


def decode(
    script: list[int], budget: int = 100, cut: RepeatCut | None = None
) -> tuple[Forcer, FakeHost, list[int]]:
    """A decoder whose model writes ``script`` in order: it moves on to its next token only
    when the hook kept the one it chose (a forced step drops the model's choice)."""
    host = FakeHost()
    forcer = Forcer(host, {END, EOT}, budget, cut)
    gen: list[int] = []
    pos = 0
    for _ in range(40):
        kept, stop = forcer.step(script[min(pos, len(script) - 1)])
        gen.append(kept)
        if not forcer.forced[-1]:
            pos += 1
        if stop:
            break
    return forcer, host, gen


def test_machine_response_is_forced_after_tool_call_close() -> None:
    forcer, host, gen = decode([5, TCC, 6, END])
    assert gen == [5, TCC, 100, 101, 102, 6, END]
    assert forcer.forced == [False, False, True, True, True, False, False]
    assert host.fed == [5, TCC, 6]  # forced tokens and the end of turn are never fed back
    assert forcer.n_qwen == 4 and not forcer.budget_hit


def test_any_end_token_ends_the_rollout_unfed() -> None:
    forcer, host, gen = decode([5, 6, EOT, 7])
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


def test_repeated_call_ends_the_rollout_at_its_close_unrun() -> None:
    script = [TCO, 7, TCC, 5, TCO, 8, TCC, 6, TCO, 7, TCC, 9, END]
    forcer, host, gen = decode(script, cut=repeat_cut())
    assert gen == [TCO, 7, TCC, *RESPONSE, 5, TCO, 8, TCC, *RESPONSE, 6, TCO, 7, TCC]
    assert host.fed.count(TCC) == 2  # the repeat never reaches the machine
    assert forcer.repeat_hit and not forcer.budget_hit


def test_repeats_run_on_without_a_cut() -> None:
    forcer, _, gen = decode([TCO, 7, TCC, TCO, 7, TCC, END])
    assert gen[-1] == END and not forcer.repeat_hit


def test_distinct_calls_run_to_the_end_of_turn() -> None:
    forcer, _, gen = decode([TCO, 7, TCC, TCO, 7, 7, TCC, END], cut=repeat_cut())
    assert gen[-1] == END and not forcer.repeat_hit


def test_call_bodies_are_read_as_parse_calls_reads_them() -> None:
    """A body runs from the first open to the first close (``<tool_call>a<tool_call>b
    </tool_call>`` is one call, body ``a<tool_call>b``); a close outside a call is no call."""
    cut = repeat_cut()
    assert not any(cut.repeats(t) for t in [TCO, 7, TCO, 8, TCC, TCC, TCO, 7, TCC])
    assert cut.sent == {f"7,{TCO},8", "7"}
    assert [cut.repeats(t) for t in [TCO, 7, TCO, 8, TCC]] == [False] * 4 + [True]
