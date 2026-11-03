import pytest

from qwenputer_handover.curation.selection import select
from qwenputer_handover.rollout.records import Call

from .helpers import rollout


def test_keeps_only_exact_machine_answered_finished_rollouts() -> None:
    good = rollout()
    rejected = [
        rollout(sample=1, exact=False),
        rollout(sample=2, stop="budget"),
        rollout(sample=3, calls=[Call("12*34", "407")]),  # answer not from the machine
        rollout(sample=4, calls=[Call("12*34", "408"), Call("12*34", "408")]),  # re-verified
    ]
    assert select([good, *rejected], per_problem=5) == [good]


def test_error_followed_by_recovery_is_kept() -> None:
    r = rollout(calls=[Call("12*34*1", "error (too complex: split it)"), Call("12*34", "408")])
    assert select([r], per_problem=1) == [r]


def test_shortest_distinct_generations_per_problem() -> None:
    long = rollout(sample=0, n_qwen=50, gen_ids=[1, 2, 3])
    short = rollout(sample=1, n_qwen=10, gen_ids=[4, 5, 6])
    dup = rollout(sample=2, n_qwen=10, gen_ids=[4, 5, 6])
    other = rollout(problem_id="p2", n_qwen=30)
    assert select([long, short, dup, other], per_problem=1) == [other, short]
    assert select([long, short, dup], per_problem=2) == [short, long]


def test_eval_problems_never_reach_an_sft_set() -> None:
    with pytest.raises(ValueError, match="train rollouts only"):
        select([rollout(), rollout(problem_id="arith-3", split="eval")], per_problem=1)
