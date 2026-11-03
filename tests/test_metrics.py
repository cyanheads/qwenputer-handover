from fractions import Fraction

import pytest

from qwenputer_handover.curation.metrics import (
    as_fraction,
    error_calls,
    from_machine,
    repeat_calls,
    summarize,
)
from qwenputer_handover.rollout.records import Call

from .helpers import rollout


def test_as_fraction() -> None:
    assert as_fraction("1,234") == 1234
    assert as_fraction(" -2.50 ") == Fraction(-5, 2)
    assert as_fraction("14/6") == Fraction(7, 3)
    assert as_fraction("error (too complex: split it)") is None
    assert as_fraction("1/0") is None


def test_from_machine_compares_exact_values_not_strings() -> None:
    assert from_machine(rollout(pred_exact="5/2", calls=[Call("5/2", "2.5")]))
    assert not from_machine(rollout(pred_exact="408", calls=[Call("12*33", "396")]))
    assert not from_machine(rollout(pred_exact=None))


def test_error_and_repeat_calls() -> None:
    r = rollout(
        calls=[Call("1+1", "2"), Call("1+1", "2"), Call("9**999", "error (too long: split it)")]
    )
    assert error_calls(r) == 1
    assert repeat_calls(r) == 1


def test_summarize_groups_by_dataset_and_variant() -> None:
    rows = summarize(
        [
            rollout(),
            rollout(
                problem_id="p2", exact=False, pred_exact="1", calls=[], stop="budget", n_qwen=9
            ),
            rollout(variant="bare", sample=0),
        ]
    )
    nudge = next(s for s in rows if s.variant == "nudge")
    assert (nudge.rollouts, nudge.problems, nudge.solved) == (2, 2, 1)
    assert nudge.exact == nudge.handover == nudge.from_machine == nudge.budget == 0.5
    assert nudge.calls == 0.5
    assert nudge.qwen_tok == pytest.approx(6.0)
    assert [s.variant for s in rows] == ["bare", "nudge"]
