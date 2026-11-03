import pytest

from qwenputer_handover.training.sft_data import IGNORE, build_example


def test_prompt_and_forced_spans_are_masked() -> None:
    e = build_example([1, 2], [10, 11, 12, 13, 14, 15], [(2, 4), (5, 6)])
    assert e.input_ids == [1, 2, 10, 11, 12, 13, 14, 15]
    assert e.labels == [IGNORE, IGNORE, 10, 11, IGNORE, IGNORE, 14, IGNORE]
    assert e.n_labeled == 3


def test_span_outside_the_generation_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_example([1], [10, 11], [(1, 3)])
