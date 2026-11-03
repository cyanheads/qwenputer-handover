import re
from fractions import Fraction

from qwenputer_handover.problems.synth import synth


def test_same_seed_same_problems_and_larger_n_extends() -> None:
    assert synth(12, 3) == synth(12, 3)
    assert synth(20, 3)[:12] == synth(12, 3)
    assert synth(8, 3) != synth(8, 4)


def test_every_kind_appears_and_split_is_train() -> None:
    probs = synth(8, 0)
    assert {p.id.split("-")[1] for p in probs} == {"mul", "div", "chain", "word"}
    assert {p.split for p in probs} == {"train"}


def test_golds_match_the_questions() -> None:
    for p in synth(200, 1):
        nums = [int(x) for x in re.findall(r"\d+", p.question)]
        kind = p.id.split("-")[1]
        if kind == "mul":
            assert Fraction(p.gold) == nums[0] * nums[1]
        elif kind == "div":
            assert Fraction(p.gold) == Fraction(nums[0], nums[1])
        elif kind == "chain":
            expr = p.question.removeprefix("Evaluate ").removesuffix(".")
            assert Fraction(p.gold) == eval(expr)  # test oracle: Python's own evaluator
        else:
            Fraction(p.gold)  # word problems: gold is an exact rational


def test_numbers_fit_the_calc_firmware() -> None:
    for p in synth(400, 2):
        assert all(len(x) <= 62 for x in re.findall(r"\d+", p.question + " " + p.gold))
