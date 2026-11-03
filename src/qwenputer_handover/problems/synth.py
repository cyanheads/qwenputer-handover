"""src/qwenputer_handover/problems/synth.py — seeded synthetic training problems.

The kinds target where Qwen3.5-4B keeps the arithmetic to itself instead of calling the calc
tool: long multiplications, exact division with a fractional answer, multi-term expressions,
and word problems that bury large-number arithmetic in prose. Every number stays far below the
calc firmware's 62-digit limit.
"""

import ast
import operator
import random
from collections.abc import Callable
from fractions import Fraction

from qwenputer_handover.problems.problem import Problem

KINDS = ("mul", "div", "chain", "word")


def _num(rng: random.Random, lo_digits: int, hi_digits: int) -> int:
    d = rng.randint(lo_digits, hi_digits)
    return rng.randint(10 ** (d - 1), 10**d - 1)


def _fmt(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _mul(rng: random.Random) -> tuple[str, Fraction]:
    a, b = _num(rng, 4, 15), _num(rng, 3, 12)
    q = rng.choice(["What is {a} * {b}?", "Multiply {a} by {b}.", "Calculate {a} times {b}."])
    return q.format(a=a, b=b), Fraction(a * b)


def _div(rng: random.Random) -> tuple[str, Fraction]:
    b = _num(rng, 2, 9)
    a = b * _num(rng, 2, 10) + (rng.randint(1, b - 1) if rng.random() < 0.5 else 0)
    q = rng.choice(["Divide {a} by {b}.", "What is {a} / {b}?", "Calculate {a} divided by {b}."])
    return q.format(a=a, b=b), Fraction(a, b)


def _chain(rng: random.Random) -> tuple[str, Fraction]:
    """3-5 integer terms joined by + - *, with one parenthesised pair; Python's own precedence
    gives the gold, so the question is exactly the expression it evaluates."""
    terms = [str(_num(rng, 2, 9)) for _ in range(rng.randint(3, 5))]
    ops = [rng.choice("+-*") for _ in range(len(terms) - 1)]
    i = rng.randrange(len(ops))
    parts = []
    for j, t in enumerate(terms):
        if j == i:
            t = "(" + t
        if j == i + 1:
            t = t + ")"
        parts.append(t)
        if j < len(ops):
            parts.append(ops[j])
    expr = " ".join(parts)
    return f"Evaluate {expr}.", Fraction(_eval_int(expr))


_OPS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}


def _eval_int(expr: str) -> int:
    """The value of an integer + - * ( ) expression, with Python's precedence."""

    def walk(node: ast.expr) -> int:
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        raise ValueError(f"not an integer + - * expression: {expr!r}")

    return walk(ast.parse(expr, mode="eval").body)


WORD: list[tuple[str, Callable[[int, int, int], Fraction]]] = [
    (
        "A warehouse holds {a} crates with {b} items in each crate. A shipment takes away "
        "{c} items. How many items are left in the warehouse?",
        lambda a, b, c: Fraction(a * b - c),
    ),
    (
        "A factory makes {a} parts per hour for {b} hours, then packs them into boxes of {c}. "
        "How many boxes does it fill, counting a partly filled box as a fraction of a box?",
        lambda a, b, c: Fraction(a * b, c),
    ),
    (
        "A city's water system delivers {a} liters to each of {b} homes, plus {c} liters to "
        "parks. How many liters does it deliver in total?",
        lambda a, b, c: Fraction(a * b + c),
    ),
    (
        "An investor buys {a} shares at {b} cents each and pays a {c} cent fee. How many cents "
        "does the purchase cost?",
        lambda a, b, c: Fraction(a * b + c),
    ),
    (
        "A rocket burns {a} grams of fuel per second for {b} seconds. Its tank started with {c} "
        "kilograms more than that. How many grams did the tank start with?",
        lambda a, b, c: Fraction(a * b + c * 1000),
    ),
]


def _word(rng: random.Random) -> tuple[str, Fraction]:
    text, gold = rng.choice(WORD)
    a, b, c = _num(rng, 4, 9), _num(rng, 3, 7), _num(rng, 3, 8)
    return text.format(a=a, b=b, c=c), gold(a, b, c)


GENERATORS: dict[str, Callable[[random.Random], tuple[str, Fraction]]] = {
    "mul": _mul,
    "div": _div,
    "chain": _chain,
    "word": _word,
}


def synth(n: int, seed: int) -> list[Problem]:
    """``n`` problems cycling through every kind; the same (n, seed) always gives the same set,
    and a larger ``n`` extends a smaller one."""
    out = []
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        question, gold = GENERATORS[kind](random.Random(f"{seed}-{i}"))
        out.append(
            Problem(
                id=f"synth-{kind}-{seed}-{i}",
                dataset="synth",
                split="train",
                question=question,
                gold=_fmt(gold),
            )
        )
    return out
