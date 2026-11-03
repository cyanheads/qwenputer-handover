"""src/qwenputer_handover/curation/metrics.py — what a set of rollouts says about hand-over.

Every figure is a direct count over stored rollouts, grouped by (dataset, variant):

| Column | Meaning |
|:---|:---|
| exact | final answer equals gold as an exact rational (upstream's ``exact_fields``) |
| handover | at least one calc call |
| from_machine | the final answer is a value the machine printed |
| budget | stopped at the Qwen-token budget (ran out instead of answering) |
| cut | ended at a repeated calc call (only scans run with ``--stop-at-repeat``) |
| calls | mean calc calls per rollout |
| err_calls | share of calls answered with an error |
| repeat_calls | share of calls repeating an expression already sent (re-verification) |
| qwen_tok | mean tokens Qwen wrote |
| solved | problems with at least one exact rollout |
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from qwenputer_handover.rollout.records import Rollout


def as_fraction(s: str) -> Fraction | None:
    """An integer, decimal or ``p/q`` string as an exact rational; ``None`` for anything else
    (an error message, an expression)."""
    s = s.replace(",", "").strip()
    try:
        return Fraction(s) if "/" in s else Fraction(Decimal(s))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def from_machine(r: Rollout) -> bool:
    if r.pred_exact is None:
        return False
    pred = as_fraction(r.pred_exact)
    return pred is not None and any(as_fraction(c.response) == pred for c in r.calls)


def error_calls(r: Rollout) -> int:
    return sum(c.response.startswith("error") for c in r.calls)


def repeat_calls(r: Rollout) -> int:
    return len(r.calls) - len({c.expression for c in r.calls})


@dataclass(frozen=True)
class Summary:
    dataset: str
    variant: str
    rollouts: int
    problems: int
    exact: float
    handover: float
    from_machine: float
    budget: float
    cut: float
    calls: float
    err_calls: float
    repeat_calls: float
    qwen_tok: float
    solved: int


def summarize(rollouts: Iterable[Rollout]) -> list[Summary]:
    groups: dict[tuple[str, str], list[Rollout]] = defaultdict(list)
    for r in rollouts:
        groups[r.dataset, r.variant].append(r)
    out = []
    for (dataset, variant), rs in sorted(groups.items()):
        n, n_calls = len(rs), sum(len(r.calls) for r in rs)
        out.append(
            Summary(
                dataset=dataset,
                variant=variant,
                rollouts=n,
                problems=len({r.problem_id for r in rs}),
                exact=sum(r.exact for r in rs) / n,
                handover=sum(bool(r.calls) for r in rs) / n,
                from_machine=sum(from_machine(r) for r in rs) / n,
                budget=sum(r.stop == "budget" for r in rs) / n,
                cut=sum(r.stop == "repeat" for r in rs) / n,
                calls=n_calls / n,
                err_calls=sum(map(error_calls, rs)) / n_calls if n_calls else 0.0,
                repeat_calls=sum(map(repeat_calls, rs)) / n_calls if n_calls else 0.0,
                qwen_tok=sum(r.n_qwen for r in rs) / n,
                solved=len({r.problem_id for r in rs if r.exact}),
            )
        )
    return out


def table(rows: list[Summary]) -> str:
    head = (
        "| dataset | variant | rollouts | problems | exact | handover | from_machine | budget "
        "| cut | calls | err_calls | repeat_calls | qwen_tok | solved |"
    )
    lines = [head, "|:---|:---|" + "---:|" * 12]
    for s in rows:
        lines.append(
            f"| {s.dataset} | {s.variant} | {s.rollouts} | {s.problems} | {s.exact:.1%} "
            f"| {s.handover:.1%} | {s.from_machine:.1%} | {s.budget:.1%} | {s.cut:.1%} "
            f"| {s.calls:.2f} "
            f"| {s.err_calls:.1%} | {s.repeat_calls:.1%} | {s.qwen_tok:.0f} "
            f"| {s.solved}/{s.problems} |"
        )
    return "\n".join(lines)
