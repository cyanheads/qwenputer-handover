"""src/qwenputer_handover/curation/selection.py — rejection sampling: the rollouts worth imitating.

A rollout is kept when it ended on its own (no budget or length cut), its answer is exact, that
answer is a value the machine printed, and it never re-sent an expression it already had an
answer for. Error responses are allowed: a call refused with ``error (too complex: split it)``
followed by smaller calls that reach the exact answer is the recovery the model should learn.
Per problem the shortest (fewest Qwen tokens) distinct generations win.
"""

from collections import defaultdict
from collections.abc import Iterable

from qwenputer_handover.curation.metrics import from_machine, repeat_calls
from qwenputer_handover.rollout.records import Rollout


def accept(r: Rollout) -> bool:
    return r.stop == "eos" and r.exact and from_machine(r) and repeat_calls(r) == 0


def select(rollouts: Iterable[Rollout], per_problem: int) -> list[Rollout]:
    """Up to ``per_problem`` accepted rollouts per problem; raises on any non-train rollout so
    an evaluation item can never reach an SFT set."""
    by_problem: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        if r.split != "train":
            raise ValueError(
                f"{r.problem_id} is an {r.split} problem: SFT takes train rollouts only"
            )
        if accept(r):
            by_problem[r.problem_id].append(r)
    out = []
    for problem_id in sorted(by_problem):
        seen: set[tuple[int, ...]] = set()
        for r in sorted(by_problem[problem_id], key=lambda r: (r.n_qwen, r.variant, r.sample)):
            gen = tuple(r.gen_ids)
            if gen not in seen and len(seen) < per_problem:
                seen.add(gen)
                out.append(r)
    return out
