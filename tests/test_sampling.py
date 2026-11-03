import torch

from qwenputer_handover.rollout.sampling import choose, uniform

N = 40_000


def draws(logits: list[float], temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """N draws from one row of logits, each with its own seed (as N admitted prompts get)."""
    row = torch.full((N, 8), float("-inf"))
    row[:, : len(logits)] = torch.tensor(logits)
    u = uniform(torch.arange(N), torch.zeros(N, dtype=torch.long), torch.arange(top_k))
    return choose(row, u, temperature, top_k, top_p)


def test_uniform_is_open_unit_and_deterministic() -> None:
    seed, pos = torch.arange(1000), torch.arange(1000) * 7
    u = uniform(seed, pos, torch.arange(20))
    assert u.shape == (1000, 20)
    assert bool(((u > 0) & (u < 1)).all())
    assert torch.equal(u, uniform(seed, pos, torch.arange(20)))
    assert abs(float(u.mean()) - 0.5) < 0.01


def test_uniform_changes_with_seed_and_position() -> None:
    base = uniform(torch.tensor([3]), torch.tensor([10]), torch.arange(20))
    assert not torch.equal(base, uniform(torch.tensor([4]), torch.tensor([10]), torch.arange(20)))
    assert not torch.equal(base, uniform(torch.tensor([3]), torch.tensor([11]), torch.arange(20)))


def test_draws_follow_the_softmax() -> None:
    picks = draws([2.0, 1.0, 0.0], temperature=1.0, top_k=3, top_p=1.0)
    freq = torch.bincount(picks, minlength=3)[:3].float() / N
    expected = torch.tensor([0.6652, 0.2447, 0.0900])
    assert float((freq - expected).abs().max()) < 0.01


def test_temperature_sharpens() -> None:
    picks = draws([2.0, 1.0, 0.0], temperature=0.5, top_k=3, top_p=1.0)
    share = float((picks == 0).float().mean())
    assert abs(share - 0.8668) < 0.01  # softmax([4, 2, 0])[0]


def test_top_k_drops_the_tail() -> None:
    picks = draws([2.0, 1.9, 1.8, 1.7], temperature=1.0, top_k=2, top_p=1.0)
    assert set(picks.tolist()) == {0, 1}


def test_top_p_keeps_the_shortest_run_holding_the_mass() -> None:
    # probabilities about [0.70, 0.26, 0.04]: 0.6 keeps only the first, 0.9 the first two
    assert set(draws([3.0, 2.0, 0.0], 1.0, 3, 0.6).tolist()) == {0}
    assert set(draws([3.0, 2.0, 0.0], 1.0, 3, 0.9).tolist()) == {0, 1}
