import pytest

from qwenputer_handover.training.lora import lr_at


def test_warmup_then_cosine_to_zero() -> None:
    total = 100
    assert lr_at(0, total, 0.1) == pytest.approx(0.1)
    assert lr_at(9, total, 0.1) == pytest.approx(1.0)
    assert lr_at(10, total, 0.1) == pytest.approx(1.0)
    assert lr_at(55, total, 0.1) == pytest.approx(0.5)
    assert lr_at(100, total, 0.1) == pytest.approx(0.0)
