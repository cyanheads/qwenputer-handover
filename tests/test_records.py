from pathlib import Path

import pytest

from qwenputer_handover.rollout.records import Call, append, forced_spans, parse_calls, read

from .helpers import rollout

TEXT = (
    "First the product.\n<tool_call>\n<function=calc>\n<parameter=expression>\n123*456\n"
    "</parameter>\n</function>\n</tool_call>\n<tool_response>\n56088\n</tool_response>\n"
    "<think>\n\n</think>\n\nThen <tool_call>\n<function=calc>\n</function>\n</tool_call>\n"
    "<tool_response>\nerror\n</tool_response>\n<think>\n\n</think>\n\n\\boxed{56088}<|im_end|>"
)


def test_parse_calls_pairs_each_call_with_its_response() -> None:
    assert parse_calls(TEXT) == [
        Call(expression="123*456", response="56088"),
        Call(expression="<function=calc>\n</function>", response="error"),
    ]


def test_parse_calls_ignores_a_call_without_response() -> None:
    assert parse_calls("<tool_call>\n<parameter=expression>\n1+1\n</parameter>\n</tool_call>") == []


def test_forced_spans() -> None:
    assert forced_spans([]) == []
    assert forced_spans([False, True, True, False, True]) == [(1, 3), (4, 5)]
    assert forced_spans([True, True]) == [(0, 2)]


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    rows = [rollout(), rollout(sample=1, calls=[], forced=[])]
    append(tmp_path / "x.jsonl", rows[:1])
    append(tmp_path / "x.jsonl", rows[1:])
    assert list(read(tmp_path / "x.jsonl")) == rows


def test_parse_variants_rejects_unknown_names() -> None:
    from qwenputer_handover.rollout.variants import parse_variants

    assert parse_variants("bare,nudge") == ["bare", "nudge"]
    with pytest.raises(ValueError, match="nudg"):
        parse_variants("nudg")
