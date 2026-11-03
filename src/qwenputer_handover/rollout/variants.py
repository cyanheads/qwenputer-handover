"""src/qwenputer_handover/rollout/variants.py — the prompt variant names (rendered in
``rollout/prompts.py``)."""

from typing import Literal

Variant = Literal["nudge", "bare"]
VARIANTS: tuple[Variant, ...] = ("nudge", "bare")


def parse_variants(names: str) -> list[Variant]:
    """A comma-separated variant list; any unknown name is an error."""
    by_name: dict[str, Variant] = {v: v for v in VARIANTS}
    unknown = [n for n in names.split(",") if n not in by_name]
    if unknown:
        raise ValueError(f"unknown variant {unknown[0]!r}: choose from {', '.join(VARIANTS)}")
    return [by_name[n] for n in names.split(",")]
