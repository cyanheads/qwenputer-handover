"""src/qwenputer_handover/training/sft_data.py — selected rollouts as (input_ids, labels) pairs.

The loss covers only what Qwen wrote: the prompt is masked, and so is every machine-forced span
(the tool response the circuit prints), because the merged model never predicts those tokens
with Qwen's head. ``</tool_call>``, the reasoning around each call and the final ``<|im_end|>``
are all labeled.
"""

from dataclasses import dataclass

IGNORE = -100  # the label value torch's cross-entropy skips


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    labels: list[int]

    @property
    def n_labeled(self) -> int:
        return sum(label != IGNORE for label in self.labels)


def build_example(
    prompt_ids: list[int], gen_ids: list[int], forced: list[tuple[int, int]]
) -> Example:
    masked = [False] * len(gen_ids)
    for start, end in forced:
        if not 0 <= start < end <= len(gen_ids):
            raise ValueError(
                f"forced span {(start, end)} outside a {len(gen_ids)}-token generation"
            )
        masked[start:end] = [True] * (end - start)
    labels = [IGNORE] * len(prompt_ids) + [
        IGNORE if m else t for t, m in zip(gen_ids, masked, strict=True)
    ]
    return Example(input_ids=prompt_ids + gen_ids, labels=labels)
