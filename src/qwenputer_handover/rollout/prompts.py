"""src/qwenputer_handover/rollout/prompts.py — the prompt variants, rendered as Qwen chat ids.

Both variants offer upstream's ``calc`` tool and ask for a boxed answer (upstream's ``SUFFIX``);
they differ only in the system message:

- ``nudge``: upstream's system nudge to use calc for every arithmetic step (its eval prompt)
- ``bare``: no system message, so calling calc is the model's own decision

Rollouts sampled under ``nudge`` are trained under ``bare`` (context distillation): the target
behavior is handing arithmetic over without being told to.
"""

from typing import assert_never

from chipformer.q35.evals import SUFFIX
from chipformer.rv.chat import CALC_TOOL
from chipformer.rv.evalarith import NUDGE
from transformers import PreTrainedTokenizerBase

from qwenputer_handover.rollout.variants import Variant


def messages(question: str, variant: Variant) -> list[dict[str, str]]:
    user = {"role": "user", "content": question + SUFFIX}
    match variant:
        case "nudge":
            return [{"role": "system", "content": NUDGE}, user]
        case "bare":
            return [user]
        case _:
            assert_never(variant)


def prompt_ids(tok: PreTrainedTokenizerBase, question: str, variant: Variant) -> list[int]:
    """The chat-templated prompt through the assistant header, thinking disabled (upstream's
    eval setting)."""
    ids = tok.apply_chat_template(
        messages(question, variant),
        tools=[CALC_TOOL],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise TypeError(f"apply_chat_template returned {type(ids).__name__}, not token ids")
    return [i for i in ids if isinstance(i, int)]
