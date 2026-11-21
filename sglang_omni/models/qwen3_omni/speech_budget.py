# SPDX-License-Identifier: Apache-2.0
"""Token budgets for Qwen3-Omni spoken replies."""

import math
from collections.abc import Mapping

THINKER_DEFAULT_MAX_NEW_TOKENS = 2048
TALKER_DEFAULT_MAX_NEW_TOKENS = 4096
# note (luojiaxuan): Over 193 replies of Qwen3-Omni-30B-A3B-Instruct's default voice that ended on their own (6 to 1830 thinker tokens, English and Chinese), a reply of n tokens never took more than 6.905 n codec frames nor more than 600 + 4.405 n (a list of numbers read out is the slowest). A spoken reply's thinker budget lets 1.5 times that fit in the talker's: a split by prompt left 0.21 of about 96 held-out replies above 1.5 times the envelope fitted on the rest, and 0.17 above twice it.
SLOWEST_SPEECH_FRAMES_PER_TOKEN = 6.905
SLOWEST_LONG_SPEECH_FRAMES = 600
SLOWEST_LONG_SPEECH_FRAMES_PER_TOKEN = 4.405
SPEECH_BUDGET_MARGIN = 1.5


def speech_token_budget(talker_max_new_tokens: int) -> int:
    """The thinker tokens of a spoken reply whose slowest speech, with the margin, fits in talker_max_new_tokens codec frames; a thinker writes at least one."""
    frames = talker_max_new_tokens / SPEECH_BUDGET_MARGIN
    return max(
        1,
        math.floor(frames / SLOWEST_SPEECH_FRAMES_PER_TOKEN),
        math.floor(
            (frames - SLOWEST_LONG_SPEECH_FRAMES) / SLOWEST_LONG_SPEECH_FRAMES_PER_TOKEN
        ),
    )


def thinker_max_new_tokens(
    parameters: Mapping[str, object],
    *,
    speaks: bool,
    reject_explicit_overflow: bool = False,
) -> int:
    """Resolve the thinker limit for a text or spoken reply."""
    requested_max_new_tokens = parameters.get(
        "max_new_tokens", THINKER_DEFAULT_MAX_NEW_TOKENS
    )
    if not isinstance(requested_max_new_tokens, (int, float, str)):
        raise ValueError("max_new_tokens must be an integer")
    elif not speaks:
        return int(requested_max_new_tokens)
    else:
        thinker_token_limit = int(requested_max_new_tokens)

    talker_max_new_tokens = parameters.get(
        "talker_max_new_tokens", TALKER_DEFAULT_MAX_NEW_TOKENS
    )
    if not isinstance(talker_max_new_tokens, (int, float, str)):
        raise ValueError("talker_max_new_tokens must be an integer")
    else:
        speech_token_limit = speech_token_budget(int(talker_max_new_tokens))

    if (
        reject_explicit_overflow
        and "max_new_tokens" in parameters
        and thinker_token_limit > speech_token_limit
    ):
        raise ValueError(
            f"Qwen3-Omni speech token budget exceeded: requested {thinker_token_limit} "
            f"text tokens, but talker_max_new_tokens={talker_max_new_tokens} allows "
            f"at most {speech_token_limit}. Lower max_tokens or "
            "max_completion_tokens, increase talker_max_new_tokens, "
            "or request text-only output."
        )
    else:
        return min(thinker_token_limit, speech_token_limit)
