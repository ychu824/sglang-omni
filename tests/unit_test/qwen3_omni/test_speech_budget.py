# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Qwen3-Omni speech limits and caller-facing errors."""

import pytest

from sglang_omni.models.qwen3_omni.speech_budget import (
    SLOWEST_LONG_SPEECH_FRAMES,
    SLOWEST_LONG_SPEECH_FRAMES_PER_TOKEN,
    SLOWEST_SPEECH_FRAMES_PER_TOKEN,
    SPEECH_BUDGET_MARGIN,
    speech_token_budget,
    thinker_max_new_tokens,
)
from sglang_omni.serve.openai_errors import generation_error_status_code


def slowest_speech_frames(tokens: int) -> float:
    return min(
        SLOWEST_SPEECH_FRAMES_PER_TOKEN * tokens,
        SLOWEST_LONG_SPEECH_FRAMES + SLOWEST_LONG_SPEECH_FRAMES_PER_TOKEN * tokens,
    )


@pytest.mark.parametrize("talker_frames", [50, 1000, 4096, 16384])
def test_the_budget_is_the_most_tokens_whose_slowest_speech_fits(
    talker_frames: int,
) -> None:
    budget = speech_token_budget(talker_frames)
    assert SPEECH_BUDGET_MARGIN * slowest_speech_frames(budget) <= talker_frames
    assert SPEECH_BUDGET_MARGIN * slowest_speech_frames(budget + 1) > talker_frames


@pytest.mark.parametrize(
    ("parameters", "speaks", "expected_max_new_tokens"),
    [
        pytest.param({}, True, 483, id="automatic-speech-limit"),
        pytest.param({"max_new_tokens": 483}, True, 483, id="exact-limit-allowed"),
        pytest.param({"max_new_tokens": 2000}, False, 2000, id="text-only"),
        pytest.param(
            {"max_new_tokens": 2000, "talker_max_new_tokens": 16384},
            True,
            2000,
            id="raised-talker-budget",
        ),
    ],
)
def test_chat_thinker_token_limit(
    parameters: dict[str, int], speaks: bool, expected_max_new_tokens: int
) -> None:
    assert (
        thinker_max_new_tokens(parameters, speaks=speaks, reject_explicit_overflow=True)
        == expected_max_new_tokens
    )


def test_explicit_chat_limit_above_speech_budget_is_a_bad_request() -> None:
    """An oversized spoken chat request reports the conflict to the caller."""
    with pytest.raises(ValueError, match=r"requested 2000.*at most 483") as error:
        thinker_max_new_tokens(
            {"max_new_tokens": 2000}, speaks=True, reject_explicit_overflow=True
        )

    assert generation_error_status_code(error.value) == 400


def test_speech_token_limit_can_still_be_clamped_during_generation() -> None:
    assert thinker_max_new_tokens({"max_new_tokens": 2000}, speaks=True) == 483
