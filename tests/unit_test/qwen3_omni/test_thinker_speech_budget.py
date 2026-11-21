# SPDX-License-Identifier: Apache-2.0
"""A spoken Qwen3-Omni reply's thinker budget is no more than its talker can say."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytest.importorskip("xxhash")

from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_scheduler_adapters,
)
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer, make_qwen_payload


@pytest.mark.parametrize(
    ("speech_enabled", "output_modalities", "params", "max_new_tokens"),
    [
        pytest.param(True, None, {}, 483, id="spoken"),
        pytest.param(True, ["text", "audio"], {}, 483, id="spoken_and_written"),
        pytest.param(True, ["text"], {}, 2048, id="text_only_request"),
        pytest.param(False, None, {}, 2048, id="text_only_deployment"),
        pytest.param(True, None, {"max_new_tokens": 100}, 100, id="smaller_own_limit"),
        pytest.param(
            True, None, {"talker_max_new_tokens": 8192}, 1103, id="raised_talker"
        ),
        pytest.param(
            True, None, {"talker_max_new_tokens": 16384}, 2048, id="thinker_limit"
        ),
        pytest.param(True, None, {"talker_max_new_tokens": 50}, 4, id="short_talker"),
    ],
)
def test_a_spoken_request_writes_no_more_than_its_talker_can_say(
    monkeypatch: pytest.MonkeyPatch,
    speech_enabled: bool,
    output_modalities: list[str] | None,
    params: dict[str, int],
    max_new_tokens: int,
) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        capture_prompt_hidden_states=False,
        speech_enabled=speech_enabled,
    )
    payload = make_qwen_payload(params=params)
    payload.request.metadata = {"output_modalities": output_modalities}
    request = request_builder(payload).req
    assert request.sampling_params.max_new_tokens == max_new_tokens
