# SPDX-License-Identifier: Apache-2.0
"""Ming TTS process-boundary contracts."""

from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    PREPROCESSING_STAGE,
    REFERENCE_ENCODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)
from tests.unit_test.config.legacy_recipes import RECIPES_BY_FILE, legacy_config
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


def test_recipe_process_topology_compiles() -> None:
    config = legacy_config(RECIPES_BY_FILE["ming_omni_tts.yaml"])
    assert isinstance(config, MingTTSPipelineConfig)

    plan = build_compiled_process_topology(config)

    assert plan.stage_to_process == {
        PREPROCESSING_STAGE: "preprocessing",
        REFERENCE_ENCODE_STAGE: "ming_tts_aux",
        TTS_ENGINE_STAGE: "tts_engine",
        AUDIO_DECODE_STAGE: "ming_tts_aux",
    }
