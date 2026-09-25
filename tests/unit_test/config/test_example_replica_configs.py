# SPDX-License-Identifier: Apache-2.0
"""The documented replica recipes must compile as the removed files did."""

from __future__ import annotations

import pytest

from sglang_omni.config.topology import compile_logical_processes
from tests.unit_test.config.legacy_recipes import RECIPES_BY_FILE, legacy_config

REPLICA_RECIPES = [
    "qwen3_omni_speech_replica2.yaml",
    "qwen3_omni_speech_code2wav_replica2_ci.yaml",
]


@pytest.mark.parametrize("legacy_file", REPLICA_RECIPES)
def test_replica_recipe_loads_and_declares_its_placement(legacy_file: str) -> None:
    config = legacy_config(RECIPES_BY_FILE[legacy_file])

    plan, _ = compile_logical_processes(config)
    replicated = [process for process in plan.processes if process.is_replicated]
    assert replicated, legacy_file
    # Note (Jiaxin Deng): replica_devices colocation is only valid with a
    # declared budget on every GPU stage it places, so the recipes must carry
    # gpu_memory_fraction for each replicated GPU stage.
    for process in replicated:
        for stage_name in process.stage_names:
            stage = config.stage_named(stage_name)
            if stage.gpu is not None:
                assert stage.gpu_memory_fraction is not None, (legacy_file, stage_name)
