# SPDX-License-Identifier: Apache-2.0
"""Every deleted example YAML launches identically from its documented argv.

The recipes themselves live in legacy_recipes so model tests can share them;
this module is the equivalence gate, plus the consumers that compose the
same profiles (the router manifest, the CI worker args, the docs generator).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from sglang_omni.config.schema import PipelineConfig
from tests.unit_test.config.legacy_recipes import (
    LEGACY_RECIPES,
    RECIPES_BY_FILE,
    LegacyRecipe,
    launch_cli,
    legacy_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
INT4_QWEN3_OMNI = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"


@pytest.fixture
def launch(monkeypatch: pytest.MonkeyPatch):
    return lambda argv: launch_cli(monkeypatch, argv)


def assert_same_launch(subject: PipelineConfig, expected: PipelineConfig) -> None:
    assert type(subject) is type(expected)
    assert subject.model_dump() == expected.model_dump()
    for stage, expected_stage in zip(subject.stages, expected.stages, strict=True):
        assert stage.name == expected_stage.name
        if stage.engine is not None:
            assert stage.engine.overrides() == expected_stage.engine.overrides()


def test_the_audit_lists_every_deleted_example_exactly_once() -> None:
    assert len(LEGACY_RECIPES) == 33
    assert len(RECIPES_BY_FILE) == 33
    assert not any(
        (REPO_ROOT / "examples" / "configs" / name).exists() for name in RECIPES_BY_FILE
    )


@pytest.mark.parametrize(
    "recipe", LEGACY_RECIPES, ids=lambda recipe: recipe.legacy_file
)
def test_the_documented_argv_launches_what_the_example_file_did(
    hub_metadata, launch, recipe: LegacyRecipe
) -> None:
    expected = legacy_config(recipe)
    launched = launch(recipe.argv)

    assert_same_launch(launched, expected)
    assert launched.model_path == recipe.model_path
    assert launched.name == (recipe.name or recipe.model_path)


def test_ming_graph_batch_sizes_reach_the_engine_as_integers(
    hub_metadata, launch
) -> None:
    launched = launch(RECIPES_BY_FILE["ming_omni_tts.yaml"].argv)
    overrides = launched.stage_named("tts_engine").engine.overrides()

    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8]
    assert all(type(size) is int for size in overrides["cuda_graph_bs"])


def test_the_router_manifest_launches_the_h20_colocated_recipe(
    hub_metadata, launch
) -> None:
    """The shipped launcher manifest still names the H20 profile: as flags now."""
    from sglang_omni_router.python.launcher.config import load_launcher_config
    from sglang_omni_router.python.launcher.local import LocalLauncher

    manifest = load_launcher_config(
        REPO_ROOT / "examples" / "configs" / "qwen3_omni_router.yaml"
    )
    command = LocalLauncher(manifest).build_worker_command(8011)
    assert command[:2] == ["sgl-omni", "serve"]

    launched = launch(command[2:])

    assert_same_launch(
        launched, legacy_config(RECIPES_BY_FILE["qwen3_omni_colocated_h20.yaml"])
    )


def test_the_ci_worker_args_keep_their_profiles_and_context_overrides(
    hub_metadata, launch
) -> None:
    from tests.test_model import conftest as ci

    fp8 = launch(
        [
            "--model-path",
            ci.QWEN3_OMNI_FP8_MODEL_PATH,
            *shlex.split(ci.QWEN3_OMNI_FP8_COLOCATED_VIDEO_ARGS),
        ]
    )
    bf16 = launch(
        [
            "--model-path",
            ci.QWEN3_OMNI_MODEL_PATH,
            *shlex.split(ci.QWEN3_OMNI_BF16_COLOCATED_VIDEO_ARGS),
        ]
    )
    thinker = launch(
        [
            "--model-path",
            ci.QWEN3_OMNI_MODEL_PATH,
            *shlex.split(ci.QWEN3_OMNI_BF16_THINKER_ARGS),
        ]
    )

    for launched, legacy_file in (
        (fp8, "qwen3_omni_colocated_h100_fp8.yaml"),
        (bf16, "qwen3_omni_colocated_h100_bf16.yaml"),
    ):
        expected = legacy_config(RECIPES_BY_FILE[legacy_file])
        assert type(launched) is type(expected)
        assert launched.name == expected.name
        assert launched.model_path == expected.model_path
        for stage in expected.stages:
            assert (
                launched.stage_named(stage.name).gpu_memory_fraction
                == stage.gpu_memory_fraction
            )
            if stage.engine is not None:
                assert (
                    launched.stage_named(stage.name).engine.overrides()
                    == stage.engine.overrides()
                )
        assert (
            launched.stage_named("thinker").factory.max_seq_len
            == ci.QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN
        )
        assert (
            launched.stage_named("preprocessing").factory.max_seq_len
            == ci.QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN
        )
    assert_same_launch(
        thinker, legacy_config(RECIPES_BY_FILE["qwen3_omni_mmmu_h100.yaml"])
    )


def test_the_replica_ci_args_are_the_frozen_profile_and_its_single_instance_twin(
    hub_metadata, launch
) -> None:
    from tests.test_model import test_qwen3_omni_process_replicas as replica_ci

    replica = launch(["--model-path", replica_ci.MODEL_PATH, *replica_ci.REPLICA_ARGS])
    single = launch(
        ["--model-path", replica_ci.MODEL_PATH, *replica_ci.SINGLE_INSTANCE_ARGS]
    )
    profile = RECIPES_BY_FILE["qwen3_omni_speech_code2wav_replica2_ci.yaml"]

    assert_same_launch(replica, legacy_config(profile))
    # The A/B baseline keeps every stage budget and drops only the replica
    # placement and the pipeline name that tells the two servers apart.
    twin = replace(
        profile,
        name="qwen3-omni-speech-code2wav-single-ci",
        writes=tuple(
            write for write in profile.writes if not write[0].startswith("processes.")
        ),
    )
    assert_same_launch(single, legacy_config(twin))
    assert single.processes == {}


GENERATOR_JS = REPO_ROOT / "docs" / "_static" / "js" / "qwen3_omni_server_generator.js"
ENUMERATE_GENERATOR_COMMANDS = """
const generator = require(process.argv[1]);
const dims = generator.dimensions;
const out = [];
for (const mode of Object.keys(dims.MODES))
  for (const topo of Object.keys(dims.TOPOLOGIES))
    for (const prec of Object.keys(dims.PRECISIONS))
      for (const tp of Object.keys(dims.THINKER_TP))
        for (const hw of Object.keys(dims.HARDWARE)) {
          const ctx = { mode, topo, prec, tp, hw };
          out.push({ ctx, command: generator.buildCommand(ctx) });
        }
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the docs generator"
)
def test_every_generated_docs_command_parses_and_colocated_ones_match_the_profiles(
    hub_metadata, launch
) -> None:
    """The interactive generator is a docs surface: every command it can emit
    must be one the CLI accepts, and the colocated ones must land on the
    hardware profiles the example files used to carry."""
    completed = subprocess.run(
        ["node", "-e", ENUMERATE_GENERATOR_COMMANDS, str(GENERATOR_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(completed.stdout)
    assert len(generated) == 2 * 2 * 3 * 2 * 2

    profiles = {
        ("bf16", "h20"): "qwen3_omni_colocated_h20.yaml",
        ("bf16", "h200"): "qwen3_omni_colocated_h200.yaml",
        ("fp8", "h20"): "qwen3_omni_fp8_colocated.yaml",
        ("fp8", "h200"): "qwen3_omni_fp8_colocated.yaml",
        ("int4", "h20"): "qwen3_omni_colocated_h20.yaml",
        ("int4", "h200"): "qwen3_omni_colocated_h200.yaml",
    }
    for entry in generated:
        ctx, command = entry["ctx"], entry["command"]
        tokens = shlex.split(command.replace("\\\n", " "))
        assert tokens[:2] == ["sgl-omni", "serve"], command
        assert "--config" not in tokens, command
        launched = launch(tokens[2:])
        if ctx["mode"] == "speech" and ctx["topo"] == "colocated":
            profile = RECIPES_BY_FILE[profiles[(ctx["prec"], ctx["hw"])]]
            if ctx["prec"] == "int4":
                # INT4 colocated keeps the BF16 hardware budgets on the
                # AutoRound checkpoint.
                profile = replace(profile, model_path=INT4_QWEN3_OMNI)
            assert_same_launch(launched, legacy_config(profile))
        elif ctx["mode"] == "text-only":
            assert type(launched).__name__ == "Qwen3OmniPipelineConfig"
        else:
            assert type(launched).__name__ == "Qwen3OmniSpeechPipelineConfig"
