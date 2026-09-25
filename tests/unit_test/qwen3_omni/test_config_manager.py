# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.config import build_stage_placement_plan, resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni import config as qwen3_omni_config
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from tests.unit_test.config.legacy_recipes import RECIPES_BY_FILE, legacy_config
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


def make_stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_config_manager_parses_dotted_fraction_overrides_as_numbers() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    extra_args = manager.parse_extra_args(
        [
            "--image_encoder.gpu_memory_fraction",
            "0.05",
            "--audio_encoder.gpu_memory_fraction",
            "0.05",
            "--thinker.gpu_memory_fraction",
            "0.35",
            "--thinker.engine.mem_fraction_static",
            "0.35",
            "--talker_ar.gpu_memory_fraction",
            "0.35",
            "--talker_ar.engine.mem_fraction_static",
            "0.35",
            "--code2wav.gpu_memory_fraction",
            "0.05",
        ]
    )

    merged = manager.merge_config(extra_args)
    plan = build_stage_placement_plan(merged)

    assert make_stage(merged, "thinker").gpu_memory_fraction == pytest.approx(0.35)
    assert make_stage(merged, "thinker").engine.mem_fraction_static == pytest.approx(
        0.35
    )
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.85)


def test_config_manager_applies_dotted_tp_size_override() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"thinker.tp_size": 2, "thinker.gpu": [0, 1]})
    thinker = make_stage(merged, "thinker")

    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]


def test_config_manager_sets_tp_size_directly() -> None:
    """tp_size is the only spelling; the parallelism.tp mirror is gone."""
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"thinker.tp_size": 2, "thinker.gpu": [0, 1]})
    thinker = make_stage(merged, "thinker")

    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]


def test_config_manager_rejects_trailing_key_without_value() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))

    with pytest.raises(ValueError, match="Missing value"):
        manager.parse_extra_args(
            [
                "--thinker.gpu_memory_fraction",
                "0.35",
                "--thinker.engine.mem-fraction-static",
            ]
        )


def test_qwen3_omni_h20_colocated_recipe_loads_and_plans() -> None:
    config = legacy_config(RECIPES_BY_FILE["qwen3_omni_colocated_h20.yaml"])
    plan = build_stage_placement_plan(config)
    topology = build_compiled_process_topology(config)

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated-h20"
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]
    assert make_stage(config, "thinker").engine.mem_fraction_static is None
    assert make_stage(config, "talker_ar").engine.mem_fraction_static is None
    assert {
        stage.name: stage.gpu
        for stage in config.stages
        if stage.name
        in {
            "image_encoder",
            "audio_encoder",
            "thinker",
            "talker_ar",
            "code2wav",
        }
    } == {
        "image_encoder": 0,
        "audio_encoder": 0,
        "thinker": 0,
        "talker_ar": 0,
        "code2wav": 0,
    }


def test_qwen3_omni_mmsu_recipe_uses_text_pipeline() -> None:
    config = legacy_config(RECIPES_BY_FILE["qwen3_omni_mmsu.yaml"])
    plan = build_stage_placement_plan(config)
    thinker_args = resolve_stage_factory_args(make_stage(config, "thinker"), config)

    assert isinstance(config, Qwen3OmniPipelineConfig)
    assert config.name == "qwen3-omni-mmsu"
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert "talker_ar" not in {stage.name for stage in config.stages}
    assert "code2wav" not in {stage.name for stage in config.stages}
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.8)
    assert thinker_args["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert thinker_args["server_args_overrides"]["max_running_requests"] == 4


def test_qwen_preprocessing_model_video_fps_resolves_to_factory_arg() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    merged = ConfigManager(config).merge_config(
        [("preprocessing.factory.video_fps", "2.0")]
    )

    args = resolve_stage_factory_args(make_stage(merged, "preprocessing"), merged)

    assert args["video_fps"] == 2.0


def test_h20_colocated_recipe_reserve_keeps_raw_budget_in_resolved_config() -> None:
    config = legacy_config(RECIPES_BY_FILE["qwen3_omni_colocated_h20.yaml"])

    merged = ConfigManager(config).merge_config(
        [("thinker.factory.encoder_mem_reserve", "0.05")]
    )
    plan = build_stage_placement_plan(merged)
    thinker = make_stage(merged, "thinker")
    thinker_args = resolve_stage_factory_args(thinker, merged)

    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert thinker.gpu_memory_fraction == pytest.approx(0.75)
    assert thinker_args["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert thinker_args["encoder_mem_reserve"] == pytest.approx(0.05)


def test_config_manager_rejects_unknown_stage_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stages:
  missing_stage:
    gpu_memory_fraction: 0.05
"""
    )

    # Stage topology lives in the model's config class; an unknown name in
    # the stages: mapping is refused, not created.
    with pytest.raises(Exception, match="no stage named"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_rejects_removed_stage_overrides_block(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stage_overrides:
  thinker:
    gpu: 0
"""
    )

    with pytest.raises(ValueError, match="stages: mapping"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_validates_stage_entry_values(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stages:
  image_encoder:
    gpu_memory_fraction: 1.5
"""
    )

    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        ConfigManager.from_file(str(config_path))


def test_qwen3_omni_h100_bf16_recipe_enables_speech_prefill_graph() -> None:
    config = legacy_config(RECIPES_BY_FILE["qwen3_omni_colocated_h100_bf16.yaml"])
    overrides = make_stage(config, "thinker").engine.overrides()

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert "disable_radix_cache" not in overrides
    assert overrides["cuda_graph_backend_prefill"] == "breakable"
    assert "cuda_graph_bs_prefill" not in overrides
    assert overrides["cuda_graph_max_bs_prefill"] == 2048


def test_qwen3_omni_gfx950_bf16_recipe_uses_colocated_budgets() -> None:
    config = legacy_config(RECIPES_BY_FILE["qwen3_omni_colocated_gfx950_bf16.yaml"])
    plan = build_stage_placement_plan(config)
    overrides = make_stage(config, "thinker").engine.overrides()

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated-gfx950-bf16"
    assert "prefill_attention_backend" not in overrides
    assert overrides["cuda_graph_backend_prefill"] == "disabled"
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert {
        name: make_stage(config, name).gpu_memory_fraction
        for name in (
            "image_encoder",
            "audio_encoder",
            "thinker",
            "talker_ar",
            "code2wav",
        )
    } == {
        "image_encoder": pytest.approx(0.02),
        "audio_encoder": pytest.approx(0.02),
        "thinker": pytest.approx(0.78),
        "talker_ar": pytest.approx(0.10),
        "code2wav": pytest.approx(0.02),
    }


@pytest.mark.parametrize(
    ("is_rocm", "expected_env"),
    [
        (
            True,
            {
                "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
                "SGLANG_DISABLE_AITER_GREEDY_SAMPLE": "1",
            },
        ),
        (False, {"SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0"}),
    ],
)
def test_qwen3_omni_talker_stage_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    expected_env: dict[str, str],
) -> None:
    """Talker disables fused atomic MoE finalize; ROCm also disables aiter greedy."""
    monkeypatch.setattr(qwen3_omni_config.current_platform, "is_rocm", lambda: is_rocm)

    for config_cls in (
        Qwen3OmniSpeechPipelineConfig,
        Qwen3OmniSpeechColocatedPipelineConfig,
    ):
        config = config_cls(model_path="dummy")

        assert make_stage(config, "talker_ar").env == expected_env
        assert make_stage(config, "thinker").env == {}


def test_qwen3_omni_xpu_b60_recipe_loads_and_plans() -> None:
    config = legacy_config(RECIPES_BY_FILE["qwen3_omni_speech_xpu_b60.yaml"])
    plan = build_stage_placement_plan(config)
    topology = build_compiled_process_topology(config)

    assert isinstance(config, Qwen3OmniSpeechPipelineConfig)
    assert config.name == "qwen3-omni-speech-xpu-b60"
    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "decode",
        "talker_ar",
        "code2wav",
    ]

    thinker = make_stage(config, "thinker")
    assert thinker.tp_size == 8
    assert thinker.gpu == [0, 1, 2, 3, 4, 5, 6, 7]
    assert thinker.engine.mem_fraction_static == pytest.approx(0.55)
    assert make_stage(config, "talker_ar").engine.mem_fraction_static == pytest.approx(
        0.35
    )

    assert make_stage(config, "talker_ar").gpu == 6
    assert make_stage(config, "code2wav").gpu == 7
    assert make_stage(config, "code2wav").gpu_memory_fraction == pytest.approx(0.05)
    assert plan.stages["thinker"].gpu_ids == tuple(range(8))
    assert plan.stages["talker_ar"].gpu_ids == (6,)
    assert plan.stages["code2wav"].gpu_ids == (7,)


@pytest.mark.parametrize("enabled", [False, True])
def test_talker_start_topology_reaches_bootstrap(monkeypatch, enabled):
    from sglang.srt import runtime_context

    from sglang_omni.models.qwen3_omni import bootstrap, stages

    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    config = manager.merge_config(
        {
            "talker_ar.factory.enable_talker_start_topology": enabled,
            "talker_ar.factory.enable_partial_start": True,
            "talker_ar.engine.disable_cuda_graph": True,
        }
    )
    args = resolve_stage_factory_args(make_stage(config, "talker_ar"), config)
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)
    monkeypatch.setattr(stages, "validate_generation_batch_policy", lambda **_: None)
    monkeypatch.setattr(
        bootstrap, "create_talker_scheduler", lambda *_, **kwargs: kwargs
    )
    monkeypatch.setattr(
        runtime_context,
        "get_schedule",
        lambda: SimpleNamespace(mem_fraction_static=0.5),
    )
    received = stages.create_talker_ar_executor_from_config(**args)
    assert received["enable_talker_start_topology"] is enabled
    assert received["enable_partial_start"] is True
    assert received["partial_start_min_chunks"] == 5


@pytest.mark.parametrize(
    ("variant", "config_cls"),
    [
        (None, Qwen3OmniSpeechPipelineConfig),
        ("text", Qwen3OmniPipelineConfig),
        ("speech", Qwen3OmniSpeechPipelineConfig),
        ("speech-colocated", Qwen3OmniSpeechColocatedPipelineConfig),
    ],
)
def test_from_model_path_variant_selects_from_the_module_variants(
    monkeypatch, variant, config_cls
) -> None:
    from sglang_omni.config import manager

    monkeypatch.setattr(
        manager,
        "resolve_config_cls_for_model_path",
        lambda model_path: Qwen3OmniSpeechPipelineConfig,
    )

    config = ConfigManager.from_model_path("dummy", variant=variant).config

    assert type(config) is config_cls
    assert config.model_path == "dummy"


def test_from_model_path_names_the_variants_a_model_declares(monkeypatch) -> None:
    from sglang_omni.config import manager
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    monkeypatch.setattr(
        manager,
        "resolve_config_cls_for_model_path",
        lambda model_path: Qwen3OmniSpeechPipelineConfig,
    )
    with pytest.raises(
        manager.VariantSelectionError,
        match="'speech_colocated' for 'dummy'.*variants: speech, speech-colocated, text",
    ):
        ConfigManager.from_model_path("dummy", variant="speech_colocated")
    with pytest.raises(manager.VariantSelectionError, match="Unknown variant ''"):
        ConfigManager.from_model_path("dummy", variant="")

    # A model without a Variants map declares none, rather than a default.
    monkeypatch.setattr(
        manager,
        "resolve_config_cls_for_model_path",
        lambda model_path: Qwen3TTSPipelineConfig,
    )
    with pytest.raises(manager.VariantSelectionError, match="variants: none"):
        ConfigManager.from_model_path("dummy", variant="default")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(variant=None, text_only=False, config=None), None),
        (dict(variant=None, text_only=True, config=None), "text"),
        (dict(variant=None, text_only=True, config="pipeline.yaml"), None),
        (dict(variant="text", text_only=True, config=None), "text"),
        (dict(variant="speech", text_only=False, config=None), "speech"),
    ],
)
def test_resolve_variant_selection_keeps_the_historical_combinations(
    kwargs, expected
) -> None:
    from sglang_omni.config.manager import resolve_variant_selection

    assert resolve_variant_selection(**kwargs) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(variant="", text_only=False, config=None), "cannot be empty"),
        (dict(variant="speech", text_only=False, config="p.yaml"), "config_cls"),
        (dict(variant="speech", text_only=True, config=None), "conflicts with"),
        (
            dict(
                variant="speech-colocated", text_only=False, config=None, colocate=True
            ),
            "--colocate",
        ),
    ],
)
def test_resolve_variant_selection_refuses_ambiguous_combinations(
    kwargs, message
) -> None:
    from sglang_omni.config.manager import resolve_variant_selection

    with pytest.raises(ValueError, match=message):
        resolve_variant_selection(**kwargs)
