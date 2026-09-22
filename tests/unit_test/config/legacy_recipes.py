# SPDX-License-Identifier: Apache-2.0
"""The example YAMLs removed in favour of model-path launches, frozen as recipes.

Each recipe pairs the ``sgl-omni serve`` arguments now documented for a
scenario with the contract the deleted file declared: config class, pipeline
name, model path (pins included) and every leaf the file wrote. The expected
side is copied from the files as they were, so the argv is the subject under
test and never the oracle. Model tests that used to load a file build the
same profile through legacy_config or launch the argv through the CLI.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any

from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.cli.serve import apply_tensor_parallel_engine_overrides
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

QWEN3_OMNI = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN3_OMNI_FP8 = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
DOTS_MF_PINNED = "dots-studio/dots.tts-mf@c28105adc8228143392b4e346994ff613ee48a06"


@dataclass(frozen=True)
class LegacyRecipe:
    legacy_file: str
    argv: tuple[str, ...]
    config_cls: str
    model_path: str
    name: str | None = None
    writes: tuple[tuple[str, Any], ...] = ()


def flags(*pairs: tuple[str, str]) -> tuple[str, ...]:
    return tuple(token for key, value in pairs for token in (f"--{key}", value))


def budgets(**fractions: float) -> tuple[tuple[str, float], ...]:
    return tuple(
        (f"stages.{stage}.gpu_memory_fraction", fraction)
        for stage, fraction in fractions.items()
    )


def budget_flags(**fractions: float) -> tuple[str, ...]:
    return flags(
        *(
            (f"{stage}.gpu_memory_fraction", str(fraction))
            for stage, fraction in fractions.items()
        )
    )


QWEN3_TTS_NPU_WRITES: tuple[tuple[str, Any], ...] = (
    ("stages.preprocessing.factory.max_concurrency", 1),
    ("stages.tts_engine.gpu", 0),
    ("stages.tts_engine.tp_size", 1),
    ("stages.tts_engine.factory.dtype", "bfloat16"),
    ("stages.tts_engine.factory.attn_implementation", "sdpa"),
    ("stages.tts_engine.engine.attention_backend", "ascend"),
    ("stages.tts_engine.engine.disable_cuda_graph", True),
    ("stages.tts_engine.engine.disable_overlap_schedule", True),
    ("stages.tts_engine.engine.enable_torch_compile", False),
    ("stages.tts_engine.engine.torch_compile_max_bs", 1),
    ("stages.tts_engine.engine.max_prefill_tokens", 4096),
    ("stages.tts_engine.engine.sampling_backend", "pytorch"),
    ("stages.vocoder.gpu", 0),
    ("stages.vocoder.factory.dtype", "bfloat16"),
    ("stages.vocoder.factory.attn_implementation", "sdpa"),
    ("stages.vocoder.factory.initial_max_batch_size", 1),
    ("stages.vocoder.factory.followup_max_batch_size", 1),
)

QWEN3_TTS_NPU_FLAGS = flags(
    ("preprocessing.factory.max_concurrency", "1"),
    ("tts_engine.gpu", "0"),
    ("tts_engine.tp_size", "1"),
    ("tts_engine.factory.dtype", "bfloat16"),
    ("tts_engine.factory.attn_implementation", "sdpa"),
    ("tts_engine.engine.attention_backend", "ascend"),
    ("tts_engine.engine.disable_cuda_graph", "true"),
    ("tts_engine.engine.disable_overlap_schedule", "true"),
    ("tts_engine.engine.enable_torch_compile", "false"),
    ("tts_engine.engine.torch_compile_max_bs", "1"),
    ("tts_engine.engine.max_prefill_tokens", "4096"),
    ("tts_engine.engine.sampling_backend", "pytorch"),
    ("vocoder.gpu", "0"),
    ("vocoder.factory.dtype", "bfloat16"),
    ("vocoder.factory.attn_implementation", "sdpa"),
    ("vocoder.factory.initial_max_batch_size", "1"),
    ("vocoder.factory.followup_max_batch_size", "1"),
)


def qwen3_tts_npu_recipe(
    legacy_file: str,
    model_path: str,
    *,
    max_running_requests: int,
    max_queued_requests: int,
    mem_fraction_static: float,
    vocoder_max_batch_size: int,
) -> LegacyRecipe:
    return LegacyRecipe(
        legacy_file=legacy_file,
        argv=(
            "--model-path",
            model_path,
            *QWEN3_TTS_NPU_FLAGS,
            *flags(
                ("tts_engine.engine.max_running_requests", str(max_running_requests)),
                ("tts_engine.engine.max_queued_requests", str(max_queued_requests)),
                ("tts_engine.engine.mem_fraction_static", str(mem_fraction_static)),
                ("vocoder.factory.max_batch_size", str(vocoder_max_batch_size)),
            ),
        ),
        config_cls="Qwen3TTSPipelineConfig",
        model_path=model_path,
        writes=(
            *QWEN3_TTS_NPU_WRITES,
            ("stages.tts_engine.engine.max_running_requests", max_running_requests),
            ("stages.tts_engine.engine.max_queued_requests", max_queued_requests),
            ("stages.tts_engine.engine.mem_fraction_static", mem_fraction_static),
            ("stages.vocoder.factory.max_batch_size", vocoder_max_batch_size),
        ),
    )


def moss_single_process_recipe(
    legacy_file: str, *, mem_fraction_static: float, max_total_tokens: int | None
) -> LegacyRecipe:
    engine_flags = [("tts_engine.engine.max_running_requests", "1")]
    engine_writes: list[tuple[str, Any]] = [
        ("stages.tts_engine.engine.max_running_requests", 1)
    ]
    if max_total_tokens is not None:
        engine_flags.append(
            ("tts_engine.engine.max_total_tokens", str(max_total_tokens))
        )
        engine_writes.append(
            ("stages.tts_engine.engine.max_total_tokens", max_total_tokens)
        )
    engine_flags += [
        ("tts_engine.engine.mem_fraction_static", str(mem_fraction_static)),
        ("tts_engine.engine.cuda_graph_max_bs", "1"),
    ]
    engine_writes += [
        ("stages.tts_engine.engine.mem_fraction_static", mem_fraction_static),
        ("stages.tts_engine.engine.cuda_graph_max_bs", 1),
    ]
    return LegacyRecipe(
        legacy_file=legacy_file,
        argv=(
            "--model-path",
            "OpenMOSS-Team/MOSS-TTS-v1.5",
            "--variant",
            "single_process",
            *flags(
                ("preprocessing.factory.device", "cpu"),
                ("preprocessing.factory.compute_dtype", "bfloat16"),
                ("preprocessing.factory.max_concurrency", "1"),
                *engine_flags,
                ("vocoder.factory.dtype", "bfloat16"),
                ("vocoder.factory.max_batch_size", "1"),
                ("vocoder.factory.max_batch_wait_ms", "2"),
            ),
        ),
        config_cls="MossTTSSingleProcessPipelineConfig",
        model_path="OpenMOSS-Team/MOSS-TTS-v1.5",
        writes=(
            ("stages.preprocessing.factory.device", "cpu"),
            ("stages.preprocessing.factory.compute_dtype", "bfloat16"),
            ("stages.preprocessing.factory.max_concurrency", 1),
            *engine_writes,
            ("stages.vocoder.factory.dtype", "bfloat16"),
            ("stages.vocoder.factory.max_batch_size", 1),
            ("stages.vocoder.factory.max_batch_wait_ms", 2),
        ),
    )


def dots_recipe(
    legacy_file: str,
    model_path: str,
    *,
    num_steps: int,
    max_running_requests: int,
    extra_flags: tuple[tuple[str, str], ...],
    extra_writes: tuple[tuple[str, Any], ...],
) -> LegacyRecipe:
    """dots.tts wrote the solver schedule through a shared: selector into both
    the preprocessing and latent_engine stages; the flags name each stage."""
    return LegacyRecipe(
        legacy_file=legacy_file,
        argv=(
            "--model-path",
            model_path,
            *flags(
                ("preprocessing.factory.num_steps", str(num_steps)),
                ("preprocessing.factory.max_generate_length", "500"),
                ("latent_engine.factory.num_steps", str(num_steps)),
                ("latent_engine.factory.max_generate_length", "500"),
                ("latent_engine.factory.optimize", "true"),
                ("latent_engine.engine.mem_fraction_static", "0.20"),
                (
                    "latent_engine.engine.max_running_requests",
                    str(max_running_requests),
                ),
                ("latent_engine.engine.disable_cuda_graph", "false"),
                ("latent_engine.engine.cuda_graph_max_bs", str(max_running_requests)),
                ("vocoder.factory.optimize", "true"),
                *extra_flags,
            ),
        ),
        config_cls="DotsTTSPipelineConfig",
        model_path=model_path,
        writes=(
            ("stages.preprocessing.factory.num_steps", num_steps),
            ("stages.preprocessing.factory.max_generate_length", 500),
            ("stages.latent_engine.factory.num_steps", num_steps),
            ("stages.latent_engine.factory.max_generate_length", 500),
            ("stages.latent_engine.factory.optimize", True),
            ("stages.latent_engine.engine.mem_fraction_static", 0.20),
            ("stages.latent_engine.engine.max_running_requests", max_running_requests),
            ("stages.latent_engine.engine.disable_cuda_graph", False),
            ("stages.latent_engine.engine.cuda_graph_max_bs", max_running_requests),
            ("stages.vocoder.factory.optimize", True),
            *extra_writes,
        ),
    )


def qwen3_omni_colocated_recipe(
    legacy_file: str,
    name: str,
    *,
    model_path: str = QWEN3_OMNI,
    fractions: dict[str, float],
    thinker_engine: tuple[tuple[str, str | int], ...] = (),
) -> LegacyRecipe:
    return LegacyRecipe(
        legacy_file=legacy_file,
        argv=(
            "--model-path",
            model_path,
            "--variant",
            "speech-colocated",
            "--name",
            name,
            *budget_flags(**fractions),
            *flags(
                *(
                    (f"thinker.engine.{key}", str(value))
                    for key, value in thinker_engine
                )
            ),
        ),
        config_cls="Qwen3OmniSpeechColocatedPipelineConfig",
        model_path=model_path,
        name=name,
        writes=(
            *budgets(**fractions),
            *((f"stages.thinker.engine.{key}", value) for key, value in thinker_engine),
        ),
    )


def qwen3_omni_text_recipe(
    legacy_file: str,
    name: str,
    *,
    fractions: dict[str, float],
    thinker_engine: tuple[tuple[str, int], ...] = (),
) -> LegacyRecipe:
    return LegacyRecipe(
        legacy_file=legacy_file,
        argv=(
            "--model-path",
            QWEN3_OMNI,
            "--variant",
            "text",
            "--name",
            name,
            *budget_flags(**fractions),
            *flags(
                *(
                    (f"thinker.engine.{key}", str(value))
                    for key, value in thinker_engine
                )
            ),
        ),
        config_cls="Qwen3OmniPipelineConfig",
        model_path=QWEN3_OMNI,
        name=name,
        writes=(
            *budgets(**fractions),
            *((f"stages.thinker.engine.{key}", value) for key, value in thinker_engine),
        ),
    )


COLOCATED_H20_FRACTIONS = {
    "image_encoder": 0.025,
    "audio_encoder": 0.025,
    "thinker": 0.75,
    "talker_ar": 0.12,
    "code2wav": 0.02,
}
COLOCATED_H100_BF16_FRACTIONS = {
    "image_encoder": 0.02,
    "audio_encoder": 0.02,
    "thinker": 0.78,
    "talker_ar": 0.10,
    "code2wav": 0.02,
}

# One entry per deleted example file, in the order the audit listed them.
LEGACY_RECIPES: tuple[LegacyRecipe, ...] = (
    LegacyRecipe(
        "moss_tts.yaml",
        ("--model-path", "OpenMOSS-Team/MOSS-TTS-v1.5"),
        "MossTTSPipelineConfig",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
    ),
    LegacyRecipe(
        "moss_tts_local.yaml",
        ("--model-path", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"),
        "MossTTSLocalPipelineConfig",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    ),
    LegacyRecipe(
        "qwen3_tts_0_6b.yaml",
        ("--model-path", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        "Qwen3TTSPipelineConfig",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ),
    LegacyRecipe(
        "qwen3_tts_0_6b_customvoice.yaml",
        ("--model-path", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        "Qwen3TTSPipelineConfig",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ),
    LegacyRecipe(
        "qwen3_tts_1_7b.yaml",
        ("--model-path", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        "Qwen3TTSPipelineConfig",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ),
    LegacyRecipe(
        "qwen3_tts_1_7b_customvoice.yaml",
        ("--model-path", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "Qwen3TTSPipelineConfig",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ),
    LegacyRecipe(
        "qwen3_tts_1_7b_voicedesign.yaml",
        ("--model-path", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        "Qwen3TTSPipelineConfig",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ),
    LegacyRecipe(
        "s2pro_tts.yaml",
        ("--model-path", "fishaudio/s2-pro"),
        "S2ProPipelineConfig",
        "fishaudio/s2-pro",
    ),
    LegacyRecipe(
        "voxtral_tts.yaml",
        ("--model-path", "mistralai/Voxtral-4B-TTS-2603"),
        "VoxtralTTSPipelineConfig",
        "mistralai/Voxtral-4B-TTS-2603",
    ),
    dots_recipe(
        "dots_tts.yaml",
        DOTS_MF_PINNED,
        num_steps=4,
        max_running_requests=16,
        extra_flags=(
            ("reference_encode.factory.max_concurrency", "8"),
            ("reference_encode.factory.max_batch_size", "1"),
            ("reference_encode.factory.max_batch_wait_ms", "4"),
            ("vocoder.factory.max_batch_size", "4"),
            ("vocoder.factory.max_batch_wait_ms", "2"),
        ),
        extra_writes=(
            ("stages.reference_encode.factory.max_concurrency", 8),
            ("stages.reference_encode.factory.max_batch_size", 1),
            ("stages.reference_encode.factory.max_batch_wait_ms", 4),
            ("stages.vocoder.factory.max_batch_size", 4),
            ("stages.vocoder.factory.max_batch_wait_ms", 2),
        ),
    ),
    dots_recipe(
        "dots_tts_soar.yaml",
        "dots-studio/dots.tts-soar",
        num_steps=10,
        max_running_requests=1,
        extra_flags=(),
        extra_writes=(),
    ),
    LegacyRecipe(
        "ming_omni_tts.yaml",
        (
            "--model-path",
            "inclusionAI/Ming-omni-tts-16.8B-A3B",
            "--name",
            "ming-omni-tts",
            *flags(
                ("preprocessing.process", "preprocessing"),
                ("preprocessing.factory.context_length", "8192"),
                ("preprocessing.factory.max_decode_steps_cap", "256"),
                ("preprocessing.factory.max_concurrency", "1"),
                ("reference_encode.process", "ming_tts_aux"),
                ("reference_encode.gpu", "0"),
                ("reference_encode.gpu_memory_fraction", "0.08"),
                ("reference_encode.factory.dtype", "bfloat16"),
                ("reference_encode.factory.context_length", "8192"),
                ("reference_encode.factory.max_concurrency", "1"),
                ("tts_engine.process", "tts_engine"),
                ("tts_engine.gpu", "0"),
                ("tts_engine.tp_size", "1"),
                ("tts_engine.gpu_memory_fraction", "0.72"),
                ("tts_engine.factory.dtype", "bfloat16"),
                ("tts_engine.factory.context_length", "8192"),
                ("tts_engine.engine.disable_cuda_graph", "false"),
                ("tts_engine.engine.disable_overlap_schedule", "true"),
                ("tts_engine.engine.disable_radix_cache", "true"),
                ("tts_engine.engine.enable_torch_compile", "false"),
                ("tts_engine.engine.max_prefill_tokens", "8192"),
                ("tts_engine.engine.max_running_requests", "8"),
                ("tts_engine.engine.sampling_backend", "pytorch"),
                ("tts_engine.engine.trust_remote_code", "false"),
                ("tts_engine.engine.chunked_prefill_size", "0"),
                ("tts_engine.engine.mem_fraction_static", "0.75"),
                ("tts_engine.engine.cuda_graph_bs", "[1, 2, 4, 8]"),
                ("tts_engine.engine.cuda_graph_max_bs", "8"),
                ("audio_decode.process", "ming_tts_aux"),
                ("audio_decode.gpu", "0"),
                ("audio_decode.gpu_memory_fraction", "0.12"),
                ("audio_decode.factory.dtype", "bfloat16"),
                ("audio_decode.factory.initial_chunk_patches", "2"),
                ("audio_decode.factory.steady_chunk_patches", "4"),
                ("audio_decode.factory.streaming_cuda_graph", "true"),
                ("audio_decode.factory.stream_slots", "8"),
                ("audio_decode.factory.max_batch_size", "1"),
                ("audio_decode.factory.max_batch_wait_ms", "0"),
            ),
        ),
        "MingTTSPipelineConfig",
        "inclusionAI/Ming-omni-tts-16.8B-A3B",
        name="ming-omni-tts",
        writes=(
            ("stages.preprocessing.process", "preprocessing"),
            ("stages.preprocessing.factory.context_length", 8192),
            ("stages.preprocessing.factory.max_decode_steps_cap", 256),
            ("stages.preprocessing.factory.max_concurrency", 1),
            ("stages.reference_encode.process", "ming_tts_aux"),
            ("stages.reference_encode.gpu", 0),
            ("stages.reference_encode.gpu_memory_fraction", 0.08),
            ("stages.reference_encode.factory.dtype", "bfloat16"),
            ("stages.reference_encode.factory.context_length", 8192),
            ("stages.reference_encode.factory.max_concurrency", 1),
            ("stages.tts_engine.process", "tts_engine"),
            ("stages.tts_engine.gpu", 0),
            ("stages.tts_engine.tp_size", 1),
            ("stages.tts_engine.gpu_memory_fraction", 0.72),
            ("stages.tts_engine.factory.dtype", "bfloat16"),
            ("stages.tts_engine.factory.context_length", 8192),
            ("stages.tts_engine.engine.disable_cuda_graph", False),
            ("stages.tts_engine.engine.disable_overlap_schedule", True),
            ("stages.tts_engine.engine.disable_radix_cache", True),
            ("stages.tts_engine.engine.enable_torch_compile", False),
            ("stages.tts_engine.engine.max_prefill_tokens", 8192),
            ("stages.tts_engine.engine.max_running_requests", 8),
            ("stages.tts_engine.engine.sampling_backend", "pytorch"),
            ("stages.tts_engine.engine.trust_remote_code", False),
            ("stages.tts_engine.engine.chunked_prefill_size", 0),
            ("stages.tts_engine.engine.mem_fraction_static", 0.75),
            ("stages.tts_engine.engine.cuda_graph_bs", [1, 2, 4, 8]),
            ("stages.tts_engine.engine.cuda_graph_max_bs", 8),
            ("stages.audio_decode.process", "ming_tts_aux"),
            ("stages.audio_decode.gpu", 0),
            ("stages.audio_decode.gpu_memory_fraction", 0.12),
            ("stages.audio_decode.factory.dtype", "bfloat16"),
            ("stages.audio_decode.factory.initial_chunk_patches", 2),
            ("stages.audio_decode.factory.steady_chunk_patches", 4),
            ("stages.audio_decode.factory.streaming_cuda_graph", True),
            ("stages.audio_decode.factory.stream_slots", 8),
            ("stages.audio_decode.factory.max_batch_size", 1),
            ("stages.audio_decode.factory.max_batch_wait_ms", 0),
        ),
    ),
    moss_single_process_recipe(
        "moss_tts_24gb.yaml", mem_fraction_static=0.78, max_total_tokens=8192
    ),
    moss_single_process_recipe(
        "moss_tts_32gb.yaml", mem_fraction_static=0.70, max_total_tokens=None
    ),
    LegacyRecipe(
        "moss_tts_local_non_streaming.yaml",
        (
            "--model-path",
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
            "--vocoder_cuda_graph",
            "false",
        ),
        "MossTTSLocalPipelineConfig",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        writes=(("vocoder_cuda_graph", False),),
    ),
    LegacyRecipe(
        "qwen3_asr_rtx4090.yaml",
        (
            "--model-path",
            "Qwen/Qwen3-ASR-1.7B",
            "--name",
            "qwen3-asr-rtx4090",
            *flags(
                ("asr.factory.dtype", "bfloat16"),
                ("asr.engine.max_running_requests", "16"),
                ("asr.engine.mem_fraction_static", "0.65"),
            ),
        ),
        "Qwen3ASRPipelineConfig",
        "Qwen/Qwen3-ASR-1.7B",
        name="qwen3-asr-rtx4090",
        writes=(
            ("stages.asr.factory.dtype", "bfloat16"),
            ("stages.asr.engine.max_running_requests", 16),
            ("stages.asr.engine.mem_fraction_static", 0.65),
        ),
    ),
    LegacyRecipe(
        "qwen3_asr_rtx5090.yaml",
        (
            "--model-path",
            "Qwen/Qwen3-ASR-1.7B",
            "--name",
            "qwen3-asr-rtx5090",
            *flags(
                ("asr.factory.dtype", "bfloat16"),
                ("asr.engine.max_running_requests", "16"),
                ("asr.engine.cuda_graph_max_bs", "16"),
                ("asr.engine.mem_fraction_static", "0.65"),
                ("asr.engine.enable_torch_compile", "false"),
            ),
        ),
        "Qwen3ASRPipelineConfig",
        "Qwen/Qwen3-ASR-1.7B",
        name="qwen3-asr-rtx5090",
        writes=(
            ("stages.asr.factory.dtype", "bfloat16"),
            ("stages.asr.engine.max_running_requests", 16),
            ("stages.asr.engine.cuda_graph_max_bs", 16),
            ("stages.asr.engine.mem_fraction_static", 0.65),
            ("stages.asr.engine.enable_torch_compile", False),
        ),
    ),
    qwen3_omni_colocated_recipe(
        "qwen3_omni_colocated_gfx950_bf16.yaml",
        "qwen3-omni-colocated-gfx950-bf16",
        fractions=COLOCATED_H100_BF16_FRACTIONS,
        thinker_engine=(("cuda_graph_backend_prefill", "disabled"),),
    ),
    qwen3_omni_colocated_recipe(
        "qwen3_omni_colocated_h100_bf16.yaml",
        "qwen3-omni-colocated-h100-bf16",
        fractions=COLOCATED_H100_BF16_FRACTIONS,
        thinker_engine=(
            ("cuda_graph_backend_prefill", "breakable"),
            ("cuda_graph_max_bs_prefill", 2048),
        ),
    ),
    qwen3_omni_colocated_recipe(
        "qwen3_omni_colocated_h100_fp8.yaml",
        "qwen3-omni-colocated-h100-fp8",
        model_path=QWEN3_OMNI_FP8,
        fractions={
            "image_encoder": 0.025,
            "audio_encoder": 0.025,
            "thinker": 0.55,
            "talker_ar": 0.12,
            "code2wav": 0.02,
        },
    ),
    qwen3_omni_colocated_recipe(
        "qwen3_omni_colocated_h20.yaml",
        "qwen3-omni-colocated-h20",
        fractions=COLOCATED_H20_FRACTIONS,
    ),
    qwen3_omni_colocated_recipe(
        "qwen3_omni_colocated_h200.yaml",
        "qwen3-omni-colocated-h200",
        fractions={
            "image_encoder": 0.017,
            "audio_encoder": 0.017,
            "thinker": 0.769,
            "talker_ar": 0.123,
            "code2wav": 0.014,
        },
    ),
    qwen3_omni_colocated_recipe(
        "qwen3_omni_fp8_colocated.yaml",
        "qwen3-omni-fp8-colocated",
        model_path=QWEN3_OMNI_FP8,
        fractions=COLOCATED_H20_FRACTIONS,
    ),
    qwen3_omni_text_recipe(
        "qwen3_omni_mmmu.yaml",
        "qwen3-omni-mmmu",
        fractions={"image_encoder": 0.025, "audio_encoder": 0.025, "thinker": 0.75},
    ),
    qwen3_omni_text_recipe(
        "qwen3_omni_mmmu_h100.yaml",
        "qwen3-omni-mmmu-h100",
        fractions={"image_encoder": 0.03, "audio_encoder": 0.03, "thinker": 0.92},
    ),
    qwen3_omni_text_recipe(
        "qwen3_omni_mmsu.yaml",
        "qwen3-omni-mmsu",
        fractions={"image_encoder": 0.025, "audio_encoder": 0.025, "thinker": 0.75},
        thinker_engine=(("max_running_requests", 4),),
    ),
    LegacyRecipe(
        "qwen3_omni_speech_code2wav_replica2_ci.yaml",
        (
            "--model-path",
            QWEN3_OMNI,
            "--name",
            "qwen3-omni-speech-code2wav-replica2-ci",
            *flags(
                ("processes.code2wav.num_replicas", "2"),
                ("processes.code2wav.replica_devices", "[0, 1]"),
            ),
            *budget_flags(**COLOCATED_H100_BF16_FRACTIONS),
        ),
        "Qwen3OmniSpeechPipelineConfig",
        QWEN3_OMNI,
        name="qwen3-omni-speech-code2wav-replica2-ci",
        writes=(
            ("processes.code2wav.num_replicas", 2),
            ("processes.code2wav.replica_devices", [0, 1]),
            *budgets(**COLOCATED_H100_BF16_FRACTIONS),
        ),
    ),
    LegacyRecipe(
        "qwen3_omni_speech_replica2.yaml",
        (
            "--model-path",
            QWEN3_OMNI,
            "--name",
            "qwen3-omni-speech-replica2",
            *budget_flags(talker_ar=0.123, code2wav=0.014),
            *flags(
                ("processes.talker_ar.num_replicas", "2"),
                ("processes.talker_ar.replica_devices", "[1, 2]"),
                ("processes.code2wav.num_replicas", "2"),
                ("processes.code2wav.replica_devices", "[1, 2]"),
            ),
        ),
        "Qwen3OmniSpeechPipelineConfig",
        QWEN3_OMNI,
        name="qwen3-omni-speech-replica2",
        writes=(
            *budgets(talker_ar=0.123, code2wav=0.014),
            ("processes.talker_ar.num_replicas", 2),
            ("processes.talker_ar.replica_devices", [1, 2]),
            ("processes.code2wav.num_replicas", 2),
            ("processes.code2wav.replica_devices", [1, 2]),
        ),
    ),
    LegacyRecipe(
        "qwen3_omni_speech_xpu_b60.yaml",
        (
            "--model-path",
            QWEN3_OMNI,
            "--name",
            "qwen3-omni-speech-xpu-b60",
            *flags(
                ("image_encoder.gpu", "0"),
                ("audio_encoder.gpu", "0"),
                ("thinker.gpu", "[0, 1, 2, 3, 4, 5, 6, 7]"),
                ("thinker.tp_size", "8"),
                ("thinker.engine.mem_fraction_static", "0.55"),
                ("talker_ar.gpu", "6"),
                ("talker_ar.engine.mem_fraction_static", "0.35"),
                ("code2wav.gpu", "7"),
                ("code2wav.gpu_memory_fraction", "0.05"),
            ),
        ),
        "Qwen3OmniSpeechPipelineConfig",
        QWEN3_OMNI,
        name="qwen3-omni-speech-xpu-b60",
        writes=(
            ("stages.image_encoder.gpu", 0),
            ("stages.audio_encoder.gpu", 0),
            ("stages.thinker.gpu", [0, 1, 2, 3, 4, 5, 6, 7]),
            ("stages.thinker.tp_size", 8),
            ("stages.thinker.engine.mem_fraction_static", 0.55),
            ("stages.talker_ar.gpu", 6),
            ("stages.talker_ar.engine.mem_fraction_static", 0.35),
            ("stages.code2wav.gpu", 7),
            ("stages.code2wav.gpu_memory_fraction", 0.05),
        ),
    ),
    qwen3_tts_npu_recipe(
        "qwen3_tts_0_6b_customvoice_npu.yaml",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        max_running_requests=1,
        max_queued_requests=8,
        mem_fraction_static=0.70,
        vocoder_max_batch_size=1,
    ),
    qwen3_tts_npu_recipe(
        "qwen3_tts_0_6b_npu.yaml",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        max_running_requests=16,
        max_queued_requests=16,
        mem_fraction_static=0.70,
        vocoder_max_batch_size=8,
    ),
    qwen3_tts_npu_recipe(
        "qwen3_tts_1_7b_npu.yaml",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        max_running_requests=1,
        max_queued_requests=8,
        mem_fraction_static=0.60,
        vocoder_max_batch_size=1,
    ),
    qwen3_tts_npu_recipe(
        "qwen3_tts_1_7b_voicedesign_npu.yaml",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        max_running_requests=1,
        max_queued_requests=8,
        mem_fraction_static=0.60,
        vocoder_max_batch_size=1,
    ),
)

RECIPES_BY_FILE = {recipe.legacy_file: recipe for recipe in LEGACY_RECIPES}


def legacy_config(recipe: LegacyRecipe) -> PipelineConfig:
    """The config the deleted file resolved to, rebuilt from its frozen leaves
    through the file surface's own patch shape (class + model_path baseline,
    every other value a user-file patch)."""
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(recipe.config_cls)
    baseline = config_cls(model_path=recipe.model_path)
    source = ConfigSource(SourceKind.YAML_FILE, recipe.legacy_file)
    patches = ConfigPatchSet()
    if recipe.name is not None:
        patches.add(ConfigPatch.create("name", recipe.name, source, root=config_cls))
    for path, value in recipe.writes:
        patches.add(ConfigPatch.create(path, value, source, root=config_cls))
    resolved = ConfigResolver(baseline).resolve(patches).config
    return apply_tensor_parallel_engine_overrides(resolved)


def launch_cli(monkeypatch, argv: tuple[str, ...] | list[str]) -> PipelineConfig:
    """Run ``sgl-omni serve`` through the real CLI and hand back the config it
    would have launched, instead of launching."""
    launched: list[PipelineConfig] = []
    monkeypatch.setattr(
        "sglang_omni.cli.serve.launch_server",
        lambda config, **kwargs: launched.append(config),
    )
    runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})
    result = runner.invoke(app, ["serve", *argv])
    if result.exit_code != 0:
        detail = "".join(traceback.format_exception(*result.exc_info))
        raise AssertionError(f"{result.output}\n{detail}")
    return launched.pop()
