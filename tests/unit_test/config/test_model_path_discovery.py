# SPDX-License-Identifier: Apache-2.0
"""Model-path discovery for the launches that used to need an example YAML.

A plain ``--model-path`` launch is only equivalent to the deleted
``config_cls`` + ``model_path`` file when architecture discovery lands on the
same config class, from the metadata the checkpoint actually ships. These
cases replay that metadata for Hub ids, local directories and renamed local
directories, and confirm the one checkpoint that ships none (Audar) is still
refused rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from tests.unit_test.config.conftest import HUB_METADATA

DEFAULT_LAUNCHES = [
    ("OpenMOSS-Team/MOSS-TTS-v1.5", "MossTTSPipelineConfig"),
    ("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", "MossTTSLocalPipelineConfig"),
    ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen3TTSPipelineConfig"),
    ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3TTSPipelineConfig"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3TTSPipelineConfig"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3TTSPipelineConfig"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen3TTSPipelineConfig"),
    ("fishaudio/s2-pro", "S2ProPipelineConfig"),
    ("mistralai/Voxtral-4B-TTS-2603", "VoxtralTTSPipelineConfig"),
]

TUNED_LAUNCHES = [
    ("dots-studio/dots.tts-mf", "DotsTTSPipelineConfig"),
    ("dots-studio/dots.tts-soar", "DotsTTSPipelineConfig"),
    ("inclusionAI/Ming-omni-tts-16.8B-A3B", "MingTTSPipelineConfig"),
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3ASRPipelineConfig"),
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen3OmniSpeechPipelineConfig"),
    ("marksverdhei/Qwen3-Omni-30B-A3B-FP8", "Qwen3OmniSpeechPipelineConfig"),
]


def write_checkpoint_metadata(directory: Path, repo_id: str) -> Path:
    directory.mkdir(parents=True)
    for filename, content in HUB_METADATA[repo_id].items():
        (directory / filename).write_text(json.dumps(content), encoding="utf-8")
    return directory


@pytest.mark.parametrize(("repo_id", "config_cls"), DEFAULT_LAUNCHES + TUNED_LAUNCHES)
def test_hub_metadata_selects_the_class_the_example_named(
    hub_metadata, repo_id: str, config_cls: str
) -> None:
    assert resolve_config_cls_for_model_path(repo_id).__name__ == config_cls
    assert all(revision is None for _, _, revision in hub_metadata.downloads)


@pytest.mark.parametrize(("repo_id", "config_cls"), DEFAULT_LAUNCHES + TUNED_LAUNCHES)
def test_a_renamed_local_checkpoint_resolves_from_its_metadata_alone(
    hub_metadata, tmp_path: Path, repo_id: str, config_cls: str
) -> None:
    """Directory names carry no information: Base, CustomVoice, VoiceDesign and
    the Qwen3-Omni FP8 repack all resolve from what the files say."""
    local = write_checkpoint_metadata(tmp_path / "ckpt", repo_id)

    assert resolve_config_cls_for_model_path(str(local)).__name__ == config_cls
    assert hub_metadata.downloads == []


def test_a_pinned_revision_reads_metadata_at_that_revision(hub_metadata) -> None:
    pinned = "dots-studio/dots.tts-mf@c28105adc8228143392b4e346994ff613ee48a06"

    assert resolve_config_cls_for_model_path(pinned).__name__ == "DotsTTSPipelineConfig"
    assert hub_metadata.downloads == [
        (
            "dots-studio/dots.tts-mf",
            "config.json",
            "c28105adc8228143392b4e346994ff613ee48a06",
        )
    ]


def test_voxtral_resolves_from_params_json_without_a_config_json(hub_metadata) -> None:
    resolve_config_cls_for_model_path("mistralai/Voxtral-4B-TTS-2603")

    assert (
        "mistralai/Voxtral-4B-TTS-2603",
        "config.json",
        None,
    ) in hub_metadata.downloads
    assert (
        "mistralai/Voxtral-4B-TTS-2603",
        "params.json",
        None,
    ) in hub_metadata.downloads


def test_a_checkpoint_without_metadata_is_refused_not_guessed(hub_metadata) -> None:
    """Audar's Turbo checkpoint ships GGUF weights and no config: it keeps its
    explicit config file rather than being matched on its repo name."""
    with pytest.raises(ValueError, match="Could not resolve model architecture"):
        resolve_config_cls_for_model_path("audarai/Audar-TTS-V1-Turbo")
