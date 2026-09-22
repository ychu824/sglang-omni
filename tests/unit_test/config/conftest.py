# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the configuration-core unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest

from sglang_omni.config.schema import EngineStageConfig, PipelineConfig, StageConfig


class FakePipelineConfig(PipelineConfig):
    """A minimal two-stage pipeline that needs no model weights.

    ``thinker`` is declared an engine stage so the ``engine.*`` group exists
    on it and path compilation can be exercised against a per-stage type.
    The factory-kwargs hook seeds one author-owned constructor kwarg so the
    code-channel/config-channel overlay is part of the fixture too.
    """

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
    }

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, object]:
        if stage_name == "thinker":
            return {"lookahead": 4}
        return {}


def build_pipeline_config() -> PipelineConfig:
    return FakePipelineConfig(
        model_path="/models/fake-omni",
        stages=[
            StageConfig(
                name="preprocessing",
                factory_path="tests.fake:create_preprocessing",
                process="front",
                next="thinker",
                env={"OMP_NUM_THREADS": "4"},
            ),
            EngineStageConfig(
                name="thinker",
                factory_path="tests.fake:create_thinker",
                process="gen",
                terminal=True,
                factory={"max_concurrency": 4, "max_seq_len": 8192},
            ),
        ],
    )


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    return build_pipeline_config()


# Metadata the migrated launch recipes read from the Hub, as the repos served
# it when the example YAMLs were removed: only the fields architecture
# discovery looks at. Voxtral ships params.json and no config.json.
HUB_METADATA: dict[str, dict[str, dict[str, object]]] = {
    "OpenMOSS-Team/MOSS-TTS-v1.5": {
        "config.json": {
            "architectures": ["MossTTSDelayModel"],
            "model_type": "moss_tts_delay",
        }
    },
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5": {
        "config.json": {
            "architectures": ["MossTTSLocalModel"],
            "model_type": "moss_tts_local",
        }
    },
    "fishaudio/s2-pro": {"config.json": {"model_type": "fish_qwen3_omni"}},
    "mistralai/Voxtral-4B-TTS-2603": {"params.json": {"model_type": "voxtral_tts"}},
    "inclusionAI/Ming-omni-tts-16.8B-A3B": {
        "config.json": {
            "architectures": ["BailingMMNativeForConditionalGeneration"],
            "model_type": "bailingmm",
        }
    },
    "Qwen/Qwen3-ASR-1.7B": {
        "config.json": {
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "model_type": "qwen3_asr",
        }
    },
}
for _qwen3_tts_repo in (
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
):
    HUB_METADATA[_qwen3_tts_repo] = {
        "config.json": {
            "architectures": ["Qwen3TTSForConditionalGeneration"],
            "model_type": "qwen3_tts",
        }
    }
for _dots_repo in ("dots-studio/dots.tts-mf", "dots-studio/dots.tts-soar"):
    HUB_METADATA[_dots_repo] = {
        "config.json": {
            "architectures": ["DotsTTSForConditionalGeneration"],
            "model_type": "dots_tts",
        }
    }
for _qwen3_omni_repo in (
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "marksverdhei/Qwen3-Omni-30B-A3B-FP8",
    "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound",
):
    HUB_METADATA[_qwen3_omni_repo] = {
        "config.json": {
            "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
            "model_type": "qwen3_omni_moe",
        }
    }


@dataclass
class HubMetadataRequests:
    """What the fake Hub served: (repo_id, filename, revision) per download."""

    downloads: list[tuple[str, str, str | None]] = field(default_factory=list)


@pytest.fixture
def hub_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> HubMetadataRequests:
    """Serve HUB_METADATA through the same download calls discovery makes.

    AutoConfig is refused the way the Hub refuses these repos without
    trust_remote_code or a registered model type, so discovery takes its
    raw-metadata path exactly as it does against the real Hub.
    """
    from sglang_omni.config import manager
    from sglang_omni.utils import hf as hf_utils

    requests = HubMetadataRequests()

    def refuse_auto_config(*args: object, **kwargs: object) -> None:
        raise OSError("AutoConfig is not consulted in this test")

    def download(
        repo_id: str, filename: str, revision: str | None = None, **_: object
    ) -> str:
        requests.downloads.append((repo_id, filename, revision))
        served = HUB_METADATA.get(repo_id, {}).get(filename)
        if served is None:
            raise FileNotFoundError(f"{repo_id} has no {filename}")
        target = tmp_path / "hub" / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(served), encoding="utf-8")
        return str(target)

    monkeypatch.setattr(manager.AutoConfig, "from_pretrained", refuse_auto_config)
    monkeypatch.setattr(hf_utils, "hf_hub_download", download)
    return requests
