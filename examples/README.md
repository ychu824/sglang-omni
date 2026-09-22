# Examples

Run these commands from the repository root after installing `sglang-omni`.

## Unified Launcher

`run_omni.py` keeps model and topology choices in reusable presets. Use
`python examples/run_omni.py --help` to list them, then add `--help` after a
preset to inspect its options.

| Preset | Workload |
| --- | --- |
| `qwen3-text-server` | Qwen3-Omni OpenAI server with text output |
| `qwen3-speech-server` | Qwen3-Omni OpenAI server with text and audio output |
| `qwen3-speech` | One offline Qwen3-Omni speech request |
| `ming-text-server` | Ming-Omni OpenAI server with text output |
| `ming-speech-server` | Ming-Omni OpenAI server with text and audio output |
| `ming-speech` | One offline Ming-Omni speech request |
| `ming-text` | One offline Ming-Omni text request |

The older `run_qwen3_omni_*.py` and `run_ming_omni_*.py` paths remain as
compatibility wrappers around these presets.

New model examples should add a model-local module under `examples/launchers/`
that exports its preset map, then register that map in `_omni_launcher.py`.
Keep model defaults, stage mutations, and request schemas in the model-local
module; `_omni_launcher.py` owns only registry and CLI dispatch.

## Qwen3-Omni Server

Text output:

```bash
python examples/run_omni.py qwen3-text-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000 \
  --model-name qwen3-omni
```

Text and audio output:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 0 \
  --port 8000 \
  --model-name qwen3-omni
```

Qwen3-Omni FP8, one-GPU colocated H100/H20:

```bash
sgl-omni serve \
  --model-path marksverdhei/Qwen3-Omni-30B-A3B-FP8 \
  --variant speech-colocated \
  --name qwen3-omni-fp8-colocated \
  --image_encoder.gpu_memory_fraction 0.025 \
  --audio_encoder.gpu_memory_fraction 0.025 \
  --thinker.gpu_memory_fraction 0.75 \
  --talker_ar.gpu_memory_fraction 0.12 \
  --code2wav.gpu_memory_fraction 0.02 \
  --model-name qwen3-omni \
  --port 8000
```

Qwen3-Omni BF16, one-GPU colocated AMD MI355X (gfx950, ROCm). The breakable
prefill graph used by the H100 profile is not yet qualified on gfx950, so the
thinker keeps SGLang's default decode-only graph:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --variant speech-colocated \
  --name qwen3-omni-colocated-gfx950-bf16 \
  --image_encoder.gpu_memory_fraction 0.02 \
  --audio_encoder.gpu_memory_fraction 0.02 \
  --thinker.gpu_memory_fraction 0.78 \
  --talker_ar.gpu_memory_fraction 0.10 \
  --code2wav.gpu_memory_fraction 0.02 \
  --thinker.engine.cuda_graph_backend_prefill disabled \
  --model-name qwen3-omni \
  --port 8000
```

## Ming-Omni Server

Text output:

```bash
python examples/run_omni.py ming-text-server \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --port 8000 \
  --model-name ming-omni
```

Text and audio output:

```bash
python examples/run_omni.py ming-speech-server \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --port 8000 \
  --model-name ming-omni
```

Use a different `--port` if you run more than one server at the same time.

## Removed example configs

`examples/configs/` used to ship one YAML per model and per tuned hardware
profile. `--model-path` now resolves every model from its checkpoint metadata,
and the tuned profiles are documented as `sgl-omni serve` commands whose
dotted flags carry the same values (`--variant` selects a topology,
`--name` keeps the pipeline name). To keep any of them as a file again, run
the documented command through `sgl-omni config resolve ... --show config >
own.yaml` and launch with `--config own.yaml`; user config files keep working
unchanged.

Still shipped: `audar_tts_turbo.yaml` (the Turbo checkpoint ships GGUF weights
without the metadata discovery reads), the two router launcher manifests
(`qwen3_asr_router.yaml`, `qwen3_omni_router.yaml`) and the MPS/DP recipes in
`examples/mps_dp/configs/`.

| Removed file | Launch it with |
| --- | --- |
| `moss_tts.yaml` | `--model-path OpenMOSS-Team/MOSS-TTS-v1.5` |
| `moss_tts_local.yaml` | `--model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` |
| `qwen3_tts_0_6b.yaml` | `--model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| `qwen3_tts_0_6b_customvoice.yaml` | `--model-path Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| `qwen3_tts_1_7b.yaml` | `--model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `qwen3_tts_1_7b_customvoice.yaml` | `--model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `qwen3_tts_1_7b_voicedesign.yaml` | `--model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| `s2pro_tts.yaml` | `--model-path fishaudio/s2-pro` |
| `voxtral_tts.yaml` | `--model-path mistralai/Voxtral-4B-TTS-2603` |
| `dots_tts.yaml` | MeanFlow recipe in [dots.tts](../docs/cookbook/dots_tts.md#prerequisites) (pinned `dots-studio/dots.tts-mf@c28105ad…`) |
| `dots_tts_soar.yaml` | SOAR recipe in [dots.tts](../docs/cookbook/dots_tts.md#prerequisites) |
| `ming_omni_tts.yaml` | Recipe in [Ming-Omni-TTS](../docs/cookbook/ming_tts.md#server-configuration) |
| `moss_tts_24gb.yaml` | 24 GB recipe in [MOSS-TTS](../docs/cookbook/moss_tts.md) (`--variant single_process`) |
| `moss_tts_32gb.yaml` | 32 GB recipe in [MOSS-TTS](../docs/cookbook/moss_tts.md) (`--variant single_process`) |
| `moss_tts_local_non_streaming.yaml` | `--model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --vocoder_cuda_graph false` |
| `qwen3_asr_rtx4090.yaml` | RTX 4090 recipe in [Qwen3-ASR](../docs/cookbook/qwen3_asr.md) |
| `qwen3_asr_rtx5090.yaml` | RTX 5090 recipe in [Qwen3-ASR](../docs/cookbook/qwen3_asr.md) |
| `qwen3_omni_colocated_gfx950_bf16.yaml` | gfx950 command above |
| `qwen3_omni_colocated_h100_bf16.yaml` | `--variant speech-colocated --name qwen3-omni-colocated-h100-bf16` with budgets 0.02/0.02/0.78/0.10/0.02 and `--thinker.engine.cuda_graph_backend_prefill breakable --thinker.engine.cuda_graph_max_bs_prefill 2048` (`tests/test_model/conftest.py`) |
| `qwen3_omni_colocated_h100_fp8.yaml` | `--model-path marksverdhei/Qwen3-Omni-30B-A3B-FP8 --variant speech-colocated --name qwen3-omni-colocated-h100-fp8` with budgets 0.025/0.025/0.55/0.12/0.02 (`tests/test_model/conftest.py`) |
| `qwen3_omni_colocated_h20.yaml` | H20 command in [Qwen3-Omni](../docs/basic_usage/qwen3_omni.md#launch-the-server) or the cookbook generator |
| `qwen3_omni_colocated_h200.yaml` | H200 command in [Qwen3-Omni](../docs/basic_usage/qwen3_omni.md#launch-the-server) or the cookbook generator |
| `qwen3_omni_fp8_colocated.yaml` | FP8 command above / [Qwen3-Omni](../docs/basic_usage/qwen3_omni.md#single-gpu-fp8-on-h100h20) |
| `qwen3_omni_mmmu.yaml` | `--variant text --name qwen3-omni-mmmu` with budgets 0.025/0.025/0.75 |
| `qwen3_omni_mmmu_h100.yaml` | `--variant text --name qwen3-omni-mmmu-h100` with budgets 0.03/0.03/0.92 (`tests/test_model/conftest.py`) |
| `qwen3_omni_mmsu.yaml` | MMSU command in [Qwen3-Omni](../docs/basic_usage/qwen3_omni.md) |
| `qwen3_omni_speech_code2wav_replica2_ci.yaml` | `tests/test_model/test_qwen3_omni_process_replicas.py` (`REPLICA_ARGS`) |
| `qwen3_omni_speech_replica2.yaml` | Replica command in [Process Topology](../docs/basic_usage/process_topology.md#replicas-across-gpus) |
| `qwen3_omni_speech_xpu_b60.yaml` | `--name qwen3-omni-speech-xpu-b60 --image_encoder.gpu 0 --audio_encoder.gpu 0 --thinker.gpu "[0, 1, 2, 3, 4, 5, 6, 7]" --thinker.tp_size 8 --thinker.engine.mem_fraction_static 0.55 --talker_ar.gpu 6 --talker_ar.engine.mem_fraction_static 0.35 --code2wav.gpu 7 --code2wav.gpu_memory_fraction 0.05` |
| `qwen3_tts_*_npu.yaml` (4 files) | Ascend NPU recipe and table in [Qwen3-TTS](../docs/cookbook/qwen3_tts.md#ascend-npu-baseline) |

Budgets are listed in stage order `image_encoder/audio_encoder/thinker/talker_ar/code2wav`
(`--<stage>.gpu_memory_fraction`). Every removed file's exact contract is frozen
in `tests/unit_test/config/legacy_recipes.py`.
