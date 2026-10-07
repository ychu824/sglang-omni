# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model runner for the OmniScheduler AR stage."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from queue import Queue
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
    accept_cosyvoice3_stream_token,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    TOKEN_HOP_LEN,
    first_ar_flush_tokens,
    prompt_token_len,
)
from sglang_omni.platforms import current_platform
from sglang_omni.sampling.repetition_aware import repetition_aware_redraw
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import (
    ARRequestData,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:

    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

else:
    pass

_COSYVOICE3_RAS_WINDOW_SIZE = 10
# note (Yucheng Hu): experiment switches for the streaming time-to-first-audio study:
# "shadow" runs the redraw and emits the first draw anyway; the GPU sleep adds a
# device-side delay to every non-greedy decode step without holding the GIL; the
# switch interval changes how often Python threads in this process hand off the GIL.
COSYVOICE3_RAS_MODE_ENV = "SGLANG_OMNI_COSYVOICE3_RAS_MODE"
COSYVOICE3_DECODE_GPU_SLEEP_US_ENV = "SGLANG_OMNI_COSYVOICE3_DECODE_GPU_SLEEP_US"
COSYVOICE3_SWITCH_INTERVAL_US_ENV = "SGLANG_OMNI_COSYVOICE3_SWITCH_INTERVAL_US"
# Yields call time.sleep(0) after every non-greedy decode step's sampling, each one
# releasing the GIL without GPU work; sync waits for the decode stream with the GIL
# released; the probe records GIL wake-up latency while its flag file exists.
COSYVOICE3_DECODE_YIELDS_ENV = "SGLANG_OMNI_COSYVOICE3_DECODE_YIELDS"
# CUDA ops add N tiny in-place ops on a one-element device tensor per decode step: the
# same GIL release and kernel-launch path as the redraw's ops, without its host work.
COSYVOICE3_DECODE_CUDA_OPS_ENV = "SGLANG_OMNI_COSYVOICE3_DECODE_CUDA_OPS"
# Staged reads the step's token ids through the base runner's pinned buffer and event
# instead of .tolist() on the device tensor: a synchronous D2H into pageable memory
# blocks the vocoder thread's kernel launches for the whole GPU step.
COSYVOICE3_STAGED_TOKEN_READ_ENV = "SGLANG_OMNI_COSYVOICE3_STAGED_TOKEN_READ"
COSYVOICE3_DECODE_SYNC_ENV = "SGLANG_OMNI_COSYVOICE3_DECODE_SYNC"
COSYVOICE3_GIL_PROBE_PATH_ENV = "SGLANG_OMNI_COSYVOICE3_GIL_PROBE_PATH"
COSYVOICE3_GIL_PROBE_FLAG_ENV = "SGLANG_OMNI_COSYVOICE3_GIL_PROBE_FLAG"

logger = logging.getLogger(__name__)


def gil_probe(path: str, flag: str) -> None:
    """While `flag` exists, record how late a 1 ms sleep wakes up, i.e. the wait for the GIL."""
    with open(path, "a") as out:
        while True:
            if os.path.exists(flag):
                # note (Yucheng Hu): check the flag once per 100 samples, not per sample,
                # so the probe takes the GIL only once per wake-up.
                for _ in range(100):
                    start = time.perf_counter_ns()
                    time.sleep(0.001)
                    late = time.perf_counter_ns() - start - 1_000_000
                    out.write(f"{time.time_ns()} {late}\n")
            else:
                out.flush()
                time.sleep(0.1)


def calibrate_gpu_sleep_cycles(microseconds: int) -> int:
    """Cycles for torch.cuda._sleep to take about `microseconds` on this GPU."""
    probe_cycles = 10_000_000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(probe_cycles)  # noqa: leading-underscore
    end.record()
    end.synchronize()
    cycles = int(probe_cycles * microseconds / (start.elapsed_time(end) * 1000))
    logger.info("Fun-CosyVoice3 GPU sleep: %d us = %d cycles", microseconds, cycles)
    return cycles


class FunCosyVoice3ModelRunner(ModelRunner):
    """Runs Fun-CosyVoice3 AR steps and collects generated speech tokens."""

    tp_worker: ModelWorker
    model: FunCosyVoice3SGLangModel
    ras_mode: str = "on"
    decode_gpu_sleep_us: int = 0
    decode_gpu_sleep_cycles: int | None = None
    decode_yields: int = 0
    decode_sync: bool = False
    decode_cuda_ops: int = 0
    staged_token_read: bool = False
    decode_op_buffer: torch.Tensor | None = None

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        token_hop_len: int = TOKEN_HOP_LEN,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        hop = int(token_hop_len)
        if hop <= 0:
            raise ValueError(f"token_hop_len must be positive, got {token_hop_len}")
        else:
            pass
        self.token_hop_len = hop
        self.ar_followup_flush_tokens = hop
        self.outbox: Queue[OutgoingMessage] | None = None
        self.vocoder_target = "vocoder"
        self.cosyvoice3_recent_tokens: dict[str, list[int]] = {}
        self.ras_mode = os.environ.get(COSYVOICE3_RAS_MODE_ENV, "on")
        if self.ras_mode not in ("on", "off", "shadow"):
            raise ValueError(f"{COSYVOICE3_RAS_MODE_ENV} must be on, off or shadow")
        else:
            pass
        self.decode_gpu_sleep_us = int(
            os.environ.get(COSYVOICE3_DECODE_GPU_SLEEP_US_ENV, "0")
        )
        switch_interval_us = int(os.environ.get(COSYVOICE3_SWITCH_INTERVAL_US_ENV, "0"))
        if switch_interval_us > 0:
            sys.setswitchinterval(switch_interval_us / 1e6)
        else:
            pass
        self.decode_yields = int(os.environ.get(COSYVOICE3_DECODE_YIELDS_ENV, "0"))
        self.decode_sync = os.environ.get(COSYVOICE3_DECODE_SYNC_ENV) == "1"
        self.decode_cuda_ops = int(os.environ.get(COSYVOICE3_DECODE_CUDA_OPS_ENV, "0"))
        self.staged_token_read = os.environ.get(COSYVOICE3_STAGED_TOKEN_READ_ENV) == "1"
        probe_path = os.environ.get(COSYVOICE3_GIL_PROBE_PATH_ENV)
        if probe_path:
            threading.Thread(
                target=gil_probe,
                args=(probe_path, os.environ[COSYVOICE3_GIL_PROBE_FLAG_ENV]),
                daemon=True,
                name="cosyvoice3-gil-probe",
            ).start()
        else:
            pass
        logger.info(
            "Fun-CosyVoice3 TTFP study switches: ras_mode=%s decode_gpu_sleep_us=%d "
            "switch_interval_us=%d decode_yields=%d decode_sync=%d decode_cuda_ops=%d "
            "staged_token_read=%d gil_probe=%d",
            self.ras_mode,
            self.decode_gpu_sleep_us,
            round(sys.getswitchinterval() * 1e6),
            self.decode_yields,
            self.decode_sync,
            self.decode_cuda_ops,
            self.staged_token_read,
            bool(probe_path),
        )

    def set_stream_outbox(self, outbox: Queue[OutgoingMessage]) -> None:
        self.outbox = outbox

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self.build_prefill_input_embeds(forward_batch, requests)
        return self.forward_with_input_embeds(forward_batch, input_embeds)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_tokens(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_tokens(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        """Sample the first speech token before collecting prefill output."""
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        """Sample each speech token before collecting decode output."""
        del forward_batch, schedule_batch, requests
        return True

    def apply_repetition_penalty(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        """Leave repetition-penalty ownership to SGLang's forward snapshot.

        SGLangExecutionBridge copies SamplingBatchInfo with the
        accumulated scaling penalties before this runner samples. Applying the
        host-side incremental helper as well would penalize each token twice.
        """
        del logits_output, requests

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        if (
            logits_output.next_token_logits.device.type != "mps"
            or current_platform.is_float64_supported()
        ):
            return super().sample_next_token_ids(
                logits_output,
                forward_batch,
                schedule_batch,
                requests,
            )
        else:
            pass
        if len(requests) != 1:
            raise RuntimeError(
                "Fun-CosyVoice3 Torch MPS currently requires max_running_requests=1"
            )
        else:
            pass

        self.apply_repetition_penalty(logits_output, requests)
        self.apply_codec_suppress_tokens(logits_output, requests)
        self.install_sampling_seeds(forward_batch, requests)
        sampling_info = forward_batch.sampling_info
        installed_seeds = sampling_info.sampling_seed
        rng_context: AbstractContextManager[None] = nullcontext()
        if installed_seeds is not None:
            # Note (yexiaodong): MPS cannot represent the sampler's float64
            # probabilities, so preserve filtering while sampling from a
            # stable per-request/per-step RNG seed.
            sampling_params = requests[0].data.req.sampling_params
            row_seed = (
                int(sampling_params.sampling_seed)
                if sampling_params.sampling_seed is not None
                else 42
            )
            req = requests[0].data.req
            absolute_position = len(req.origin_input_ids) + len(req.output_ids) - 1
            step_seed = (row_seed + absolute_position * 0x9E3779B1) & SAMPLING_SEED_MASK
            device_index = logits_output.next_token_logits.device.index or 0
            rng_context = torch.random.fork_rng(
                devices=[device_index],
                device_type="mps",
            )
            sampling_info.sampling_seed = None
        else:
            pass

        wants_rollout_logprob = any(sr.data.return_logprob for sr in requests)
        if wants_rollout_logprob:
            self.enable_sampler_logprobs(forward_batch, len(requests))
        else:
            pass
        try:
            with rng_context:
                if installed_seeds is not None:
                    torch.manual_seed(step_seed)
                else:
                    pass
                # Note (yexiaodong): Scope eager mode to sampling because
                # SGLang compiles the repetition helper independently.
                with torch.compiler.set_stance("force_eager"):
                    next_token_ids = self.tp_worker.model_runner.sample(
                        logits_output,
                        forward_batch,
                    )
                next_token_ids = self.apply_ras_fallback(
                    logits_output,
                    next_token_ids,
                    sampling_info,
                    requests,
                )
        finally:
            sampling_info.sampling_seed = installed_seeds
        if wants_rollout_logprob:
            next_token_logprobs = logits_output.next_token_logprobs
            if next_token_logprobs is None:
                raise RuntimeError(
                    "Sampler did not populate next_token_logprobs when "
                    "return_logprob is enabled"
                )
            else:
                pass
            self.record_rollout_logprobs(
                next_token_logprobs,
                next_token_ids,
                requests,
            )
        else:
            pass
        return next_token_ids

    def process_sampled_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        next_token_ids: torch.Tensor,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        """Redraw a sampled speech token that repeats within the recent window.

        The redraw masks the candidate and samples the full distribution at the
        request temperature, so EOS stays reachable after top-k/top-p collapses
        onto the repeated token.
        """
        sampling_info = forward_batch.sampling_info
        if sampling_info.is_all_greedy or not forward_batch.forward_mode.is_decode():
            return next_token_ids
        else:
            pass
        if self.decode_gpu_sleep_us > 0:
            if self.decode_gpu_sleep_cycles is None:
                self.decode_gpu_sleep_cycles = calibrate_gpu_sleep_cycles(
                    self.decode_gpu_sleep_us
                )
            else:
                pass
            torch.cuda._sleep(self.decode_gpu_sleep_cycles)  # noqa: leading-underscore
        else:
            pass
        for _ in range(self.decode_yields):
            time.sleep(0)
        if self.decode_cuda_ops > 0:
            if self.decode_op_buffer is None:
                self.decode_op_buffer = torch.zeros(1, device=next_token_ids.device)
            else:
                pass
            for _ in range(self.decode_cuda_ops):
                self.decode_op_buffer.add_(0)
        else:
            pass
        if self.decode_sync:
            torch.cuda.current_stream().synchronize()
        else:
            pass
        if self.ras_mode == "off":
            return next_token_ids
        else:
            pass
        first_draw = next_token_ids
        # note (Yucheng Hu): the pytorch sampler softmaxes next_token_logits in place
        # and applies top-k/top-p to a sorted copy.
        probs = logits_output.next_token_logits
        next_token_ids, is_repeated = repetition_aware_redraw(
            probs,
            next_token_ids,
            [request.data.req.output_ids for request in requests],
            _COSYVOICE3_RAS_WINDOW_SIZE,
            (next_token_ids < VOCAB_SIZE) & (sampling_info.top_ks != 1),
            sampling_info.sampling_seed,
            forward_batch.positions,
        )
        if logits_output.next_token_logprobs is not None:
            emitted_logprobs = torch.log(
                probs.gather(1, next_token_ids.long().unsqueeze(1)).squeeze(1)
            )
            logits_output.next_token_logprobs = torch.where(
                is_repeated, emitted_logprobs, logits_output.next_token_logprobs
            )
        else:
            pass
        return first_draw if self.ras_mode == "shadow" else next_token_ids

    def apply_ras_fallback(
        self,
        logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        """Apply CosyVoice3's repetition-aware redraw on Torch/MPS.

        The upstream CosyVoice sampler draws from top-k/top-p first and, when
        that candidate appears in the recent ten speech tokens, redraws once
        from the full distribution with the candidate masked. SGLang's MPS
        sampler has already applied all request constraints and materialized
        probabilities by this point, so the second draw can stay on MPS
        without invoking the float64 seeded sampler.
        """
        request = requests[0]
        request_id = str(request.request_id)
        if (
            self.cosyvoice3_recent_tokens
            and request_id not in self.cosyvoice3_recent_tokens
        ):
            # Note (yexiaodong): MPS sampling is single-request; clear state
            # here because aborts can bypass the normal finish hook.
            self.cosyvoice3_recent_tokens.clear()
        else:
            pass
        recent = self.cosyvoice3_recent_tokens.setdefault(request_id, [])
        token_ids = next_token_ids.reshape(-1)
        token_id = int(token_ids[0].item())

        if not sampling_info.is_all_greedy and token_id < VOCAB_SIZE:
            if token_id in recent[-_COSYVOICE3_RAS_WINDOW_SIZE:]:
                probs = logits_output.next_token_logits
                if probs is None or probs.ndim != 2:
                    raise RuntimeError(
                        "CosyVoice3 Torch/MPS RAS requires sampled probabilities"
                    )
                else:
                    pass
                fallback_probs = probs[0].to(dtype=torch.float32).clone()
                fallback_probs[token_id] = 0.0
                fallback_probs.clamp_(min=0.0)
                if float(fallback_probs.sum().item()) <= 0.0:
                    # Note (yexiaodong): A collapsed top-k/top-p row has no
                    # valid redraw distribution, so keep the sampled token.
                    fallback = None
                else:
                    fallback = torch.multinomial(
                        fallback_probs.unsqueeze(0), num_samples=1
                    ).reshape(-1)
                if fallback is not None:
                    next_token_ids = next_token_ids.clone()
                    next_token_ids[0] = fallback.to(dtype=next_token_ids.dtype)
                    token_id = int(fallback[0].item())
                else:
                    pass
                if (
                    fallback is not None
                    and logits_output.next_token_logprobs is not None
                ):
                    # Note (yexiaodong): Update rollout logprobs after redraw
                    # so they describe the emitted token, not the rejected one.
                    fallback_logprobs = torch.log(
                        probs.clamp_min(torch.finfo(probs.dtype).tiny)
                    )
                    logits_output.next_token_logprobs = fallback_logprobs.gather(
                        1, next_token_ids.long().view(-1, 1)
                    ).view(-1)
                else:
                    pass
            else:
                pass
        else:
            pass

        if 0 <= token_id < VOCAB_SIZE:
            recent.append(token_id)
            del recent[:-_COSYVOICE3_RAS_WINDOW_SIZE]
        else:
            pass
        return next_token_ids

    def on_request_finished(
        self, request_id: str, req_data: ARRequestData | None
    ) -> None:
        if req_data is not None:
            self.flush_code_chunks(request_id, req_data, force=True)
        else:
            pass
        recent_tokens = getattr(self, "cosyvoice3_recent_tokens", None)
        if recent_tokens is not None:
            recent_tokens.pop(str(request_id), None)
        else:
            pass

    def collect_tokens(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        if result.next_token_ids is None:
            return
        else:
            pass
        token_ids = result.next_token_ids
        if token_ids.ndim != 1:
            token_ids = token_ids.reshape(-1)
        else:
            pass
        # note (guozhihao-224): one batched D2H instead of per-request .item() syncs.
        if self.staged_token_read:
            self.stage_token_ids(result, token_ids)
            token_ids_cpu = self.resolve_host_token_ids(result).tolist()
        else:
            token_ids_cpu = token_ids.tolist()
        for idx, sched_req in enumerate(requests):
            token_id = int(token_ids_cpu[idx])
            if token_id >= VOCAB_SIZE:
                continue
            else:
                pass
            token = torch.tensor([token_id], dtype=torch.long)
            sched_req.data.output_codes.append(token)
            self.queue_or_emit_code_chunk(sched_req, token)

    def queue_or_emit_code_chunk(
        self,
        sched_req: SchedulerRequest,
        token: torch.Tensor,
    ) -> None:
        data: CosyVoice3SGLangRequestData = sched_req.data
        if self.outbox is None or data.stream_metadata is None:
            return
        else:
            pass
        if not accept_cosyvoice3_stream_token(data, token):
            return
        else:
            pass
        data.stream_code_buffer.append(token)
        data.stream_code_seen += 1
        if int(data.stream_code_next_flush) <= 0:
            # note (guozhihao-224): first flush is hop+lookahead. Prompt hop
            # alignment is applied on Flow prompt tensors, not extra AR tokens.
            data.stream_code_next_flush = first_ar_flush_tokens(
                prompt_token_len(data.flow_prompt_speech_token),
                hop_len=self.token_hop_len,
            )
        else:
            pass
        if data.stream_code_seen >= data.stream_code_next_flush:
            self.flush_code_chunks(sched_req.request_id, data, force=False)
        else:
            pass

    def flush_code_chunks(
        self,
        request_id: str,
        data: CosyVoice3SGLangRequestData,
        *,
        force: bool,
    ) -> None:
        pending = data.stream_code_buffer
        if not pending:
            return
        else:
            pass
        payload = pending[0] if len(pending) == 1 else torch.cat(pending, dim=0)
        pending.clear()
        if not force:
            data.stream_code_next_flush = (
                int(data.stream_code_seen) + self.ar_followup_flush_tokens
            )
        else:
            pass
        self.emit_code_chunk(request_id, data, payload)

    def emit_code_chunk(
        self,
        request_id: str,
        data: CosyVoice3SGLangRequestData,
        codes: torch.Tensor,
    ) -> None:
        if self.outbox is None:
            return
        else:
            pass
        metadata = data.stream_metadata
        if metadata is None:
            return
        else:
            pass
        chunk_metadata = dict(metadata)
        # note (guozhihao-224): first chunk carries Flow prompt tensors so
        # vocoder can start before the AR payload arrives.
        if not data.stream_prompt_sent:
            chunk_metadata["flow_prompt_speech_token"] = data.flow_prompt_speech_token
            chunk_metadata["flow_prompt_speech_feat"] = data.flow_prompt_speech_feat
            chunk_metadata["flow_embedding"] = data.flow_embedding
            data.stream_prompt_sent = True
        else:
            pass
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=self.vocoder_target,
                data=codes,
                metadata=chunk_metadata,
            )
        )

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data: CosyVoice3SGLangRequestData = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prompt_input_embeds
            if prompt_embeds is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 prefill requires prompt_input_embeds"
                )
            else:
                pass
            pieces.append(prompt_embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def forward_with_input_embeds(
        self,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        else:
            pass
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )


class FunCosyVoice3MlxSchedulerModelRunner(MlxSchedulerModelRunner):
    """MLX scheduler bridge that records generated speech-code tokens.

    MlxSchedulerModelRunner finalizes lazy launches directly through its
    shared _finalize path, so the Torch runner's phase hooks are not used.
    post_process_outputs is the common point for both sync and lookahead
    MLX execution and runs after the worker has materialized the sampled ids.
    """

    def __init__(
        self,
        tp_worker: MlxTpModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        token_hop_len: int = TOKEN_HOP_LEN,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        hop = int(token_hop_len)
        if hop <= 0:
            raise ValueError(f"token_hop_len must be positive, got {token_hop_len}")
        else:
            pass
        self.token_hop_len = hop

    def set_stream_outbox(self, outbox: Queue[OutgoingMessage]) -> None:
        self.outbox = outbox
        self.vocoder_target = "vocoder"

    def on_request_finished(
        self, request_id: str, req_data: ARRequestData | None
    ) -> None:
        if req_data is not None:
            self.flush_code_chunks(request_id, req_data, force=True)
        else:
            pass

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        if len(batch.reqs) != 1 or batch.has_grammar:
            return False
        else:
            pass
        previous = self.last_mlx_pending
        if previous is not None:
            previous_ids = [req.rid for req in previous.reqs]
            current_ids = [req.rid for req in batch.reqs]
            if previous.launch.mode != "decode" or previous_ids != current_ids:
                return False
            else:
                pass
        else:
            pass
        req = batch.reqs[0]
        sampling_params = req.sampling_params
        return (
            sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and req.custom_logit_processor is None
        )

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del outputs
        token_ids = result.next_token_ids
        if token_ids is None:
            return
        else:
            pass
        token_ids = token_ids.reshape(-1).tolist()
        requests = scheduler_output.requests
        if len(token_ids) != len(requests):
            raise RuntimeError(
                "Fun-CosyVoice3 MLX sampled-token row count does not match "
                f"the scheduler batch ({len(token_ids)} != {len(requests)})"
            )
        else:
            pass
        for sched_req, token_id in zip(requests, token_ids, strict=True):
            if sched_req.request_id in self.resolve_skip_rids:
                continue
            else:
                pass
            token_id = int(token_id)
            if 0 <= token_id < VOCAB_SIZE:
                token = torch.tensor([token_id], dtype=torch.long)
                sched_req.data.output_codes.append(token)
                self.queue_or_emit_code_chunk(sched_req, token)
            else:
                pass

    def queue_or_emit_code_chunk(
        self, sched_req: SchedulerRequest, token: torch.Tensor
    ) -> None:
        data: CosyVoice3SGLangRequestData = sched_req.data
        if getattr(self, "outbox", None) is None or data.stream_metadata is None:
            return
        else:
            pass
        if not accept_cosyvoice3_stream_token(data, token):
            return
        else:
            pass
        data.stream_code_buffer.append(token)
        data.stream_code_seen += 1
        if int(data.stream_code_next_flush) <= 0:
            data.stream_code_next_flush = first_ar_flush_tokens(
                prompt_token_len(data.flow_prompt_speech_token),
                hop_len=getattr(self, "token_hop_len", TOKEN_HOP_LEN),
            )
        else:
            pass
        if data.stream_code_seen >= data.stream_code_next_flush:
            self.flush_code_chunks(sched_req.request_id, data, force=False)
        else:
            pass

    def flush_code_chunks(
        self, request_id: str, data: CosyVoice3SGLangRequestData, *, force: bool
    ) -> None:
        pending = data.stream_code_buffer
        if not pending:
            return
        else:
            pass
        payload = pending[0] if len(pending) == 1 else torch.cat(pending, dim=0)
        pending.clear()
        if not force:
            data.stream_code_next_flush = int(data.stream_code_seen) + getattr(
                self, "token_hop_len", TOKEN_HOP_LEN
            )
        else:
            pass
        self.emit_code_chunk(request_id, data, payload)

    def emit_code_chunk(
        self, request_id: str, data: CosyVoice3SGLangRequestData, codes: torch.Tensor
    ) -> None:
        outbox = getattr(self, "outbox", None)
        if outbox is None or data.stream_metadata is None:
            return
        else:
            pass
        metadata = dict(data.stream_metadata)
        if not data.stream_prompt_sent:
            metadata["flow_prompt_speech_token"] = data.flow_prompt_speech_token
            metadata["flow_prompt_speech_feat"] = data.flow_prompt_speech_feat
            metadata["flow_embedding"] = data.flow_embedding
            data.stream_prompt_sent = True
        else:
            pass
        outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=getattr(self, "vocoder_target", "vocoder"),
                data=codes,
                metadata=metadata,
            )
        )
