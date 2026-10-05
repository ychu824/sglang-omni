# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model runner for the OmniScheduler AR stage."""

from __future__ import annotations

import json
import os
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
    COSYVOICE3_QUIET_TOKEN_IDS,
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
from sglang_omni.sampling.repetition_aware import (
    conservative_repetition_aware_redraw,
    repetition_aware_redraw,
)
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
# note (Yucheng Hu): experiment switch. When set to a file path, every finished
# request appends one JSON line with its tokens and per-step sampler statistics.
COSYVOICE3_TOKEN_TRACE_ENV = "SGLANG_OMNI_COSYVOICE3_TOKEN_TRACE"
# note (Yucheng Hu): experiment switch: "off" disables the RAS redraw, "conservative"
# masks the window's quiet ids and truncates the redraw (docs in omni-exp).
COSYVOICE3_RAS_ENV = "SGLANG_OMNI_COSYVOICE3_RAS"
# note (Yucheng Hu): experiment switch; "on" enables the silence governor.
COSYVOICE3_GOVERNOR_ENV = "SGLANG_OMNI_COSYVOICE3_GOVERNOR"
# Governor thresholds in consecutive quiet tokens (25 per second), with about 2x
# headroom over the phase-0 natural pauses: before speech, mid-sentence, and
# once 2.5 x the target text tokens have been spoken.
COSYVOICE3_GOVERNOR_LEAD_TOKENS = 40
COSYVOICE3_GOVERNOR_MID_TOKENS = 25
COSYVOICE3_GOVERNOR_TAIL_TOKENS = 15
COSYVOICE3_GOVERNOR_SPEECH_STARTED_TOKENS = 6
COSYVOICE3_GOVERNOR_TAIL_RATIO = 2.5
# Logit penalty per quiet token beyond the threshold, and how much one
# non-quiet token shrinks the quiet run.
COSYVOICE3_GOVERNOR_PENALTY_PER_TOKEN = 0.5
COSYVOICE3_GOVERNOR_RUN_DECAY = 4
# The conservative redraw may pick a stop token only after this many voiced
# tokens per target text token.
COSYVOICE3_REDRAW_STOP_RATIO = 3
# Columns of a trace step: stop-token mass, quiet-class mass, top probability,
# top id, sampled candidate before RAS, whether RAS replaced it, and the
# governor's logit penalty on the quiet class.
COSYVOICE3_TRACE_COLUMNS = (
    "p_stop",
    "p_quiet",
    "top_prob",
    "top_id",
    "candidate",
    "redrawn",
    "governor_penalty",
)


def cosyvoice3_target_text_tokens(data: CosyVoice3SGLangRequestData) -> int:
    """Target text token count n, recovered from min_new_tokens = 2n."""
    return max(1, data.req.sampling_params.min_new_tokens // 2)


class FunCosyVoice3ModelRunner(ModelRunner):
    """Runs Fun-CosyVoice3 AR steps and collects generated speech tokens."""

    # Off unless COSYVOICE3_TOKEN_TRACE_ENV is set; class defaults keep runners
    # built without __init__ (tests) on the untraced path.
    token_trace_path: str | None = None
    pending_trace_stats: tuple[torch.Tensor, torch.Tensor] | None = None
    pending_governor_penalties: list[float] | None = None
    ras_mode: str = "on"
    governor_enabled: bool = False
    token_class_masks: tuple[torch.Tensor, torch.Tensor] | None = None

    tp_worker: ModelWorker
    model: FunCosyVoice3SGLangModel

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
        self.token_trace_path = os.environ.get(COSYVOICE3_TOKEN_TRACE_ENV)
        self.token_traces: dict[str, dict[str, list[object]]] = {}
        self.pending_trace_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pending_governor_penalties: list[float] | None = None
        self.ras_mode = os.environ.get(COSYVOICE3_RAS_ENV, "on")
        self.governor_enabled = os.environ.get(COSYVOICE3_GOVERNOR_ENV) == "on"
        self.token_class_masks: tuple[torch.Tensor, torch.Tensor] | None = None

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

    def token_masks(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[V] bool masks of the quiet class and the stop tokens on the logits' device."""
        if self.token_class_masks is None:
            quiet = torch.zeros(logits.shape[1], dtype=torch.bool)
            quiet[list(COSYVOICE3_QUIET_TOKEN_IDS)] = True
            stop = torch.zeros(logits.shape[1], dtype=torch.bool)
            stop[VOCAB_SIZE:] = True
            self.token_class_masks = (
                quiet.to(logits.device),
                stop.to(logits.device),
            )
        else:
            pass
        return self.token_class_masks

    def process_sampling_logits(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        """Silence governor: push the quiet class down once a quiet run is too long.

        Before speech starts it also masks the stop tokens, so the model starts
        speaking instead of ending silent. The decision runs on the host from
        counters collect_tokens keeps, so normal steps launch no kernel.
        """
        if not self.governor_enabled:
            self.pending_governor_penalties = None
            return
        else:
            pass
        penalties, force_onset = [], []
        for request in requests:
            data = request.data
            text_tokens = cosyvoice3_target_text_tokens(data)
            if data.voiced_tokens < COSYVOICE3_GOVERNOR_SPEECH_STARTED_TOKENS:
                limit = COSYVOICE3_GOVERNOR_LEAD_TOKENS
            elif data.voiced_tokens < COSYVOICE3_GOVERNOR_TAIL_RATIO * text_tokens:
                limit = COSYVOICE3_GOVERNOR_MID_TOKENS
            else:
                limit = COSYVOICE3_GOVERNOR_TAIL_TOKENS
            excess = data.quiet_run - limit
            penalties.append(max(0, excess) * COSYVOICE3_GOVERNOR_PENALTY_PER_TOKEN)
            force_onset.append(
                excess > 0
                and data.voiced_tokens < COSYVOICE3_GOVERNOR_SPEECH_STARTED_TOKENS
            )
        self.pending_governor_penalties = penalties
        if not any(penalties):
            return
        else:
            pass
        logits = logits_output.next_token_logits
        control = torch.tensor([penalties, force_onset], dtype=logits.dtype).to(
            logits.device, non_blocking=True
        )
        quiet_mask, stop_mask = self.token_masks(logits)
        logits.addcmul_(
            control[0].unsqueeze(1),
            quiet_mask.to(logits.dtype).unsqueeze(0),
            value=-1.0,
        )
        logits.masked_fill_(
            (control[1] > 0).unsqueeze(1) & stop_mask.unsqueeze(0), float("-inf")
        )

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
        onto the repeated token. The experiment's "conservative" mode uses
        conservative_repetition_aware_redraw instead.
        """
        sampling_info = forward_batch.sampling_info
        trace_ids = None
        if self.token_trace_path is not None and not sampling_info.is_all_greedy:
            trace_probs = logits_output.next_token_logits.float()
            top_prob, top_id = trace_probs.max(dim=1)
            trace_floats = torch.stack(
                [
                    trace_probs[:, VOCAB_SIZE:].sum(dim=1),
                    (trace_probs * self.token_masks(trace_probs)[0]).sum(dim=1),
                    top_prob,
                ],
                dim=1,
            )
            trace_ids = torch.stack(
                [top_id, next_token_ids.long(), torch.zeros_like(top_id)], dim=1
            )
            self.pending_trace_stats = (trace_floats, trace_ids)
        else:
            self.pending_trace_stats = None
        if (
            sampling_info.is_all_greedy
            or not forward_batch.forward_mode.is_decode()
            or self.ras_mode == "off"
        ):
            return next_token_ids
        else:
            pass
        # note (Yucheng Hu): the pytorch sampler softmaxes next_token_logits in place
        # and applies top-k/top-p to a sorted copy.
        probs = logits_output.next_token_logits
        redraw_allowed = (next_token_ids < VOCAB_SIZE) & (sampling_info.top_ks != 1)
        output_ids = [request.data.req.output_ids for request in requests]
        if self.ras_mode == "conservative":
            quiet_mask, stop_mask = self.token_masks(probs)
            next_token_ids, is_repeated = conservative_repetition_aware_redraw(
                probs,
                next_token_ids,
                output_ids,
                _COSYVOICE3_RAS_WINDOW_SIZE,
                redraw_allowed,
                quiet_mask,
                stop_mask,
                [
                    request.data.voiced_tokens
                    >= COSYVOICE3_REDRAW_STOP_RATIO
                    * cosyvoice3_target_text_tokens(request.data)
                    for request in requests
                ],
                sampling_info.top_ks,
                sampling_info.top_ps,
                max(request.data.req.sampling_params.top_k for request in requests),
                sampling_info.sampling_seed,
                forward_batch.positions,
            )
        else:
            next_token_ids, is_repeated = repetition_aware_redraw(
                probs,
                next_token_ids,
                output_ids,
                _COSYVOICE3_RAS_WINDOW_SIZE,
                redraw_allowed,
                sampling_info.sampling_seed,
                forward_batch.positions,
            )
        if trace_ids is not None:
            trace_ids[:, 2] = is_repeated.long()
        else:
            pass
        if logits_output.next_token_logprobs is not None:
            emitted_logprobs = torch.log(
                probs.gather(1, next_token_ids.long().unsqueeze(1)).squeeze(1)
            )
            logits_output.next_token_logprobs = torch.where(
                is_repeated, emitted_logprobs, logits_output.next_token_logprobs
            )
        else:
            pass
        return next_token_ids

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
        trace = (
            self.token_traces.pop(str(request_id), None)
            if self.token_trace_path is not None
            else None
        )
        if trace is not None and req_data is not None:
            sampling_params = req_data.req.sampling_params
            record = {
                "request_id": str(request_id),
                "columns": COSYVOICE3_TRACE_COLUMNS,
                "min_new_tokens": sampling_params.min_new_tokens,
                "max_new_tokens": sampling_params.max_new_tokens,
                "sampling_seed": sampling_params.sampling_seed,
                **trace,
            }
            with open(self.token_trace_path, "a") as trace_file:
                trace_file.write(json.dumps(record) + "\n")
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
        token_ids_cpu = token_ids.tolist()
        trace_rows = None
        if self.pending_trace_stats is not None:
            trace_floats, trace_ids = self.pending_trace_stats
            penalties = self.pending_governor_penalties or [0.0] * len(requests)
            trace_rows = [
                floats + ids + [penalty]
                for floats, ids, penalty in zip(
                    trace_floats.tolist(), trace_ids.tolist(), penalties
                )
            ]
        else:
            pass
        self.pending_trace_stats = None
        self.pending_governor_penalties = None
        for idx, sched_req in enumerate(requests):
            token_id = int(token_ids_cpu[idx])
            if self.token_trace_path is not None:
                trace = self.token_traces.setdefault(
                    str(sched_req.request_id), {"tokens": [], "steps": []}
                )
                trace["tokens"].append(token_id)
                trace["steps"].append(trace_rows[idx] if trace_rows else None)
            else:
                pass
            if token_id >= VOCAB_SIZE:
                continue
            else:
                pass
            data = sched_req.data
            if token_id in COSYVOICE3_QUIET_TOKEN_IDS:
                data.quiet_run += 1
            else:
                data.quiet_run = max(0, data.quiet_run - COSYVOICE3_GOVERNOR_RUN_DECAY)
                data.voiced_tokens += 1
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
