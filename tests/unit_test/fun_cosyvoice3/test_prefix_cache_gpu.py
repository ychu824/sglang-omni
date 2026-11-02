# SPDX-License-Identifier: Apache-2.0
"""Hops over a cached prefix reproduce the whole-history causal solve, and
graph replayed hops and whole-history solves their eager solves."""

from __future__ import annotations

import copy
import gc
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.causal_conv import FusedConvPositionEmbedding
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    gather_rows,
    pack_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    PrefixRowAttention,
    PrefixStepLayout,
    compile_forward_prefix,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import build_prefix_pool
from sglang_omni.models.fun_cosyvoice3.whole_history_cuda_graph import (
    WholeHistoryCudaGraphRunner,
)
from sglang_omni.platforms import current_platform

pytestmark = pytest.mark.accelerator

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")

CHUNK = 50
CHANNELS = 80
HEADS, HEAD_DIM, LAYERS = 4, 64, 3
ERROR_RATIO_OVER_BASELINE = 1.1
# about three times the largest relative error against float64 measured on compute capability 12 (6.2e-3, RTX PRO 4500)
MAX_RELATIVE_ERROR = 0.02

GemmMode = Literal["bitwise", "float64_gemms", "tolerance"]


def float64_linear(
    activations: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """F.linear as a float64 product rounded once to the weight's dtype, which is the same at every row count."""
    out = activations.to(weight.dtype).double() @ weight.double().T
    if bias is not None:
        out = out + bias.double()
    else:
        pass
    return out.to(weight.dtype)


@pytest.fixture(params=("bitwise", "float64_gemms", "tolerance"))
def gemm_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> GemmMode:
    """bitwise compares solves over different row counts bit for bit. On compute capability 12, float64_gemms keeps that with float64 GEMMs and tolerance bounds each path's error against a float64 solve."""
    # note (Yucheng Hu): cuBLAS's bf16 GEMMs round the same rows differently at different row counts on compute capability 12.
    if (request.param == "bitwise") == (torch.cuda.get_device_capability()[0] == 12):
        pytest.skip(f"{request.param} does not apply on this GPU")
    elif request.param == "float64_gemms":
        monkeypatch.setattr(F, "linear", float64_linear)
    elif request.param == "tolerance":
        # note (Yucheng Hu): the FA4 graphs are captured with float32 reductions, so every solve here runs under that setting and no ratio measures the flag.
        monkeypatch.setattr(
            torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
        )
    else:
        pass
    return request.param


def make_estimator() -> PackedDiT:
    torch.manual_seed(0)
    dit = (
        cosyvoice_dit.DiT(
            dim=HEADS * HEAD_DIM,
            depth=LAYERS,
            heads=HEADS,
            dim_head=HEAD_DIM,
            ff_mult=2,
            mel_dim=CHANNELS,
            mu_dim=CHANNELS,
            spk_dim=CHANNELS,
            out_channels=CHANNELS,
            static_chunk_size=CHUNK,
            num_decoding_left_chunks=-1,
        )
        .cuda()
        .eval()
    )
    # Note (Jiaxin Deng): a visible conv bias makes zero context differ from
    # zero padding after the first conv, which the cached path must reproduce.
    with torch.no_grad():
        for conv in (
            dit.input_embed.conv_pos_embed.conv1[0],
            dit.input_embed.conv_pos_embed.conv2[0],
        ):
            conv.bias.fill_(0.5)
        for block in dit.transformer_blocks:
            block.attn_norm.linear.bias.fill_(0.5)
        dit.norm_out.linear.bias.fill_(0.5)
    for module in dit.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    dit.input_embed.conv_pos_embed = FusedConvPositionEmbedding(
        dit.input_embed.conv_pos_embed
    )
    estimator = PackedDiT(dit, device="cuda")
    if not estimator.is_ragged:
        pytest.skip("requires ragged attention")
    else:
        pass
    return estimator


def make_float64_estimator(estimator: PackedDiT, time_span: torch.Tensor) -> PackedDiT:
    """The same weights in float64, whose solves the bf16 ones are judged against."""
    float64_estimator = PackedDiT(
        copy.deepcopy(estimator.dit).double(), device=time_span.device
    )
    with torch.inference_mode(), torch.autocast(time_span.device.type, enabled=False):
        float64_estimator.build_modulations(time_span.double(), torch.float64)
    return float64_estimator


def solve_in_float64(
    float64_estimator: PackedDiT,
    lengths: tuple[int, ...],
    *,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    mel_conditioning: torch.Tensor,
    streaming: bool,
) -> torch.Tensor:
    """The Euler solve of the same inputs as the bf16 solves, in float64."""
    twin_rows = pack_rows(lengths * 2, noise.device)
    with torch.autocast(noise.device.type, enabled=False):
        return solve_flow_euler_packed(
            float64_estimator,
            noise.double(),
            time_span.double(),
            mu.double(),
            speaker_embeddings.double(),
            mel_conditioning.double(),
            twin_rows,
            float64_estimator.row_attention(
                twin_rows, streaming=streaming, dtype=torch.float64
            ),
            float64_estimator.rope_angles(twin_rows.width),
            cfg_rate=0.7,
        )


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.float() - expected)
        / torch.linalg.vector_norm(expected)
    )


def assert_as_accurate_as_baseline(
    actual: torch.Tensor, baseline: torch.Tensor, truth: torch.Tensor, label: str
) -> None:
    """actual is at most ERROR_RATIO_OVER_BASELINE times as far from truth as baseline.
    Both stay under MAX_RELATIVE_ERROR, which an error they share would slip under the ratio.
    """
    actual_error = relative_error(actual, truth)
    baseline_error = relative_error(baseline, truth)
    assert max(actual_error, baseline_error) <= MAX_RELATIVE_ERROR, (
        label,
        actual_error,
        baseline_error,
    )
    assert actual_error <= ERROR_RATIO_OVER_BASELINE * baseline_error, (
        label,
        actual_error,
        baseline_error,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("totals", [(100, 200, 400), (70, 130, 260)])
def test_prefix_hops_match_the_whole_history_hops(
    totals: tuple[int, ...], gemm_mode: GemmMode
) -> None:
    """Every hop equals the whole-history hop, also hops that end inside a chunk."""
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(1)
    rows = 2
    noise = torch.randn(rows, CHANNELS, 400, device=device, dtype=dtype)
    mu = torch.randn(rows, CHANNELS, 400, device=device, dtype=dtype)
    cond = torch.zeros_like(mu)
    cond[:, :, :50] = torch.randn(rows, CHANNELS, 50, device=device, dtype=dtype)
    spks = torch.randn(rows, CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    caches = [(PrefixCacheRow(), PrefixCacheRow()) for _ in range(rows)]
    if gemm_mode == "tolerance":
        float64_estimator = make_float64_estimator(estimator, time_span)
    else:
        pass
    previous = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        for total in totals:
            packed = pack_rows([total] * rows, device)
            twin_rows = pack_rows([total] * 2 * rows, device)
            reference = solve_flow_euler_packed(
                estimator,
                gather_rows(noise[:, :, :total].transpose(1, 2), packed),
                time_span,
                gather_rows(mu[:, :, :total].transpose(1, 2), packed),
                spks,
                gather_rows(cond[:, :, :total].transpose(1, 2), packed),
                twin_rows,
                estimator.row_attention(twin_rows, streaming=True, dtype=dtype),
                estimator.rope_angles(total),
                cfg_rate=0.7,
            )
            for pair in caches:
                assert grow_rows(pool, list(pair), [total, total])
            start = caches[0][0].committed_frames
            new = [total - start] * rows
            take = lambda x: torch.cat(
                [x[row, :, start:total].transpose(0, 1) for row in range(rows)]
            ).unsqueeze(0)
            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                take(noise),
                time_span,
                take(mu),
                spks,
                take(cond),
                new,
                caches,
                cfg_rate=0.7,
            )
            if gemm_mode == "tolerance":
                truth = solve_in_float64(
                    float64_estimator,
                    (total,) * rows,
                    noise=gather_rows(noise[:, :, :total].transpose(1, 2), packed),
                    time_span=time_span,
                    mu=gather_rows(mu[:, :, :total].transpose(1, 2), packed),
                    speaker_embeddings=spks,
                    mel_conditioning=gather_rows(
                        cond[:, :, :total].transpose(1, 2), packed
                    ),
                    streaming=True,
                )
            else:
                pass
            for row in range(rows):
                expected = reference[0, row * total + previous : (row + 1) * total]
                actual = cached[
                    0,
                    row * (total - start)
                    + previous
                    - start : (row + 1) * (total - start),
                ]
                if gemm_mode == "tolerance":
                    assert_as_accurate_as_baseline(
                        actual,
                        expected,
                        truth[0, row * total + previous : (row + 1) * total],
                        f"{total=} {row=}",
                    )
                else:
                    assert torch.equal(actual, expected), (total, row)
            previous = total
    used = 32 - len(pool.free_blocks)
    assert used == rows * 2 * ((totals[-1] + BLOCK_FRAMES - 1) // BLOCK_FRAMES)
    for pair in caches:
        release_rows(pool, list(pair))
    assert len(pool.free_blocks) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_and_whole_history_hops_launch_without_a_host_wait() -> None:
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=16 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(1)
    lengths = [120, 70]
    noise, mu, cond = (
        torch.randn(1, sum(lengths), CHANNELS, device=device, dtype=dtype)
        for _ in range(3)
    )
    spks = torch.randn(len(lengths), CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    caches = [(PrefixCacheRow(), PrefixCacheRow()) for _ in lengths]
    for pair, length in zip(caches, lengths, strict=True):
        assert grow_rows(pool, list(pair), [length, length])
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        angles = estimator.rope_angles(max(lengths))
        torch.cuda.synchronize()
        sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            twin_rows = pack_rows(lengths * 2, device)
            whole = solve_flow_euler_packed(
                estimator,
                noise,
                time_span,
                mu,
                spks,
                cond,
                twin_rows,
                estimator.row_attention(twin_rows, streaming=True, dtype=dtype),
                angles,
                cfg_rate=0.7,
            )
            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                noise,
                time_span,
                mu,
                spks,
                cond,
                lengths,
                caches,
                cfg_rate=0.7,
            )
        finally:
            torch.cuda.set_sync_debug_mode(sync_debug_mode)
    assert torch.equal(cached, whole)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_prefix_hops_follow_each_row_across_batches() -> None:
    """Compiled hops over rows with their own prompts, prefixes and lengths,
    one joining fresh beside a cached one and the order flipped on a later hop,
    stay as close to each row's float32 whole-history solve as its bf16 eager
    solve is."""
    estimator = make_estimator()
    reference_estimator = PackedDiT(copy.deepcopy(estimator.dit).float(), device="cuda")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    pool.forward = compile_forward_prefix()
    torch.manual_seed(2)
    streams = {}
    for name, prompt_frames in (("a", 50), ("b", 80)):
        mel_conditioning = torch.zeros(CHANNELS, 400, device=device, dtype=dtype)
        mel_conditioning[:, :prompt_frames] = torch.randn(
            CHANNELS, prompt_frames, device=device, dtype=dtype
        )
        streams[name] = {
            "noise": torch.randn(CHANNELS, 400, device=device, dtype=dtype),
            "mu": torch.randn(CHANNELS, 400, device=device, dtype=dtype),
            "mel_conditioning": mel_conditioning,
            "speaker_embeddings": torch.randn(CHANNELS, device=device, dtype=dtype),
        }
    caches = {name: (PrefixCacheRow(), PrefixCacheRow()) for name in streams}
    emitted = {name: 0 for name in streams}
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    hops = [[("a", 70)], [("b", 130), ("a", 160)], [("a", 260), ("b", 210)]]

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        with torch.autocast("cuda", enabled=False):
            reference_estimator.build_modulations(time_span.float(), torch.float32)
        for hop in hops:
            names = [name for name, _ in hop]
            totals = [total for _, total in hop]
            for name, total in hop:
                assert grow_rows(pool, list(caches[name]), [total, total])
            starts = [caches[name][0].committed_frames for name in names]

            def take(key: str) -> torch.Tensor:
                return torch.cat(
                    [
                        streams[name][key][:, start:total].transpose(0, 1)
                        for name, start, total in zip(names, starts, totals)
                    ]
                ).unsqueeze(0)

            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                take("noise"),
                time_span,
                take("mu"),
                torch.stack([streams[name]["speaker_embeddings"] for name in names]),
                take("mel_conditioning"),
                [total - start for start, total in zip(starts, totals)],
                [caches[name] for name in names],
                cfg_rate=0.7,
            )
            offset = 0
            for name, start, total in zip(names, starts, totals):
                stream = streams[name]
                packed = pack_rows([total], device)
                twin_rows = pack_rows([total, total], device)

                def whole_history(
                    model: PackedDiT, value_dtype: torch.dtype
                ) -> torch.Tensor:
                    def rows_of(key: str) -> torch.Tensor:
                        return gather_rows(
                            stream[key][None, :, :total].transpose(1, 2), packed
                        ).to(value_dtype)

                    return solve_flow_euler_packed(
                        model,
                        rows_of("noise"),
                        time_span.to(value_dtype),
                        rows_of("mu"),
                        stream["speaker_embeddings"][None].to(value_dtype),
                        rows_of("mel_conditioning"),
                        twin_rows,
                        model.row_attention(
                            twin_rows, streaming=True, dtype=value_dtype
                        ),
                        model.rope_angles(total),
                        cfg_rate=0.7,
                    )[0, emitted[name] : total]

                eager = whole_history(estimator, dtype)
                with torch.autocast("cuda", enabled=False):
                    float32 = whole_history(reference_estimator, torch.float32)
                actual = cached[
                    0, offset + emitted[name] - start : offset + total - start
                ]
                assert relative_error(
                    actual, float32
                ) <= ERROR_RATIO_OVER_BASELINE * relative_error(eager, float32), (
                    name,
                    total,
                )
                emitted[name] = total
                offset += total - start
    for pair in caches.values():
        release_rows(pool, list(pair))
    assert len(pool.free_blocks) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_layouts_upload_from_pinned_memory_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    pool = PrefixKVPool(
        layer_num=1,
        euler_steps=1,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=128 * BLOCK_FRAMES,
        device=device,
        dtype=torch.bfloat16,
    )
    rows = [PrefixCacheRow(), PrefixCacheRow()]
    assert grow_rows(pool, rows, [2048, 2048])
    layout = PrefixStepLayout(
        half_frames=4096,
        row_slots=1,
        segment_count=2 * (-(-4096 // CHUNK) + 2),
        page_table_width=4096,
    )

    uploads: list[tuple[bool, bool]] = []
    to = torch.Tensor.to

    def recording_to(
        tensor: torch.Tensor,
        target: torch.device | torch.dtype,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        moved = to(tensor, target, non_blocking=non_blocking)
        if tensor.device.type == "cpu" and moved.device.type == "cuda":
            uploads.append((tensor.is_pinned(), non_blocking))
        else:
            pass
        return moved

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    PrefixRowAttention(
        rows=rows,
        new_frames=[2048, 2048],
        layout=layout,
        padding_block=pool.padding_block,
        chunk_size=CHUNK,
        device=device,
    )
    assert uploads
    assert all(pinned and non_blocking for pinned, non_blocking in uploads)


def test_grow_rows_takes_nothing_on_a_shortfall() -> None:
    pool = PrefixKVPool.__new__(PrefixKVPool)
    pool.free_blocks = [0, 1, 2]
    rows = [PrefixCacheRow(), PrefixCacheRow()]
    assert not grow_rows(pool, rows, [BLOCK_FRAMES * 2, BLOCK_FRAMES * 2])
    assert pool.free_blocks == [0, 1, 2] and rows[0].blocks == []
    assert grow_rows(pool, rows, [BLOCK_FRAMES, BLOCK_FRAMES * 2])
    assert rows[0].allocated_frames == BLOCK_FRAMES
    assert rows[1].allocated_frames == BLOCK_FRAMES * 2
    assert pool.free_blocks == []
    release_rows(pool, rows)
    assert sorted(pool.free_blocks) == [0, 1, 2] and rows[0].committed_frames == 0


def test_float64_gemms_round_the_float64_product_once_in_every_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The swap reaches nn.Linear, eager or traced, and a row gets the same bits at every row count."""
    torch.manual_seed(0)
    linear = torch.nn.Linear(40, 48).bfloat16()
    activations = torch.randn(300, 40).bfloat16()
    expected = (
        activations.double() @ linear.weight.double().T + linear.bias.double()
    ).bfloat16()
    monkeypatch.setattr(F, "linear", float64_linear)
    traced = torch.compile(linear, backend="aot_eager", fullgraph=True)
    for rows in (1, 7, 300):
        assert torch.equal(linear(activations[:rows]), expected[:rows])
        assert torch.equal(linear(activations[None, :rows]), expected[None, :rows])
        assert torch.equal(traced(activations[:rows]), expected[:rows])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(linear(activations.float()), expected)


@pytest.mark.parametrize(
    ("actual_error", "baseline_error", "is_accepted"),
    [
        (
            MAX_RELATIVE_ERROR / 10 * ERROR_RATIO_OVER_BASELINE * 0.99,
            MAX_RELATIVE_ERROR / 10,
            True,
        ),
        (
            MAX_RELATIVE_ERROR / 10 * ERROR_RATIO_OVER_BASELINE * 1.2,
            MAX_RELATIVE_ERROR / 10,
            False,
        ),
        (MAX_RELATIVE_ERROR * 1.01, MAX_RELATIVE_ERROR * 1.01, False),
    ],
    ids=["within the ratio", "past the ratio", "past the ceiling"],
)
def test_the_error_bound_rejects_a_worse_path_and_an_error_both_share(
    actual_error: float, baseline_error: float, is_accepted: bool
) -> None:
    truth = torch.ones(100)
    arguments = (
        truth * (1 + actual_error),
        truth * (1 + baseline_error),
        truth,
        "synthetic",
    )
    if is_accepted:
        assert_as_accurate_as_baseline(*arguments)
    else:
        with pytest.raises(AssertionError):
            assert_as_accurate_as_baseline(*arguments)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_pool_leaves_its_reserve_under_the_allocator_cap() -> None:
    """A pool sized under an allocator cap took its padding block and the allocator's
    rounding out of its reserve, so the step it was sized beside failed to allocate."""
    device = torch.device("cuda", torch.cuda.current_device())
    attention = SimpleNamespace(heads=16, inner_dim=1024)
    flow = SimpleNamespace(
        decoder=SimpleNamespace(
            estimator=SimpleNamespace(
                transformer_blocks=[SimpleNamespace(attn=attention)] * 22,
                input_embed=SimpleNamespace(proj=SimpleNamespace(out_features=1024)),
            )
        )
    )
    reserve_bytes = 6 * 2**30
    gc.collect()
    torch.cuda.empty_cache()
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    entry_fraction = torch.cuda.get_per_process_memory_fraction(device)
    torch.cuda.set_per_process_memory_fraction(
        (torch.cuda.memory_reserved(device) + 8 * 2**30) / total_bytes, device
    )
    try:
        pool = build_prefix_pool(
            flow,
            device,
            torch.bfloat16,
            size_gb=None,
            max_blocks=None,
            stage_reserve_bytes=reserve_bytes,
            headroom_bytes=0,
            share_ceiling_bytes=None,
        )
        assert pool is not None
        step = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
        del step, pool
    finally:
        torch.cuda.set_per_process_memory_fraction(entry_fraction, device)
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_prefix", [False, True], ids=["eager", "compiled"])
def test_prefix_graph_replays_equal_the_eager_solve(
    compile_prefix: bool, gemm_mode: GemmMode
) -> None:
    """Replays with padding frames and unused row slots equal the eager solve,
    also when they resume from a step above the largest tier that ran eagerly and
    when a whole-history graph replays from the same pool between hops."""
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    pools = [
        PrefixKVPool(
            layer_num=LAYERS,
            euler_steps=10,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            capacity_frames=64 * BLOCK_FRAMES,
            device=device,
            dtype=dtype,
        )
        for _ in range(2)
    ]
    graph_pool, eager_pool = pools
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
    if gemm_mode == "tolerance":
        float64_estimator = make_float64_estimator(estimator, time_span)
    else:
        pass
    if compile_prefix:
        assert estimator.compile(dtype)
        for pool in pools:
            pool.forward = compile_forward_prefix()
    else:
        pass
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    graphs = SolveGraphCapture(
        backend,
        device=device,
        autocast_dtype=dtype,
        should_reduce_bf16_gemms_in_float32=False,
    )
    runner = PrefixCudaGraphRunner(
        estimator,
        graph_pool,
        graphs=graphs,
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=4,
        hop_frames=2 * CHUNK,
        min_hop_frames=CHUNK,
        max_frames=1024,
    )
    whole_history = WholeHistoryCudaGraphRunner(
        estimator,
        graphs=graphs,
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        euler_steps=10,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=2,
        tier_frames=(512,),
    )
    runner.capture()
    whole_history.capture()
    torch.manual_seed(3)
    whole_history_inputs = dict(
        noise=torch.randn(1, 500, CHANNELS, device=device),
        time_span=time_span,
        mu=torch.randn(1, 500, CHANNELS, device=device),
        speaker_embeddings=torch.randn(2, CHANNELS, device=device, dtype=dtype),
        mel_conditioning=torch.randn(1, 500, CHANNELS, device=device),
        lengths=(300, 200),
    )
    streams = {}
    for name, prompt_frames in (("a", 100), ("b", 50), ("c", 50)):
        mel_conditioning = torch.zeros(CHANNELS, 650, device=device, dtype=dtype)
        mel_conditioning[:, :prompt_frames] = torch.randn(
            CHANNELS, prompt_frames, device=device, dtype=dtype
        )
        streams[name] = {
            "noise": torch.randn(CHANNELS, 650, device=device, dtype=dtype),
            "mu": torch.randn(CHANNELS, 650, device=device, dtype=dtype),
            "mel_conditioning": mel_conditioning,
            "speaker_embeddings": torch.randn(CHANNELS, device=device, dtype=dtype),
        }
    caches = {
        name: [(PrefixCacheRow(), PrefixCacheRow()) for _ in pools] for name in streams
    }
    steps = [
        [("a", 150)],
        [("b", 100), ("a", 250)],
        [("a", 450), ("c", 70), ("b", 200)],
        [("c", 150), ("b", 400)],
        [("a", 600), ("b", 600), ("c", 300)],
        [("a", 650)],
    ]
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for step in steps:
            names = [name for name, _ in step]
            totals = [total for _, total in step]
            starts = [caches[name][0][0].committed_frames for name in names]
            new_frames = [total - start for start, total in zip(starts, totals)]

            def take(key: str) -> torch.Tensor:
                return (
                    torch.cat(
                        [
                            streams[name][key][:, start:total].transpose(0, 1)
                            for name, start, total in zip(names, starts, totals)
                        ]
                    )
                    .unsqueeze(0)
                    .float()
                )

            assert whole_history.run(**whole_history_inputs) is not None
            outputs = []
            for pool_index, pool in enumerate(pools):
                pairs = [caches[name][pool_index] for name in names]
                for pair, total in zip(pairs, totals):
                    assert grow_rows(pool, list(pair), [total, total])
                inputs = dict(
                    noise=take("noise"),
                    time_span=time_span,
                    mu=take("mu"),
                    speaker_embeddings=torch.stack(
                        [streams[name]["speaker_embeddings"] for name in names]
                    ),
                    mel_conditioning=take("mel_conditioning"),
                )
                if pool is graph_pool and sum(new_frames) <= runner.tier_frames[-1]:
                    generated = runner.run(
                        **inputs, new_frames=new_frames, caches=pairs
                    )
                elif pool is graph_pool:
                    assert (
                        runner.run(**inputs, new_frames=new_frames, caches=pairs)
                        is None
                    )
                    generated = solve_flow_euler_prefix(
                        estimator,
                        pool,
                        **inputs,
                        new_frames=new_frames,
                        caches=pairs,
                        cfg_rate=0.7,
                    )
                else:
                    generated = solve_flow_euler_prefix(
                        estimator,
                        pool,
                        **inputs,
                        new_frames=new_frames,
                        caches=pairs,
                        cfg_rate=0.7,
                    )
                outputs.append(generated)
            if gemm_mode == "tolerance":
                offset = 0
                for name, start, total in zip(names, starts, totals):
                    graph_rows, eager_rows = (
                        output[0, offset : offset + total - start] for output in outputs
                    )
                    stream = streams[name]
                    truth = solve_in_float64(
                        float64_estimator,
                        (total,),
                        noise=stream["noise"][:, :total].T[None],
                        time_span=time_span,
                        mu=stream["mu"][:, :total].T[None],
                        speaker_embeddings=stream["speaker_embeddings"][None],
                        mel_conditioning=stream["mel_conditioning"][:, :total].T[None],
                        streaming=True,
                    )
                    assert_as_accurate_as_baseline(
                        graph_rows,
                        eager_rows,
                        truth[0, start:total],
                        f"{step=} {name=}",
                    )
                    offset += total - start
            else:
                assert torch.equal(outputs[0], outputs[1]), step
            for name in names:
                for graph_row, eager_row in zip(*caches[name]):
                    assert graph_row.committed_frames == eager_row.committed_frames
                    if gemm_mode == "tolerance":
                        continue
                    else:
                        pass
                    assert torch.equal(graph_row.conv_context, eager_row.conv_context)
                    graph_pages = graph_row.pages(device)[: graph_row.committed_frames]
                    eager_pages = eager_row.pages(device)[: eager_row.committed_frames]
                    for euler_step in range(10):
                        for layer in range(LAYERS):
                            assert torch.equal(
                                graph_pool.keys[euler_step][layer][graph_pages],
                                eager_pool.keys[euler_step][layer][eager_pages],
                            )
                            assert torch.equal(
                                graph_pool.values[euler_step][layer][graph_pages],
                                eager_pool.values[euler_step][layer][eager_pages],
                            )
    for pair_per_pool in caches.values():
        for pool, pair in zip(pools, pair_per_pool):
            release_rows(pool, list(pair))
    assert all(len(pool.free_blocks) == 64 for pool in pools)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_graph_tiers_capture_over_one_set_of_input_buffers() -> None:
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=LAYERS,
        euler_steps=10,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        capacity_frames=8 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    runner = PrefixCudaGraphRunner(
        estimator,
        pool,
        graphs=SolveGraphCapture(
            backend,
            device=device,
            autocast_dtype=dtype,
            should_reduce_bf16_gemms_in_float32=False,
        ),
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=4,
        hop_frames=2 * CHUNK,
        min_hop_frames=CHUNK,
        max_frames=1024,
    )
    runner.capture()
    assert len({captured.layout.row_slots for captured in runner.captured}) > 1
    for field in (
        "noise",
        "mu",
        "mel_conditioning",
        "speaker_embeddings",
        "time_span",
        "context",
        "next_context",
    ):
        storages = {
            getattr(captured, field).untyped_storage().data_ptr()
            for captured in runner.captured
        }
        assert len(storages) == 1, field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_packed", [False, True], ids=["eager", "compiled"])
def test_whole_history_graph_replays_equal_the_eager_solve(
    compile_packed: bool, gemm_mode: GemmMode
) -> None:
    """Replays with empty row slots and a padding row, after other steps left their
    frames in the tier's buffers, equal the eager solve of the rows alone, also
    through a tier whose padding row is longer than any row. Steps above the
    largest tier or past the row slots are refused."""
    estimator = make_estimator()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
    if gemm_mode == "tolerance":
        float64_estimator = make_float64_estimator(estimator, time_span)
    else:
        pass

    def step_inputs(lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(sum(lengths))
        frames = (1, sum(lengths), CHANNELS)
        return dict(
            noise=torch.randn(frames, device=device, generator=generator),
            time_span=time_span,
            mu=torch.randn(frames, device=device, generator=generator),
            speaker_embeddings=torch.randn(
                len(lengths), CHANNELS, device=device, generator=generator
            ).to(dtype),
            mel_conditioning=torch.randn(frames, device=device, generator=generator),
        )

    def eager_solve(lengths: tuple[int, ...]) -> torch.Tensor:
        inputs = step_inputs(lengths)
        twin_rows = pack_rows(lengths * 2, device)
        return solve_flow_euler_packed(
            estimator,
            inputs["noise"],
            inputs["time_span"],
            inputs["mu"],
            inputs["speaker_embeddings"],
            inputs["mel_conditioning"],
            twin_rows,
            estimator.row_attention(twin_rows, streaming=False, dtype=dtype),
            estimator.rope_angles(twin_rows.width),
            cfg_rate=0.7,
        )

    if compile_packed:
        assert estimator.compile(dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for lengths in ((53,), (53, 78)):
                eager_solve(lengths)
    else:
        pass
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    runner = WholeHistoryCudaGraphRunner(
        estimator,
        graphs=SolveGraphCapture(
            backend,
            device=device,
            autocast_dtype=dtype,
            should_reduce_bf16_gemms_in_float32=False,
        ),
        frame_dtype=torch.float32,
        speaker_dtype=dtype,
        cfg_rate=0.7,
        euler_steps=10,
        mel_channels=CHANNELS,
        speaker_channels=CHANNELS,
        max_rows=3,
        tier_frames=(128, 256, 2048),
    )
    runner.capture()

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for lengths in (
            (40, 70),
            (100, 90, 50),
            (60,),
            (128,),
            (200, 56),
            (900, 900),
            (40, 70),
        ):
            inputs = step_inputs(lengths)
            replayed = runner.run(**inputs, lengths=lengths)
            if gemm_mode == "tolerance":
                truth = solve_in_float64(
                    float64_estimator, lengths, **inputs, streaming=False
                )
                assert_as_accurate_as_baseline(
                    replayed, eager_solve(lengths), truth, f"{lengths=}"
                )
            else:
                assert torch.equal(replayed, eager_solve(lengths)), lengths
        for lengths in ((900, 900, 400), (20, 20, 20, 20)):
            assert runner.run(**step_inputs(lengths), lengths=lengths) is None
