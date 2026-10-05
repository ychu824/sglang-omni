# SPDX-License-Identifier: Apache-2.0
"""Repetition-aware sampling: redraw a sampled token that already appears among
the latest generated tokens, as VALL-E 2 and upstream CosyVoice do."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sglang.srt.layers.sampler import sampling_from_probs_torch

# note (Yucheng Hu): the redraw reusing the first draw's seeded noise would be
# conditioned on the rejected candidate having won that draw.
REDRAW_SEED_SALT = 0x2545F491


def repetition_aware_redraw(
    probs: torch.Tensor,
    sampled_ids: torch.Tensor,
    output_ids: Sequence[Sequence[int]],
    window_size: int,
    redraw_allowed: torch.Tensor,
    sampling_seeds: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redraw each allowed row whose sampled token is among its last outputs.

    probs holds the full-vocabulary probabilities after penalties and
    temperature but before top-k/top-p, so masking the repeated token keeps a
    stop token reachable after truncation collapsed onto it. Returns the token
    ids and a mask of the redrawn rows.
    """
    recent_rows = [list(row_ids[-window_size:]) for row_ids in output_ids]
    recent_ids = torch.tensor(
        [row + [-1] * (window_size - len(row)) for row in recent_rows],
        dtype=torch.long,
    ).to(probs.device, non_blocking=True)
    candidate_ids = sampled_ids.long().unsqueeze(1)
    is_repeated = (recent_ids == candidate_ids).any(dim=1) & redraw_allowed
    redraw_probs = probs.to(torch.float32, copy=True)
    redraw_probs.scatter_(1, candidate_ids, 0.0)
    # note (Yucheng Hu): multinomial rejects an all-zero row, and a row whose
    # whole mass sat on the candidate has nothing else to draw.
    redraw_probs = torch.where(
        redraw_probs.sum(dim=1, keepdim=True) > 0, redraw_probs, probs
    )
    redraw_seeds = None if sampling_seeds is None else sampling_seeds ^ REDRAW_SEED_SALT
    redraw_ids = sampling_from_probs_torch(redraw_probs, redraw_seeds, positions)
    return torch.where(is_repeated, redraw_ids, sampled_ids), is_repeated


def conservative_repetition_aware_redraw(
    probs: torch.Tensor,
    sampled_ids: torch.Tensor,
    output_ids: Sequence[Sequence[int]],
    window_size: int,
    redraw_allowed: torch.Tensor,
    loop_class: torch.Tensor,
    stop_token_mask: torch.Tensor,
    stop_allowed: Sequence[bool],
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int,
    sampling_seeds: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experiment variant: mask, then truncate, then redraw.

    When the repeated candidate belongs to loop_class (a [V] bool mask), every
    loop_class id in the window is masked too, so a loop cannot hop between
    them. Stop tokens (stop_token_mask) are masked for rows whose stop_allowed
    is False. The request's top-k / top-p are then applied to what remains,
    with top-p measured against the remaining mass, and the redraw samples
    [B, max_top_k] instead of the whole vocabulary; a row with nothing left
    keeps its first draw.
    """
    recent_rows = [list(row_ids[-window_size:]) for row_ids in output_ids]
    host_rows = torch.tensor(
        [
            row + [-1] * (window_size - len(row)) + [allowed]
            for row, allowed in zip(recent_rows, stop_allowed)
        ],
        dtype=torch.long,
    ).to(probs.device, non_blocking=True)
    recent_ids, row_stop_allowed = host_rows[:, :-1], host_rows[:, -1].bool()
    candidate_ids = sampled_ids.long().unsqueeze(1)
    is_repeated = (recent_ids == candidate_ids).any(dim=1) & redraw_allowed
    safe_recent_ids = recent_ids.clamp_min(0)
    window_in_class = (
        loop_class[safe_recent_ids] & (recent_ids >= 0) & loop_class[candidate_ids]
    )
    # Entries outside the class point at the candidate, so every scattered
    # index is zeroed and duplicate indices cannot race.
    masked_ids = torch.where(window_in_class, safe_recent_ids, candidate_ids)
    redraw_probs = probs.to(torch.float32, copy=True)
    redraw_probs.scatter_(1, torch.cat([masked_ids, candidate_ids], dim=1), 0.0)
    redraw_probs.masked_fill_(
        stop_token_mask.unsqueeze(0) & ~row_stop_allowed.unsqueeze(1), 0.0
    )
    remaining = redraw_probs.sum(dim=1, keepdim=True)
    top_probs, top_ids = redraw_probs.topk(min(max_top_k, probs.shape[1]), dim=1)
    ranks = torch.arange(top_probs.shape[1], device=probs.device).unsqueeze(0)
    top_probs.masked_fill_(ranks >= top_ks.unsqueeze(1), 0.0)
    top_probs.masked_fill_(
        top_probs.cumsum(dim=1) - top_probs > top_ps.unsqueeze(1) * remaining, 0.0
    )
    has_mass = top_probs.sum(dim=1, keepdim=True) > 0
    # note (Yucheng Hu): multinomial rejects an all-zero row.
    top_probs = torch.where(has_mass, top_probs, 1.0)
    is_repeated &= has_mass.squeeze(1)
    redraw_seeds = None if sampling_seeds is None else sampling_seeds ^ REDRAW_SEED_SALT
    redraw_ranks = sampling_from_probs_torch(top_probs, redraw_seeds, positions)
    redraw_ids = top_ids.gather(1, redraw_ranks.long().unsqueeze(1)).squeeze(1)
    return (
        torch.where(is_repeated, redraw_ids.to(sampled_ids.dtype), sampled_ids),
        is_repeated,
    )
