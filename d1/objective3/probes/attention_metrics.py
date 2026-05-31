"""
attention_metrics.py

Attention probes for LLaDA (and compatible models).

All functions expect an attention tensor of shape:
    attn_layer: [B, H, L, L]
where:
  - B = batch size
  - H = number of heads
  - L = sequence length
"""

from typing import Tuple
import math

import torch


def _get_gen_span(prompt_len: int, gen_length: int, seq_len: int) -> Tuple[int, int]:
    """
    Compute [start, end) indices of the generative region G.

    prompt_len: number of prompt tokens (non-editable region at the left).
    gen_length: number of generated tokens.
    seq_len:    total sequence length.
    """
    start = max(0, prompt_len)
    end = min(seq_len, prompt_len + gen_length)
    return start, end


def compute_bidirectionality_index(
    attn_layer: torch.Tensor,
    prompt_len: int,
    gen_length: int,
) -> torch.Tensor:
    """
    Bidirectionality Index (BI) for a single attention layer.

    For each generated token i in the generative region G:
      - average over heads → a row vector over all positions j
      - split row into:
          left_mass(i)  = sum_{j <= i} a[i, j]
          right_mass(i) = sum_{j >  i} a[i, j]
      - BI_i = right_mass(i) / (left_mass(i) + right_mass(i))

    We then average BI_i over all tokens in G.

    Args:
      attn_layer: [B, H, L, L] attention (already softmaxed).
      prompt_len: number of prompt tokens.
      gen_length: number of generated tokens.

    Returns:
      bi: [B] tensor, mean BI for each example.
    """
    assert attn_layer.dim() == 4, f"Expected [B, H, L, L], got {attn_layer.shape}"
    attn = attn_layer.float()
    B, H, L, _ = attn.shape
    eps = 1e-8

    start, end = _get_gen_span(prompt_len, gen_length, L)
    if start >= end:
        # No generative region; return zeros
        return attn.new_zeros(B)

    # Mean over heads → [B, L, L]
    attn_mean = attn.mean(dim=1)

    bi_per_example = []
    for b in range(B):
        rows = attn_mean[b, start:end, :]  # [G, L]
        if rows.numel() == 0:
            bi_per_example.append(attn.new_tensor(0.0))
            continue

        # Normalize rows over positions
        row_sums = rows.sum(dim=-1, keepdim=True) + eps
        rows = rows / row_sums

        bi_rows = []
        # rel_i ∈ [0, G), absolute index = start + rel_i
        for rel_i, row in enumerate(rows):
            pos_i = start + rel_i
            left_mass = row[: pos_i + 1].sum()
            right_mass = row[pos_i + 1 :].sum()
            total = left_mass + right_mass + eps
            bi_rows.append(right_mass / total)

        bi_rows = torch.stack(bi_rows, dim=0)  # [G]
        bi_per_example.append(bi_rows.mean())

    return torch.stack(bi_per_example, dim=0)  # [B]


def compute_attention_entropy(
    attn_layer: torch.Tensor,
    prompt_len: int,
    gen_length: int,
) -> torch.Tensor:
    """
    Attention entropy over positions for generated tokens.

    For each generated token i:
      - average over heads → distribution over all positions
      - compute H_i = -sum_j p_ij log p_ij
    and then average over all generated tokens.

    Args:
      attn_layer: [B, H, L, L]
      prompt_len: number of prompt tokens.
      gen_length: number of generated tokens.

    Returns:
      ent: [B] tensor, mean entropy for each example.
    """
    assert attn_layer.dim() == 4, f"Expected [B, H, L, L], got {attn_layer.shape}"
    attn = attn_layer.float()
    B, H, L, _ = attn.shape
    eps = 1e-8

    start, end = _get_gen_span(prompt_len, gen_length, L)
    if start >= end:
        return attn.new_zeros(B)

    attn_mean = attn.mean(dim=1)  # [B, L, L]
    ent_per_example = []

    for b in range(B):
        rows = attn_mean[b, start:end, :]  # [G, L]
        if rows.numel() == 0:
            ent_per_example.append(attn.new_tensor(0.0))
            continue

        row_sums = rows.sum(dim=-1, keepdim=True) + eps
        rows = rows / row_sums

        ent_rows = -(rows * (rows + eps).log()).sum(dim=-1)  # [G]
        ent_per_example.append(ent_rows.mean())

    return torch.stack(ent_per_example, dim=0)  # [B]


def compute_attention_localization_index(
    attn_layer: torch.Tensor,
    prompt_len: int,
    gen_length: int,
) -> torch.Tensor:
    """
    Attention Localization Index (ALI) — analogous to Flip Localization Index.

    We:
      1) Restrict to the generated region G in both rows and columns.
      2) Sum attention over heads and over rows (tokens in G) to get weights w_j
         for each column j in G.
      3) Normalize: p_j = w_j / sum_k w_k.
      4) Compute positional entropy H_pos = -sum_j p_j log p_j.
      5) ALI = 1 - H_pos / log(|G|).

    Low ALI  -> attention spread fairly uniformly.
    High ALI -> attention focused in a narrow corridor.

    Args:
      attn_layer: [B, H, L, L]
      prompt_len: number of prompt tokens.
      gen_length: number of generated tokens.

    Returns:
      ali: [B] tensor.
    """
    assert attn_layer.dim() == 4, f"Expected [B, H, L, L], got {attn_layer.shape}"
    attn = attn_layer.float()
    B, H, L, _ = attn.shape
    eps = 1e-8

    start, end = _get_gen_span(prompt_len, gen_length, L)
    G = end - start
    if G <= 1:
        return attn.new_zeros(B)

    # Restrict to G × G in rows & cols: [B, H, G, G]
    sub = attn[:, :, start:end, start:end]

    # Sum over heads and rows → [B, G]
    w = sub.sum(dim=(1, 2))

    # Normalize per example
    w_sum = w.sum(dim=-1, keepdim=True) + eps
    p = w / w_sum

    H_pos = -(p * (p + eps).log()).sum(dim=-1)  # [B]
    max_ent = math.log(G)
    ali = 1.0 - H_pos / max_ent

    return ali
