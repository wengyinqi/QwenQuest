"""Attention math shared by every backend (reference implementations).

SGLang runs these with FlashInfer: a paged *prefill* kernel for EXTEND and a
paged *decode* kernel whose ``kv_indices`` list the tokens to read.  The
decode function below has the same semantics as FlashInfer decode with
``page_size = 1``: it attends over exactly the rows it is given, in order,
so a token listed twice is counted twice by the softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["decode_attention", "extend_attention"]


def extend_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_len: int,
    scaling: float,
) -> torch.Tensor:
    """Causal attention of one request's new tokens.

    Args:
      q: ``[n, H, D]`` queries of positions ``prefix_len .. prefix_len + n - 1``.
      k, v: ``[prefix_len + n, KV, D]`` keys / values of positions ``0 ..``.
    Returns:
      ``[n, H * D]``.
    """
    n, num_heads, head_dim = q.shape
    kv_len, kv_heads, _ = k.shape
    group = num_heads // kv_heads
    qh = q.transpose(0, 1).unsqueeze(0)  # [1, H, n, D]
    kh = k.repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
    vh = v.repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
    if prefix_len == 0 and n == kv_len:
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scaling)
    else:
        q_pos = prefix_len + torch.arange(n, device=q.device)
        mask = torch.arange(kv_len, device=q.device).unsqueeze(0) <= q_pos.unsqueeze(1)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scaling)
    return out[0].transpose(0, 1).reshape(n, num_heads * head_dim)


def decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaling: float
) -> torch.Tensor:
    """One query token against an explicit list of KV rows.

    Args:
      q: ``[H, D]``.
      k, v: ``[n, KV, D]`` - the selected tokens (Quest top-k, or all of them
        for dense decode).  Query head ``h`` reads KV head ``h // (H // KV)``.
    Returns:
      ``[H * D]`` in ``q.dtype`` (softmax and accumulation in float32).
    """
    num_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    qg = q.view(kv_heads, num_heads // kv_heads, head_dim).float()
    scores = torch.einsum("hgd,nhd->hgn", qg, k.float()) * scaling
    probs = scores.softmax(dim=-1)
    out = torch.einsum("hgn,nhd->hgd", probs, v.float())
    return out.reshape(num_heads * head_dim).to(q.dtype)
