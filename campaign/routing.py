"""Experimental route metadata kernel. Not installed into vLLM."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pack(order, ids, weights, tokens_out, weights_out, counts, SIZE: tl.constexpr,
          TOPK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    source = tl.load(order + i, valid, 0)
    expert = tl.load(ids + source, valid, 0)
    weight = tl.load(weights + source, valid, 0).to(tl.float16)
    tl.store(tokens_out + i, (source // TOPK).to(tl.int64), valid)
    tl.store(weights_out + i, weight, valid)
    tl.atomic_add(counts + expert, 1, valid, sem='relaxed')


def baseline(ids, weights, n_exp):
    tokens, topk = ids.shape
    local = ids.reshape(-1)
    flat_token = torch.arange(tokens, device=ids.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    counts = torch.zeros(n_exp + 1, dtype=torch.long, device=ids.device)
    counts.scatter_add_(0, local.long(), torch.ones(local.shape, dtype=torch.long, device=ids.device))
    return token_sorted, weight_sorted, counts


def candidate(ids, weights, n_exp):
    # Preconditions mirror flattened mapped local IDs used by the current overlay.
    assert ids.dtype == torch.int64 and ids.is_contiguous() and weights.is_contiguous()
    tokens, topk = ids.shape
    local = ids.reshape(-1)
    order = local.argsort()
    token_sorted = torch.empty(local.numel(), dtype=torch.long, device=ids.device)
    weight_sorted = torch.empty(local.numel(), dtype=torch.float16, device=ids.device)
    counts = torch.zeros(n_exp + 1, dtype=torch.long, device=ids.device)
    _pack[(triton.cdiv(local.numel(), 256),)](order, local, weights, token_sorted,
          weight_sorted, counts, local.numel(), topk, 256)
    return token_sorted, weight_sorted, counts
