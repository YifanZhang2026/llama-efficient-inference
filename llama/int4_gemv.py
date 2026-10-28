"""
v4 (EXPERIMENTS.md Part 3 / I3): fused INT4 dequant + GEMV in one Triton kernel.

    y[M, N] = x[M, K] @ W[N, K]^T,   W stored as QuantizedLinear buffers:
        packed [N, K/2] uint8 (low nibble = even k, high nibble = odd k)
        scale, zp [N, K/group] fp16,  W = (q - zp) * scale

v1-v3 write the whole fp16 W to HBM and cuBLAS reads it back (>= 4.5 B/weight).
Here every program reads its int4 tile once (0.5 B/weight + scale/zp), rebuilds W in
registers and multiplies right away; the fp16 W never exists in memory.

Design:
  - one program = BLOCK_N output columns; loops over K one quant group (BLOCK_K = group)
    at a time, so each row needs exactly one scale and one zp per iteration
  - no interleave: sum_k W[k] x[k] = lo-nibbles . x[0::2] + hi-nibbles . x[1::2]
  - dequant result is rounded to fp16 first, so the weights equal v1's bit for bit;
    only the fp32 accumulation order differs from cuBLAS
  - M == 1: broadcast multiply + tl.sum (tl.dot needs every dim >= 16)
    1 < M <= 16: pad M to 16 and use tl.dot (tensor cores)
  - split-K when N alone gives too few programs for the SMs (wk/wv: N = 512):
    partial sums go to an fp32 buffer with atomic_add, then one cast to fp16

Meant for decode (M = batch size). Prefill (M = bs * prompt_len) stays on v3.
"""
import torch
import triton
import triton.language as tl

MAX_M = 16


@triton.jit
def _int4_gemv_kernel(
    x_ptr, packed_ptr, scale_ptr, zp_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_pn, stride_sn, stride_ym,
    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    GROUPS_PER_SPLIT: tl.constexpr, ATOMIC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    HALF: tl.constexpr = GROUP // 2

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # output columns = rows of W
    offs_h = tl.arange(0, HALF)                            # byte index inside one group
    mask_n = offs_n < N

    if BLOCK_M == 1:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    else:
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    g0 = pid_k * GROUPS_PER_SPLIT
    for g in range(g0, g0 + GROUPS_PER_SPLIT):
        # int4 tile [BLOCK_N, GROUP/2] bytes, plus one scale / zp per row
        p = tl.load(packed_ptr + offs_n[:, None] * stride_pn + (g * HALF + offs_h)[None, :],
                    mask=mask_n[:, None], other=0)
        s = tl.load(scale_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
        z = tl.load(zp_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
        # same arithmetic as v1 ((q - zp) * scale in fp32, then fp16)
        w_lo = ((p & 0x0F).to(tl.float32) - z[:, None]) * s[:, None]
        w_hi = ((p >> 4).to(tl.float32) - z[:, None]) * s[:, None]
        w_lo = w_lo.to(tl.float16)
        w_hi = w_hi.to(tl.float16)

        k_even = g * GROUP + 2 * offs_h                    # x[k] for the low nibbles
        if BLOCK_M == 1:
            xe = tl.load(x_ptr + k_even).to(tl.float32)
            xo = tl.load(x_ptr + k_even + 1).to(tl.float32)
            acc += tl.sum(w_lo.to(tl.float32) * xe[None, :] + w_hi.to(tl.float32) * xo[None, :], axis=1)
        else:
            x_rows = x_ptr + offs_m[:, None] * stride_xm
            xe = tl.load(x_rows + k_even[None, :], mask=mask_m[:, None], other=0.0)
            xo = tl.load(x_rows + k_even[None, :] + 1, mask=mask_m[:, None], other=0.0)
            acc += tl.dot(xe, tl.trans(w_lo))
            acc += tl.dot(xo, tl.trans(w_hi))

    if BLOCK_M == 1:
        y_ptrs = y_ptr + offs_n
        mask = mask_n
    else:
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
    if ATOMIC:
        tl.atomic_add(y_ptrs, acc, mask=mask)
    else:
        tl.store(y_ptrs, acc.to(tl.float16), mask=mask)


_N_SM = {}


def _split_k(n_programs, n_groups, device):
    """Smallest power-of-2 split that gives >= 2 programs per SM, if K allows."""
    if device not in _N_SM:
        _N_SM[device] = torch.cuda.get_device_properties(device).multi_processor_count
    split = 1
    while n_programs * split < 2 * _N_SM[device] and n_groups % (2 * split) == 0 and n_groups // (2 * split) >= 2:
        split *= 2
    return split


def int4_linear(x, packed, scale, zp, group_size=128, block_n=128, split_k=None, num_warps=8):
    """x [..., K] fp16 with prod(...) <= 16  ->  [..., N] fp16.

    Defaults from a sweep over block_n {16..128} x num_warps {2,4,8} on RTX 5070 Laptop
    (results/int4_gemv_sweep_local_1025.log); re-check on A100.
    """
    N, half_k = packed.shape
    K = half_k * 2
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    assert M <= MAX_M and x.dtype == torch.float16 and K % group_size == 0
    n_groups = K // group_size

    block_m = 1 if M == 1 else 16
    n_programs = triton.cdiv(N, block_n)
    split = split_k or _split_k(n_programs, n_groups, x.device)
    atomic = split > 1

    if atomic:
        y = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    else:
        y = torch.empty(M, N, dtype=torch.float16, device=x.device)
    _int4_gemv_kernel[(n_programs, split)](
        x2, packed, scale, zp, y,
        M, N, K,
        x2.stride(0), packed.stride(0), scale.stride(0), y.stride(0),
        GROUP=group_size, BLOCK_N=block_n, BLOCK_M=block_m,
        GROUPS_PER_SPLIT=n_groups // split, ATOMIC=atomic,
        num_warps=num_warps,
    )
    if atomic:
        y = y.half()
    return y.reshape(*lead, N)
