"""
T7 (EXPERIMENTS.md Part 3): our v4 Triton kernel vs PyTorch's built-in INT4 GEMM
(tinygemm, torch.ops.aten._weight_int4pack_mm), same quantized weights.

Conversion from QuantizedLinear buffers (W = (q - zp) * scale, low nibble = even k):
  - tinygemm wants uint8 [N, K/2] with the EVEN k in the HIGH nibble, then
    _convert_weight_to_int4pack(.., inner_k_tiles) for its tiled layout
  - its dequant is W = (q - 8) * scale + zero, so zero = (8 - zp) * scale;
    scales_and_zeros is [K/group, N, 2] bf16
  - activations / output are bf16 (the CUDA kernel has no fp16 path)
Because of bf16 the result is not bit-identical to v1; the script reports the error
against an fp32 reference next to cuBLAS's.

1. per layer: GPU time (do_bench) and host dispatch cost per call for fp16 F.linear,
   v4 (defaults) and tinygemm, M in {1, 8, 16}
2. --e2e: bs=1 decode with the decode-path Linear swapped to tinygemm (x.to(bf16)
   and .half() included), vs fp16 and v4

Usage:
    python tinygemm_bench.py --e2e
"""
import argparse
import csv
import datetime
import os
import statistics
import time

import torch
import torch.nn.functional as F
import triton.testing

import llama.quantize as quantize
from llama.quantize import QuantizedLinear, dequant_fn, int4_linear_fn

SHAPES = [(2048, 2048, "wq/wo"), (512, 2048, "wk/wv"), (8192, 2048, "w1/w3"),
          (2048, 8192, "w2"), (128256, 2048, "lm_head")]


def to_tinygemm(packed, scale, zp, group_size, inner_k_tiles):
    lo = packed & 0x0F            # even k
    hi = packed >> 4              # odd k
    tg = (lo << 4) | hi           # tinygemm: even k in the high nibble
    w_pack = torch.ops.aten._convert_weight_to_int4pack(tg.contiguous(), inner_k_tiles)
    s = scale.float()
    zero = (8.0 - zp.float()) * s
    sz = torch.stack([s, zero], dim=-1).transpose(0, 1).contiguous().to(torch.bfloat16)  # [K/g, N, 2]
    return w_pack, sz


def tinygemm(x, w_pack, sz, group_size):
    K = x.shape[-1]
    y = torch.ops.aten._weight_int4pack_mm(x.reshape(-1, K).to(torch.bfloat16), w_pack, group_size, sz)
    return y.to(torch.float16).reshape(*x.shape[:-1], -1)


def gpu_us(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


def host_us(fn, n=200):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / n * 1e6


def per_layer(a):
    v4 = int4_linear_fn()
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | inner_k_tiles {a.inner_k_tiles}")
    for N, K, name in SHAPES:
        if a.shapes and name not in a.shapes:
            continue
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        p, s, z = QuantizedLinear.quantize_tensor(w, 128)
        w16 = dequant_fn("v1")(p, s, z, 128)
        wp, sz = to_tinygemm(p, s, z, 128, a.inner_k_tiles)
        for M in (1, 8, 16):
            x = torch.randn(M, K, dtype=torch.float16, device="cuda")
            ref = F.linear(x.float(), w16.float())
            err = {"cublas": (F.linear(x, w16).float() - ref).abs().max().item(),
                   "v4": (v4(x, p, s, z, 128).float() - ref).abs().max().item(),
                   "tinygemm": (tinygemm(x, wp, sz, 128).float() - ref).abs().max().item()}
            t = {"fp16": gpu_us(lambda: F.linear(x, w16)), "v4": gpu_us(lambda: v4(x, p, s, z, 128)),
                 "tinygemm": gpu_us(lambda: tinygemm(x, wp, sz, 128)),
                 "tinygemm_kernel_only": gpu_us(lambda: torch.ops.aten._weight_int4pack_mm(
                     x.to(torch.bfloat16), wp, 128, sz))}
            h = {"fp16": host_us(lambda: F.linear(x, w16)), "v4": host_us(lambda: v4(x, p, s, z, 128)),
                 "tinygemm": host_us(lambda: tinygemm(x, wp, sz, 128))}
            print(f"{name:>8} N={N:>6} K={K:>5} M={M:>2} | GPU us: fp16 {t['fp16']:7.1f} v4 {t['v4']:7.1f} "
                  f"tinygemm {t['tinygemm']:7.1f} (kernel only {t['tinygemm_kernel_only']:7.1f}) | host us: fp16 {h['fp16']:.1f} "
                  f"v4 {h['v4']:.1f} tinygemm {h['tinygemm']:.1f} | max err vs fp32: cuBLAS {err['cublas']:.1e} "
                  f"v4 {err['v4']:.1e} tinygemm {err['tinygemm']:.1e}", flush=True)
            rows.append({"gpu": torch.cuda.get_device_name(0), "shape": name, "N": N, "K": K, "M": M,
                         **{f"{k}_us": round(v, 2) for k, v in t.items()},
                         **{f"host_{k}_us": round(v, 2) for k, v in h.items()},
                         **{f"err_{k}": v for k, v in err.items()},
                         "time": datetime.datetime.now().isoformat(timespec="seconds")})
        del w, p, s, z, w16, wp, sz
        torch.cuda.empty_cache()
    return rows


def e2e(a):
    import bench
    from llama.config import CKPT_DIR
    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    real_v4 = int4_linear_fn()
    cache = {}

    def tg_linear(x, p, s, z, g=128):
        key = p.data_ptr()
        if key not in cache:
            cache[key] = to_tinygemm(p, s, z, g, a.inner_k_tiles)
        return tinygemm(x, *cache[key], g)

    rows = []
    for label, mode, fn in [("fp16", "fp16", None), ("v4", "int4", real_v4), ("tinygemm", "int4", tg_linear)]:
        if fn is not None:
            quantize._int4_linear = fn
        torch.manual_seed(0)
        model = bench.build_model(sd, 1, a.input_len + a.output_len, mode, 128, "v4" if mode == "int4" else "v1")
        tokens = torch.randint(0, model.params.vocab_size, (1, a.input_len), device="cuda")
        for _ in range(3):
            bench.run_once(model, tokens, a.output_len)
        reps = [bench.run_once(model, tokens, a.output_len) for _ in range(10)]
        dec = statistics.median(s for _, d in reps for s in d)
        print(f"[e2e bs=1] {label:<9} decode {dec:.3f} ms/step", flush=True)
        rows.append({"config": label, "decode_ms": round(dec, 3), "time": datetime.datetime.now().isoformat(timespec="seconds")})
        del model, tokens
        cache.clear()
        torch.cuda.empty_cache()
    quantize._int4_linear = real_v4
    return rows


def write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e2e", action="store_true")
    p.add_argument("--shapes", nargs="*")
    p.add_argument("--inner-k-tiles", type=int, default=8)
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--csv", default="results/tinygemm.csv")
    p.add_argument("--e2e-csv", default="results/tinygemm_e2e.csv")
    a = p.parse_args()
    write(a.csv, per_layer(a))
    if a.e2e:
        write(a.e2e_csv, e2e(a))


if __name__ == "__main__":
    main()
